"""
One execution episode in a TRADES (DeepMarket) market.

The background market is DeepMarket's WorldAgent: it replays the first 15
minutes of a real LOBSTER day and then generates every subsequent order with
the TRADES diffusion model, conditioned on the live order book, inside
DeepMarket's bundled (legacy) ABIDES exchange. This script adds one execution
agent with the same task semantics as simbench/abides_world.py: market orders
at fixed decision times following a schedule, with the remainder forced in the
last slice.

It must run against a DeepMarket checkout (it imports that repository's code):

    python run_trades_episode.py --deepmarket /path/to/DeepMarket --ckpt <ckpt> \
        --seed 0 --side sell --qty 20000 --policy exp:2 --out ep.pkl

--policy is "exp:<b>" (w_k ∝ exp(-b k/n), b = 0 is TWAP) or "immediate".
--qty 0 gives the unperturbed twin episode used for paired costs.
"""

import argparse
import os
import pickle
import sys
import time
import warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepmarket", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--side", default="sell")
    ap.add_argument("--qty", type=int, default=0)
    ap.add_argument("--policy", default="exp:0")
    ap.add_argument("--symbol", default="INTC")
    ap.add_argument("--date", default="2012-06-21")
    ap.add_argument("--start", default="09:46:00")
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--slices", type=int, default=20)
    ap.add_argument("--pre", type=int, default=60)
    ap.add_argument("--post", type=int, default=120)
    ap.add_argument("--obs-dt", type=int, default=5)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--ddim-steps", type=int, default=1, help="diffusion sampling steps per generated order (DeepMarket's default is 10)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    root = os.path.abspath(args.deepmarket)
    ckpt = os.path.abspath(args.ckpt)
    out = os.path.abspath(args.out)
    os.chdir(root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "ABIDES"))

    import numpy as np
    import pandas as pd
    import torch

    import constants as cst

    cst.DEVICE = args.device
    torch.set_num_threads(1)
    if args.device == "cuda":  # same seed must give the same path, so twins stay identical until the first child order
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    from Kernel import Kernel
    from agent.ExchangeAgent import ExchangeAgent
    from agent.TradingAgent import TradingAgent
    from agent.WorldAgent import WorldAgent
    from models.diffusers.diffusion_engine import DiffusionEngine
    from utils.utils_data import load_compute_normalization_terms

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    day = pd.to_datetime(args.date)
    mkt_open = day + pd.to_timedelta("09:30:00")
    start = day + pd.to_timedelta(args.start)
    dt_slice = pd.to_timedelta(args.horizon / args.slices, unit="s")
    obs_dt = pd.to_timedelta(args.obs_dt, unit="s")
    obs_start = start - pd.to_timedelta(args.pre, unit="s")
    obs_end = start + pd.to_timedelta(args.horizon + args.post, unit="s")
    mkt_close = obs_end + pd.to_timedelta(5, unit="s")

    if args.policy == "immediate":
        weights = np.zeros(args.slices)
        weights[0] = 1.0
    else:
        b = float(args.policy.split(":")[1])
        weights = np.exp(-b * np.arange(args.slices) / args.slices)
        weights /= weights.sum()
    cum = np.cumsum(weights)

    class ScheduledExecutionAgent(TradingAgent):
        def __init__(self, id, random_state):
            super().__init__(id, "EXECUTION_AGENT", "ExecutionAgent", starting_cash=10**13, log_orders=False, random_state=random_state)
            self.symbol = args.symbol
            self.k = 0
            self.filled = 0
            self.obs, self.fills, self.slices = [], [], []
            self.awaiting = False
            self.vol_requested = False
            self.window_volume = None

        def getWakeFrequency(self):
            return obs_start - self.mkt_open

        def wakeup(self, currentTime):
            if not super().wakeup(currentTime) or currentTime < obs_start:
                return
            self.awaiting = True
            self.getCurrentSpread(self.symbol, depth=5)
            if currentTime >= start + pd.to_timedelta(args.horizon, unit="s") and not self.vol_requested:
                self.vol_requested = True
                self.get_transacted_volume(self.symbol, lookback_period=f"{args.horizon}s")
            if currentTime + obs_dt <= obs_end:
                self.setWakeup(currentTime + obs_dt)

        def receiveMessage(self, currentTime, msg):
            super().receiveMessage(currentTime, msg)
            kind = msg.body["msg"]
            if kind == "QUERY_TRANSACTED_VOLUME" and self.window_volume is None:
                self.window_volume = self.transacted_volume.get(self.symbol)
            if kind != "QUERY_SPREAD" or not self.awaiting:
                return
            self.awaiting = False
            bids = self.known_bids.get(self.symbol) or []
            asks = self.known_asks.get(self.symbol) or []
            bid = bids[0][0] if bids else np.nan
            ask = asks[0][0] if asks else np.nan
            t_rel = (currentTime - start).total_seconds()
            self.obs.append((t_rel, bid, ask))
            if t_rel < 0 or self.k >= args.slices or currentTime < start + self.k * dt_slice:
                return
            remaining = args.qty - self.filled
            if remaining <= 0:
                qty = 0
            elif self.k == args.slices - 1:
                qty = remaining
            else:
                qty = int(np.clip(round(args.qty * cum[self.k] - self.filled), 0, remaining))
            self.slices.append((self.k, t_rel, 0.5 * (bid + ask), qty))
            if qty > 0:
                self.placeMarketOrder(self.symbol, qty, is_buy_order=(args.side == "buy"))
            self.k += 1

        def orderExecuted(self, order):
            super().orderExecuted(order)
            self.filled += order.quantity
            self.fills.append(((self.currentTime - start).total_seconds(), order.fill_price, order.quantity))

    checkpoint = torch.load(ckpt, map_location=args.device, weights_only=False)
    config = checkpoint["hyper_parameters"]["config"]
    config.IS_WANDB = False
    config.CHOSEN_MODEL = cst.Models.TRADES
    config.SAMPLING_TYPE = "DDIM"
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.DDIM_ETA] = 0.0
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.DDIM_NSTEPS] = args.ddim_steps
    model = DiffusionEngine.load_from_checkpoint(ckpt, config=config, map_location=args.device, weights_only=False).to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    seq = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]
    gen = config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE]
    norm = load_compute_normalization_terms(args.symbol, cst.DATA_DIR, cst.Models.TRADES, n_lob_levels=10)

    rs = lambda: np.random.RandomState(seed=args.seed)
    exchange = ExchangeAgent(id=0, name="EXCHANGE_AGENT", type="ExchangeAgent", mkt_open=mkt_open, mkt_close=mkt_close,
                             symbols=[args.symbol], log_orders=False, pipeline_delay=0, computation_delay=0,
                             stream_history=2500000, book_freq=None, wide_book=False, random_state=rs())
    world = WorldAgent(id=1, name="WORLD_AGENT", type="WorldAgent", symbol=args.symbol, date=str(day.date()),
                       date_trading_days=cst.DATE_TRADING_DAYS, model=model, data_dir=cst.DATA_DIR,
                       cond_type=config.COND_TYPE, cond_seq_size=seq - gen,
                       size_type_emb=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_TYPE_EMB],
                       log_orders=False, random_state=rs(), normalization_terms=norm, using_diffusion=True,
                       chosen_model="TRADES", gen_seq_size=gen)
    ex = ScheduledExecutionAgent(2, rs())

    kernel = Kernel("simbench TRADES", random_state=rs())
    t0 = time.time()
    err = None
    try:
        kernel.runner(agents=[exchange, world, ex], startTime=mkt_open, stopTime=mkt_close + pd.to_timedelta("00:00:01"),
                      defaultComputationDelay=0, skip_log=True,
                      log_dir=f"world_agent_{args.symbol}_{args.date}_simbench_{args.seed}_{os.getpid()}")
    except Exception as e:  # DeepMarket's kernel post-processes log files we do not write
        err = repr(e)

    rec = {
        "world": "T1_trades_intc", "seed": args.seed, "side": args.side, "total_qty": args.qty, "policy": args.policy,
        "horizon_s": args.horizon, "n_slices": args.slices, "ddim_steps": args.ddim_steps, "wall_s": time.time() - t0, "post_run_error": err,
        "obs": np.array(ex.obs, dtype=np.float64), "fills": np.array(ex.fills, dtype=np.float64).reshape(-1, 3),
        "slices": np.array(ex.slices, dtype=np.float64).reshape(-1, 4), "window_volume": ex.window_volume,
        "generated_orders": int(world.count_diff_placed_orders), "ignored_cancels": int(world.ignored_cancel),
    }
    book = exchange.order_books[args.symbol]
    if hasattr(book, "tape"):  # order-level record (seconds after midnight), for conversion to LOBSTER format
        midnight = pd.Timestamp(day.date())
        rec["tape"] = [((t - midnight).value / 1e9, kind, int(oid), bool(is_buy), int(price), int(qty)) for t, kind, oid, is_buy, price, qty in book.tape]
        rec["task_start_s"] = (pd.Timestamp(f"{args.date} {args.start}") - midnight).total_seconds()
    with open(out, "wb") as f:
        pickle.dump(rec, f)
    print("done", args.policy, args.seed, "wall", round(rec["wall_s"]), "obs", len(ex.obs), "fills", len(ex.fills),
          "filled", ex.filled, "vol", ex.window_volume, "err", err, flush=True)


if __name__ == "__main__":
    main()
