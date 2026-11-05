"""
Train TRADES (DeepMarket) on several LOBSTER-format days of one stock.

Run from the root of a DeepMarket checkout (commit 8f1f89b) with the simbench
patch applied. Data must be in

    data/<SYMBOL>/<SYMBOL>_<first date>_<last date>/<SYMBOL>_<date>_34200000_57600000_{message,orderbook}_10.csv

Days are split chronologically into train / validation / test by --split
(counts of days). The same split and date range are written into
constants.py so that DeepMarket's simulation code later normalises its inputs
with exactly the training statistics.

    python train_trades_multiday.py --symbol INTC --split 5,1,2 --minutes 600

The best checkpoint (lowest validation loss) is kept in data/checkpoints/TRADES/.
Training stops by itself after --minutes so a hosted notebook can save its outputs.
"""

import argparse
import os
import random
import re
import sys
import time
from datetime import timedelta
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="INTC")
    ap.add_argument("--split", default="5,1,2", help="train,validation,test day counts")
    ap.add_argument("--minutes", type=float, default=600)
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--limit-train-batches", type=int, default=None, help="for smoke tests")
    ap.add_argument("--val-batches", type=int, default=40)
    ap.add_argument("--val-interval", type=float, default=0.5, help="fraction of an epoch between validations")
    ap.add_argument("--seed", type=int, default=None, help="overrides constants.SEED (a second training run for robustness)")
    args = ap.parse_args()

    root = Path.cwd()
    folders = sorted((root / "data" / args.symbol).glob(f"{args.symbol}_*_*"))
    folders = [f for f in folders if f.is_dir()]
    assert len(folders) == 1, f"expected one data folder under data/{args.symbol}, found {folders}"
    first, last = folders[0].name.split("_")[1:3]
    n_days = len(list(folders[0].glob("*_message_10.csv")))
    tr, va, te = (int(x) for x in args.split.split(","))
    assert tr + va + te == n_days, f"split {args.split} does not add up to the {n_days} days present"
    rates = (tr / n_days + 1e-9, va / n_days + 1e-9, te / n_days - 2e-9)  # DeepMarket truncates rate * days to an integer

    # keep constants.py in step with this run (date range and day split)
    cpath = root / "constants.py"
    src = cpath.read_text()
    src = re.sub(r"^DATE_TRADING_DAYS = .*$", f'DATE_TRADING_DAYS = ["{first}", "{last}"]', src, flags=re.M)
    src = re.sub(r"^SPLIT_RATES = .*$", f"SPLIT_RATES = ({rates[0]!r}, {rates[1]!r}, {rates[2]!r})", src, flags=re.M)
    cpath.write_text(src)

    sys.path.insert(0, str(root))
    import torch

    import constants as cst

    if args.seed is not None:
        cst.SEED = args.seed
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    cst.DEVICE = device

    import lightning as L
    import numpy as np
    from torch.utils.data import DataLoader

    import configuration
    from models.diffusers.diffusion_engine import DiffusionEngine
    from models.diffusers.TRADES.TRADES_hparam import HP_TRADES_FIXED
    from preprocessing.LOBDataset import LOBDataset
    from preprocessing.LOBSTERDataBuilder import LOBSTERDataBuilder

    torch.manual_seed(cst.SEED)
    np.random.seed(cst.SEED)
    random.seed(cst.SEED)
    torch.set_float32_matmul_precision("high")

    config = configuration.Configuration()
    config.CHOSEN_STOCK = [cst.Stocks[args.symbol]]
    config.SPLIT_RATES = rates
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_NORM] = 2
    for p in cst.LearningHyperParameter:
        if p.value in HP_TRADES_FIXED:
            config.HYPER_PARAMETERS[p] = HP_TRADES_FIXED[p.value]
    config.FILENAME_CKPT = f"{args.symbol}_{first}_{last}_train{tr}d_seed_{cst.SEED}"

    ddir = root / "data" / args.symbol
    if not (ddir / "train.npy").exists():
        LOBSTERDataBuilder(args.symbol, cst.DATA_DIR, cst.DATE_TRADING_DAYS, rates, config.CHOSEN_MODEL).prepare_save_datasets()

    seq = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]
    gen = config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE]
    bs = config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE]
    tbs = config.HYPER_PARAMETERS[cst.LearningHyperParameter.TEST_BATCH_SIZE]
    train_set = LOBDataset([str(ddir / "train.npy")], seq, gen, config.CHOSEN_MODEL, is_val=False)
    val_set = LOBDataset([str(ddir / "val.npy")], seq, gen, config.CHOSEN_MODEL, is_val=True, batch_size=tbs, limit_val_batches=args.val_batches)
    print(f"device {device} | days train/val/test {tr}/{va}/{te} | train rows {tuple(train_set.data.size())} | val rows {tuple(val_set.data.size())} | "
          f"steps per epoch {len(train_set) // bs}", flush=True)
    workers = 2 if device == "cuda" else 0
    train_dl = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True, num_workers=workers, persistent_workers=workers > 0)
    val_dl = DataLoader(val_set, batch_size=tbs, shuffle=False, num_workers=0)

    os.makedirs("data/checkpoints/TRADES", exist_ok=True)
    model = DiffusionEngine(config)
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)

    class Progress(L.Callback):
        def __init__(self):
            self.t0 = time.time()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step % 500 == 0:
                rate = trainer.global_step / max(time.time() - self.t0, 1e-9)
                print(f"step {trainer.global_step} | epoch {trainer.current_epoch} | {rate:.2f} steps/s | {(time.time() - self.t0) / 60:.0f} min", flush=True)

    trainer = L.Trainer(
        accelerator={"cuda": "gpu", "mps": "mps", "cpu": "cpu"}[device], devices=1, precision=cst.PRECISION, max_epochs=50,
        max_time=timedelta(minutes=args.minutes), limit_train_batches=args.limit_train_batches,
        num_sanity_val_steps=0, check_val_every_n_epoch=1, val_check_interval=args.val_interval if args.limit_train_batches is None else 1.0,
        enable_checkpointing=False, logger=False, enable_progress_bar=False, callbacks=[Progress()],
    )
    t0 = time.time()
    trainer.fit(model, train_dl, val_dl)
    print(f"finished: {trainer.global_step} steps in {(time.time() - t0) / 60:.1f} minutes", flush=True)
    print("checkpoints:", os.listdir("data/checkpoints/TRADES"), flush=True)


if __name__ == "__main__":
    main()
