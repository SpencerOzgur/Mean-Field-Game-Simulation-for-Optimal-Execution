"""
Retrain TRADES on the single public LOBSTER sample day bundled with DeepMarket
(INTC, 2012-06-21). The authors' checkpoints are no longer downloadable, and
the stock day-based split cannot handle a one-day dataset, so the day is split
chronologically (85/5/10) instead. Everything else uses the repository's code
and its fixed TRADES hyperparameters.

usage: python train_intc.py <device: mps|cpu> <max_minutes> [limit_train_batches]
"""
import os
import sys

import constants as cst

DEVICE = sys.argv[1]
MAX_MIN = float(sys.argv[2])
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
cst.DEVICE = DEVICE

import random
from datetime import timedelta

import lightning as L
import numpy as np
import pandas as pd
import torch

import configuration
from models.diffusers.diffusion_engine import DiffusionEngine
from models.diffusers.TRADES.TRADES_hparam import HP_TRADES_FIXED
from preprocessing.DataModule import DataModule
from preprocessing.LOBDataset import LOBDataset
from preprocessing.LOBSTERDataBuilder import LOBSTERDataBuilder
from utils.utils_data import preprocess_data


class OneDayBuilder(LOBSTERDataBuilder):
    def _prepare_dataframes(self, path):
        ob_cols = [f"{s}{i}" for i in range(1, 11) for s in ("sell", "vsell", "buy", "vbuy")]
        msg_cols = ["time", "event_type", "order_id", "size", "price", "direction"]
        files = sorted(os.listdir(path))
        messages = pd.read_csv(os.path.join(path, files[0]), names=msg_cols)
        orderbooks = pd.read_csv(os.path.join(path, files[1]), names=ob_cols)
        orderbooks, messages = preprocess_data([messages, orderbooks], self.n_lob_levels, self.chosen_model)
        n = len(messages)
        a, b = int(n * self.split_rates[0]), int(n * (self.split_rates[0] + self.split_rates[1]))
        for lo, hi in ((0, a), (a, b), (b, n)):
            self.dataframes.append([messages.iloc[lo:hi].copy(), orderbooks.iloc[lo:hi].copy()])
        print("rows train/val/test:", a, b - a, n - b)
        self._normalize_dataframes_TRADES()


if __name__ == "__main__":
    torch.manual_seed(cst.SEED); np.random.seed(cst.SEED); random.seed(cst.SEED)
    torch.set_float32_matmul_precision("high")
    config = configuration.Configuration()
    config.CHOSEN_STOCK = [cst.Stocks.INTC]
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_NORM] = 2
    for p in cst.LearningHyperParameter:
        if p.value in HP_TRADES_FIXED:
            config.HYPER_PARAMETERS[p] = HP_TRADES_FIXED[p.value]
    config.FILENAME_CKPT = "INTC_retrained_oneday_seed_%d" % cst.SEED

    if not os.path.exists("data/INTC/train.npy"):
        OneDayBuilder("INTC", cst.DATA_DIR, cst.DATE_TRADING_DAYS, config.SPLIT_RATES, config.CHOSEN_MODEL).prepare_save_datasets()

    seq = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]
    gen = config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE]
    train_set = LOBDataset(["data/INTC/train.npy"], seq, gen, config.CHOSEN_MODEL, is_val=False)
    val_set = LOBDataset(["data/INTC/val.npy"], seq, gen, config.CHOSEN_MODEL, is_val=True,
                         batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.TEST_BATCH_SIZE], limit_val_batches=4)
    print("train", tuple(train_set.data.size()), "val", tuple(val_set.data.size()), flush=True)
    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_set, batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE], shuffle=True, drop_last=True, num_workers=0)
    val_dl = DataLoader(val_set, batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.TEST_BATCH_SIZE], shuffle=False, num_workers=0)
    os.makedirs("data/checkpoints/TRADES", exist_ok=True)
    model = DiffusionEngine(config)
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    trainer = L.Trainer(
        accelerator=DEVICE, devices=1, precision=cst.PRECISION, max_epochs=50,
        max_time=timedelta(minutes=MAX_MIN), limit_train_batches=LIMIT,
        num_sanity_val_steps=0, check_val_every_n_epoch=1,
        val_check_interval=(0.2 if LIMIT is None else 1.0),
        enable_checkpointing=False, logger=False, enable_progress_bar=False,
    )
    import time; t0 = time.time()
    trainer.fit(model, train_dl, val_dl)
    print("done: steps", trainer.global_step, "minutes", round((time.time() - t0) / 60, 1), flush=True)
    print(os.listdir("data/checkpoints/TRADES"))
