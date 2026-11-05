"""
Phase 1 check: is the in-process MarS model behaving like the released one?

Scores the official validation samples (real order sequences with labels) with
each released model size. If the weights are loaded and fed correctly, the
next-order cross-entropy should be far below the uniform baseline log(49152)
and should fall as the model grows, as in the scaling curve in the MarS README.

    python simrel/experiments/pilot_mars_validation.py <MarS checkout> <data dir>
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simrel.adapters.mars_local import setup_mars

if __name__ == "__main__":
    setup_mars(sys.argv[1], sys.argv[2])
    import torch
    from market_simulation.models.order_model import OrderModel
    from market_simulation.utils import pkl_utils

    samples = pkl_utils.load_pkl_zstd(Path(sys.argv[2]) / "validation-samples/valid-00-00000000-0-32.zstd")
    f0, l0 = samples[0]
    print("samples:", len(samples), "features", np.asarray(f0).shape, "labels", np.asarray(l0).shape, np.asarray(l0).dtype)
    n = min(len(samples), 256)
    x = torch.from_numpy(np.stack([np.asarray(s[0], dtype=np.int32) for s in samples[:n]]))
    y = torch.from_numpy(np.stack([np.asarray(s[1]).reshape(-1) for s in samples[:n]]).astype(np.int64))
    out = {"uniform_nll": math.log(49152), "n_samples": n}
    for rid in ("Don-Don/mars-order-2m", "Don-Don/mars-order-5m", "Don-Don/mars-order-10m"):
        m = OrderModel.from_pretrained(rid).eval()
        nll, acc = [], []
        with torch.no_grad():
            for i in range(0, n, 16):
                logits = m(x[i : i + 16])
                yy = y[i : i + 16]
                if yy.shape[1] == logits.shape[1]:  # a label per position
                    lg, tg = logits.reshape(-1, logits.shape[-1]), yy.reshape(-1)
                else:  # a single next-order label
                    lg, tg = logits[:, -1, :], yy[:, -1]
                nll.append(torch.nn.functional.cross_entropy(lg, tg, reduction="none").numpy())
                acc.append((lg.argmax(1) == tg).float().numpy())
        out[rid] = {"nll": float(np.concatenate(nll).mean()), "top1": float(np.concatenate(acc).mean())}
        print(rid, out[rid], flush=True)
    (REPO_ROOT / "simrel" / "reports" / "pilot_mars_validation.json").write_text(json.dumps(out, indent=2))
