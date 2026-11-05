"""
Upload the training bundle to Kaggle as a private dataset and start the training notebook on a GPU.

Needs a Kaggle API token in ~/.kaggle/access_token (or ~/.kaggle/kaggle.json) and a
phone-verified account (GPU and internet access in notebooks require it).

    python simrel/trades_kaggle/kaggle_push.py dataset   # upload ~/simdata/trades_bundle/trades_bundle.zip (private)
    python simrel/trades_kaggle/kaggle_push.py run       # push and start the notebook (GPU, internet on)
    python simrel/trades_kaggle/kaggle_push.py status
    python simrel/trades_kaggle/kaggle_push.py fetch     # download trades_checkpoint.zip to ~/simdata/trades_checkpoint
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = Path("~/simdata/kaggle_stage").expanduser()
KAGGLE = [str(Path(sys.executable).parent / "kaggle")]


def user() -> str:
    out = subprocess.run(KAGGLE + ["config", "view"], capture_output=True, text=True, check=True).stdout
    return next(line.split(":", 1)[1].strip() for line in out.splitlines() if "username" in line)


def sh(args, **kw):
    print("$ kaggle", " ".join(args), flush=True)
    return subprocess.run(KAGGLE + args, check=True, **kw)


if __name__ == "__main__":
    what, u = sys.argv[1], user()
    if what == "dataset":
        d = STAGE / "dataset"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        shutil.copy(Path("~/simdata/trades_bundle/trades_bundle.zip").expanduser(), d)
        (d / "dataset-metadata.json").write_text(json.dumps({"title": "trades-bundle", "id": f"{u}/trades-bundle", "licenses": [{"name": "other"}]}))
        sh(["datasets", "create", "-p", str(d)])  # private by default
    elif what == "run":
        k = STAGE / "kernel"
        shutil.rmtree(k, ignore_errors=True)
        k.mkdir(parents=True)
        shutil.copy(HERE / "train_trades_kaggle.ipynb", k)
        (k / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{u}/train-trades-intc", "title": "train-trades-intc", "code_file": "train_trades_kaggle.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{u}/trades-bundle"], "competition_sources": [], "kernel_sources": []}))
        sh(["kernels", "push", "-p", str(k)])
    elif what == "status":
        sh(["kernels", "status", f"{u}/train-trades-intc"])
    elif what == "fetch":
        out = Path("~/simdata/trades_checkpoint").expanduser()
        out.mkdir(exist_ok=True)
        sh(["kernels", "output", f"{u}/train-trades-intc", "-p", str(out)])
