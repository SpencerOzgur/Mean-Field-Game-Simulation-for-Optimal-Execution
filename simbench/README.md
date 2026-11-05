# simbench: do market simulators agree on execution decisions?

This package runs one execution agent, with one task definition and one cost
accounting, inside several simulated markets, and compares the decisions each
market recommends. It implements the study in
`abides_marl_trades_research_plan.md`; the write-up is in `paper/`.

## What is here

| Path | Purpose |
| --- | --- |
| `abides_world.py` | Six ABIDES background configurations, the execution agent, one-episode runner |
| `policies.py` | Schedule family, immediate, POV, urgency rule |
| `mfg_policy.py` | Closed-form mean-field-game liquidation schedule |
| `runner.py` | Parallel, resumable episode runner with a per-episode time limit |
| `analysis.py` | Paired-twin cost accounting and bootstrap intervals |
| `pomegranate_shim.py` | Stand-in for the old `pomegranate` release ABIDES imports |
| `trades/` | TRADES (DeepMarket) arm: retraining script, patch, episode and batch runners |
| `../experiments/simbench_0*.py` | Experiment stages, in order |
| `../paper/make_tables.py` | Turns result files into the paper's tables and numbers |

## Setup

Python 3.9 or newer, running natively (on Apple Silicon, not under Rosetta).

```bash
python3 -m venv ~/.venvs/simbench && source ~/.venvs/simbench/bin/activate
pip install numpy pandas scipy matplotlib tqdm coloredlogs psutil
git clone https://github.com/jpmorganchase/abides-jpmc-public.git   # commit f9cbe51
pip install --no-deps -e abides-jpmc-public/abides-core -e abides-jpmc-public/abides-markets
```

## Running the ABIDES study

From the repository root:

```bash
python experiments/simbench_00_baselines.py        # gates + 200 unperturbed twins per world
python experiments/simbench_01_policies.py 40      # every policy, 3 sizes, 40 seeds
python experiments/simbench_02_analyze.py          # tables and figures -> results/simbench/
python experiments/simbench_03_short_horizon.py 40 # 10-minute task used for the TRADES comparison
```

Stages are resumable: rerun the same command and finished episodes are skipped.
On a laptop, keep the machine awake (`caffeinate -ims` on macOS); if it sleeps,
episodes freeze for minutes at a time.

## Running the TRADES arm

The authors' checkpoints are no longer downloadable, so the model is retrained
on the public LOBSTER sample day that ships with DeepMarket.

```bash
pip install torch torchvision einops lightning lion_pytorch torch_ema scikit-learn seaborn joblib shapely wandb
git clone https://github.com/LeonardoBerti00/DeepMarket.git && cd DeepMarket   # commit 8f1f89b
git apply /path/to/simbench/trades/deepmarket_simbench.patch
(cd data/INTC && unzip INTC.zip)
cp /path/to/simbench/trades/train_trades_intc.py .
python train_trades_intc.py mps 100        # device, minutes; writes data/checkpoints/TRADES/*.ckpt
cd /path/to/this/repo
python simbench/trades/run_trades_batch.py --deepmarket /path/to/DeepMarket --ckpt <ckpt> --seeds 6 --procs 3
python experiments/simbench_05_real_reference.py /path/to/DeepMarket
python experiments/simbench_04_trades_analyze.py
```

## Building the paper

```bash
python paper/make_tables.py
cd paper && tectonic main.tex      # or upload the paper/ folder to Overleaf
```

## Not included

- ABIDES-MARL: no public implementation was found.
- The authors' TRADES checkpoints and their licensed training data.
- Any real-market validation beyond one public sample day.
