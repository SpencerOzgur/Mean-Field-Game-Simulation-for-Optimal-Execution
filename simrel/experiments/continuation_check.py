"""
Continuation check (simrel/reports/continuation_check_plan.md). Exploratory; uses already-read markets.

    python simrel/experiments/continuation_check.py timing            one passive episode per kind of market, timed
    python simrel/experiments/continuation_check.py passive [procs]   passive episodes (20 seeds) for every eligible market
    python simrel/experiments/continuation_check.py classify          per-market S3 and S4 in ticks, classification,
                                                                      counts and the feasibility stop (no regret read)
    python simrel/experiments/continuation_check.py headroom          always-TWAP full-truth regret in the low subset only
    python simrel/experiments/continuation_check.py compare           the comparisons of plan section 6
    python simrel/experiments/continuation_check.py figure            regret minus TWAP against S4

The stages are separate so that no regret by group is read before the feasibility and headroom checks. The passive
episode is reversal_check.passive_job's loop with the world built by reference_worlds.data.make; the statistics are
simrel/metrics/reversal.py and reversal_check.describe, unchanged.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.eventsim import GRID_DT  # noqa: E402
from simrel.experiments import reversal_check as RC  # noqa: E402
from simrel.lob import Book  # noqa: E402
from simrel.metrics import reversal as RV  # noqa: E402
from simrel.reference_worlds import ablation, data  # noqa: E402

OUT = ROOT / "results" / "simrel" / "goal2" / "continuation_check"
RAW = OUT / "raw"
G2 = ROOT / "results" / "simrel" / "goal2"
THRESH, THRESH_SENS, MIN_LOW, HEADROOM_BP = -0.05, -0.10, 8, 0.5
N_BOOT = 2000
FAM_TB = {"q": "queue-reactive", "a": "agent-mix", "l": "latent"}
DEV_STORED = [f"R1{c}" for c in "abcde"] + [f"R2{c}" for c in "abcde"]


# ---- markets
def testb_markets() -> dict:
    acc = json.loads((ROOT / "results" / "simrel" / "raw" / "testb_markets.json").read_text())["accepted"]
    return {draw: FAM_TB[draw[2]] for draw in acc.values()}


def dev_markets() -> dict:
    d = pd.read_csv(G2 / "addendum10" / "dev_fourway.csv")
    out = {}
    for w in sorted(d[d["core"]]["world"].unique()):
        if w.startswith("W"):  # Hawkes-built: needs the fitted Hawkes pickles (results/simrel/goal2/raw), not on this machine
            continue
        out[w] = "queue-reactive" if w.startswith("R1") else "agent-mix"
    return out


def ablation_markets() -> dict:
    return {k: "queue-reactive" if fam == "qr" else "agent-mix" for k, (fam, _, _, _) in ablation.markets().items()}


def all_markets() -> pd.DataFrame:
    rows = [{"market": m, "family": f, "source": "Test B"} for m, f in testb_markets().items()]
    rows += [{"market": m, "family": f, "source": "development"} for m, f in dev_markets().items()]
    rows += [{"market": m, "family": f, "source": "ablation (descriptive)"} for m, f in ablation_markets().items()]
    return pd.DataFrame(rows)


# ---- passive episodes (reversal_check.passive_job, with the world from data.make)
def passive_job(args):
    market, seed = args
    warnings.filterwarnings("ignore")
    model = data.make(market)(seed)
    book, tape, t = Book(), [], 0.0
    model.init_book(book)
    bid, ask = [], []
    end = RC.WARMUP_S + RC.EPISODE_S
    while t < end + 1e-9:
        if t >= RC.WARMUP_S - 1e-9:
            bb, ba = book.best_bid(), book.best_ask()
            bid.append(float(bb) if bb is not None else np.nan)
            ask.append(float(ba) if ba is not None else np.nan)
        if t >= end - 1e-9:
            break
        model.advance(book, t, t + GRID_DT, tape)
        t += GRID_DT
    tr = np.array([(x[0] - RC.WARMUP_S, x[1]) for x in tape if x[0] >= RC.WARMUP_S], dtype=float).reshape(-1, 2)
    s = RV.segment_summary(np.array(bid), np.array(ask), tr, 1.0)
    s.update({"market": market, "seed": seed})
    return s


def timing() -> None:
    picks = ["TBq_d01", "TBa_d01", "TBl_d01", "Xqr_base", "Xam_base"]
    per = {}
    for m in picks:
        t0 = time.time()
        passive_job((m, RC.SEED_PASSIVE))
        per[m] = time.time() - t0
        print(f"  {m}: {per[m]:.1f} s per episode", flush=True)
    M = all_markets()
    need = M[~M["market"].isin(DEV_STORED_FULL())]
    kind = {"Test B": np.mean([per[k] for k in picks[:3]]), "development": 0.0, "ablation (descriptive)": np.mean([per[k] for k in picks[3:]])}
    tot = sum(kind[s] * RC.N_PASSIVE * n for s, n in need["source"].value_counts().items())
    print("markets needing episodes:", need["source"].value_counts().to_dict())
    print(f"estimated serial time: {tot / 60:.0f} min; with 8 processes about {tot / 60 / 8:.0f} min")


def DEV_STORED_FULL() -> list:
    return [m for m in dev_markets() if m[:3] in DEV_STORED]


def passive(procs: int) -> None:
    from simrel.experiments.run_reference import pool_map

    RAW.mkdir(parents=True, exist_ok=True)
    M = all_markets()
    for src, tag in (("Test B", "testb"), ("development", "dev"), ("ablation (descriptive)", "ablation")):
        path = RAW / f"passive_{tag}.pkl"
        if path.exists():
            continue
        ms = [m for m in M[M["source"] == src]["market"] if m not in DEV_STORED_FULL()]
        segs = pool_map(passive_job, [(m, RC.SEED_PASSIVE + s) for m in ms for s in range(RC.N_PASSIVE)], procs, tag)
        tmp = path.with_suffix(".tmp")
        pickle.dump(segs, open(tmp, "wb"))
        tmp.replace(path)
        print(src, len(segs), "episodes", flush=True)


# ---- per-market statistics and classification (no regret)
def segments() -> dict:
    segs = []
    for tag in ("testb", "dev", "ablation"):
        p = RAW / f"passive_{tag}.pkl"
        if p.exists():
            segs += pickle.load(open(p, "rb"))
    stored = pickle.load(open(G2 / "reversal_check" / "raw" / "sim_passive.pkl", "rb"))
    segs += [s for s in stored if s["market"] in DEV_STORED_FULL()]
    by = {}
    for s in segs:
        by.setdefault(s["market"], []).append(s)
    return {m: sorted(v, key=lambda s: s["seed"]) for m, v in by.items()}


def classify() -> None:
    M = all_markets()
    by = segments()
    rng = np.random.default_rng(0)
    rows = []
    for _, r in M.iterrows():
        segs = by[r["market"]]
        assert [s["seed"] for s in segs] == list(range(RC.SEED_PASSIVE, RC.SEED_PASSIVE + RC.N_PASSIVE)), r["market"]
        qty = RV.parent_qty(segs)
        gaps = [RV.window_gaps(s, qty) for s in segs]
        d = RC.describe(segs, gaps, None, rng)
        c = RV.context(segs)
        tb = c["tick_bp"]
        rows.append({**r.to_dict(), "qty": qty, "tick_bp": tb, "S4_bp": d["S4"], "S4_ticks": d["S4"] / tb, "S4_lo_ticks": d["S4_lo"] / tb,
                     "S4_hi_ticks": d["S4_hi"] / tb, "S3_slope_ticks": d["S3_slope"] / tb, "S3_slope_lo_ticks": d["S3_slope_lo"] / tb,
                     "S3_slope_hi_ticks": d["S3_slope_hi"] / tb, "S3_t": d["S3_t"], "S4_share_tie": d["S4_share_tie"],
                     "n_windows": d["n_windows"], "spread_ticks": c["spread_ticks"],
                     "low_005": bool(d["S4"] / tb >= THRESH), "low_010": bool(d["S4"] / tb >= THRESH_SENS)})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "markets.csv", index=False)

    # check against the reversal check's stored per-market S4 (same code, seeds and summaries)
    st = pd.read_csv(G2 / "reversal_check" / "sim_table.csv")
    chk = T[T["market"].isin(DEV_STORED_FULL())].merge(st[["unit", "S4"]], left_on="market", right_on="unit")
    print("reproduction of sim_table S4 for the stored development markets: max |diff| =",
          float((chk["S4_bp"] - chk["S4"]).abs().max()), "bp over", len(chk), "markets")

    cnt = []
    for src in ("Test B", "development", "ablation (descriptive)"):
        for fam, g in T[T["source"] == src].groupby("family"):
            cnt.append({"source": src, "family": fam, "markets": len(g), "low_at_-0.05": int(g["low_005"].sum()),
                        "low_at_-0.10": int(g["low_010"].sum()), "S4_min": g["S4_ticks"].min(), "S4_q25": g["S4_ticks"].quantile(0.25),
                        "S4_median": g["S4_ticks"].median(), "S4_q75": g["S4_ticks"].quantile(0.75), "S4_max": g["S4_ticks"].max()})
    C = pd.DataFrame(cnt)
    C.to_csv(OUT / "counts.csv", index=False)
    pd.set_option("display.width", 200)
    print(C.round(3).to_string(index=False))
    tb = T[T["source"] == "Test B"]
    for col, th in (("low_005", THRESH), ("low_010", THRESH_SENS)):
        n = int(tb[col].sum())
        print(f"Test B low-continuation at {th}: {n} of {len(tb)}; feasibility (>= {MIN_LOW}): {'PASS' if n >= MIN_LOW else 'STOP'}")
    print("Test B S4 (ticks) deciles:", np.round(np.quantile(tb["S4_ticks"], np.linspace(0, 1, 11)), 3).tolist())


# ---- regret (only after classify)
def market_regret() -> dict:
    """source -> (DataFrame market x arm of size-averaged regret, per regret definition)."""
    b = pd.read_csv(G2 / "addendum10" / "test_b.csv")
    out = {}
    for col, name in (("regret_full", "full-truth"), ("regret", "cross-fitted")):
        out[("Test B", name)] = b.groupby(["world", "arm"])[col].mean().unstack()
    d = pd.read_csv(G2 / "addendum10" / "dev_fourway.csv")
    d = d[d["core"] & d["world"].isin(list(dev_markets()))].groupby(["world", "arm"])["regret"].mean().unstack()
    h = pd.read_csv(G2 / "heldout" / "goal2_A2_A3_selection.csv").groupby("world")[["ensemble_all", "always_twap"]].mean()
    h = h.loc[d.index]
    assert np.allclose(h["always_twap"], d["always TWAP"]), "held-out table and dev_fourway disagree on always TWAP"
    d["ensemble (all designs)"] = h["ensemble_all"]
    out[("development", "development truth")] = d
    return out


FITTED = {"Test B": ["original h_c_f_m_k1", "compound Hawkes", "corrected"],
          "development": ["original h_k1", "original h_c_f_m_k1", "recalibrated h_k1", "recalibrated h_c_f_m_k1", "compound Hawkes", "corrected"]}
RULES = {"Test B": ["rule", "no-simulator"], "development": ["frozen rule (design set D)", "corrected + frozen rule", "ensemble (all designs)"]}
MEAN_ROW = "mean of fitted designs"


def load_classes() -> pd.DataFrame:
    p = OUT / "markets.csv"
    assert p.exists(), "run classify first"
    return pd.read_csv(p).set_index("market")


def headroom() -> None:
    T = load_classes()
    rows = []
    for (src, name), R in market_regret().items():
        if src == "development" or name == "full-truth":
            for col, th in (("low_005", THRESH), ("low_010", THRESH_SENS)):
                low = T[(T["source"] == src) & T[col]].index
                tw = R.loc[R.index.intersection(low), "always TWAP"]
                fam = T.loc[tw.index, "family"].value_counts().to_dict()
                rows.append({"source": src, "regret": name, "threshold": th, "n_low": len(tw), "families": fam,
                             "twap_regret_mean": tw.mean() if len(tw) else np.nan, "twap_regret_median": tw.median() if len(tw) else np.nan,
                             "headroom": bool(len(tw) and tw.mean() >= HEADROOM_BP)})
    H = pd.DataFrame(rows)
    H.to_csv(OUT / "headroom.csv", index=False)
    print(H.round(3).to_string(index=False))


def _strat_boot(diff: pd.Series, fam: pd.Series, rng, n: int = N_BOOT) -> np.ndarray:
    groups = [diff[fam == f].to_numpy() for f in sorted(fam.unique())]
    out = np.empty(n)
    for i in range(n):
        out[i] = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]).mean()
    return out


def compare() -> None:
    T = load_classes()
    H = pd.read_csv(OUT / "headroom.csv")
    rows = []
    for (src, name), R in market_regret().items():
        R = R.copy()
        R[MEAN_ROW] = R[FITTED[src]].mean(axis=1)
        arms = FITTED[src] + [MEAN_ROW] + RULES[src]
        for col, th in (("low_005", THRESH), ("low_010", THRESH_SENS)):
            n_low = int(T.loc[R.index, col].sum())
            if src == "Test B" and n_low < MIN_LOW:
                print(f"{src} at {th}: {n_low} low-continuation markets, below {MIN_LOW}; not compared")
                continue
            if src == "Test B":
                ok = H[(H["source"] == src) & (H["threshold"] == th)]["headroom"].iloc[0]
                if not ok:
                    print(f"{src} at {th}: no headroom; not compared")
                    continue
            for arm in arms:
                rng = np.random.default_rng(0)
                d = R[arm] - R["always TWAP"]
                fam = T.loc[R.index, "family"]
                low = T.loc[R.index, col].astype(bool)
                bl, br = _strat_boot(d[low], fam[low], rng), _strat_boot(d[~low], fam[~low], rng)
                r = {"source": src, "regret": name, "threshold": th, "arm": arm, "n_low": int(low.sum()), "n_rest": int((~low).sum())}
                for tag, est, b in (("low", d[low].mean(), bl), ("rest", d[~low].mean(), br), ("low_minus_rest", d[low].mean() - d[~low].mean(), bl - br)):
                    r.update({tag: est, tag + "_lo": np.percentile(b, 2.5), tag + "_hi": np.percentile(b, 97.5)})
                rows.append(r)
    Cmp = pd.DataFrame(rows)
    Cmp.to_csv(OUT / "comparisons.csv", index=False)
    pd.set_option("display.width", 250)
    print(Cmp.round(2).to_string(index=False))


def figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = load_classes()
    R = market_regret()[("Test B", "full-truth")].copy()
    R[MEAN_ROW] = R[FITTED["Test B"]].mean(axis=1)
    panels = FITTED["Test B"] + [MEAN_ROW]
    col = {"queue-reactive": "#1b7837", "agent-mix": "#2166ac", "latent": "#b2182b"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    x = T.loc[R.index, "S4_ticks"]
    for ax, arm in zip(axes, panels):
        ax.axvspan(-0.089, 0.058, color="0.85", zorder=0, label="real stocks' 95% intervals (union)")
        ax.axvline(THRESH, color="k", lw=1, ls="--", label="threshold -0.05")
        ax.axvline(THRESH_SENS, color="k", lw=1, ls=":", label="sensitivity -0.10")
        ax.axhline(0, color="0.4", lw=0.8)
        for f, c in col.items():
            m = T.loc[R.index, "family"] == f
            ax.scatter(x[m], (R[arm] - R["always TWAP"])[m], s=22, color=c, label=f, zorder=3)
        ax.set_title(arm)
        ax.set_xlabel("S4, impact-free P6 minus TWAP (ticks)")
    axes[0].set_ylabel("regret minus always-TWAP regret (bp, full truth)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Test B markets: fitted-design regret minus TWAP against continuation (exploratory)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_continuation_check.png", dpi=150)
    print("figure written")


def stopped() -> None:
    """Outputs when the plan stops at feasibility or headroom (plan sections 5 and 7): no arm other than always TWAP is
    read. Per-family headroom in the low subset, the ablation S4 table, and a figure of S4 and of the headroom quantity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = load_classes()
    tw = market_regret()[("Test B", "full-truth")]["always TWAP"]
    rows = []
    for col, th in (("low_005", THRESH), ("low_010", THRESH_SENS)):
        low = T[(T["source"] == "Test B") & T[col]]
        for fam, g in low.groupby("family"):
            v = tw.loc[g.index]
            rows.append({"threshold": th, "family": fam, "n_low": len(v), "twap_regret_mean": v.mean(), "twap_regret_max": v.max(),
                         "n_at_least_0.5bp": int((v >= HEADROOM_BP).sum())})
    pd.DataFrame(rows).to_csv(OUT / "headroom_by_family.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    ab = T[T["source"] == "ablation (descriptive)"].copy()
    meta = {k: (knob, lv) for k, (_, _, knob, lv) in ablation.markets().items()}
    ab["knob"], ab["level"] = [meta[m][0] for m in ab.index], [meta[m][1] for m in ab.index]
    ab = ab[["family", "knob", "level", "S4_ticks", "S4_lo_ticks", "S4_hi_ticks", "S3_slope_ticks", "S4_share_tie", "low_005"]]
    ab.sort_values(["family", "knob", "level"]).to_csv(OUT / "ablation_s4.csv")
    print(ab.sort_values(["family", "knob", "level"]).round(3).to_string())

    col = {"queue-reactive": "#1b7837", "agent-mix": "#2166ac", "latent": "#b2182b"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.0))
    srcs = ["Test B", "development"]
    for ax in (a1, a2):
        ax.axvspan(-0.089, 0.058, color="0.85", zorder=0, label="real stocks' 95% intervals (union)")
        ax.axvline(THRESH, color="k", lw=1, ls="--", label="threshold -0.05")
        ax.axvline(THRESH_SENS, color="k", lw=1, ls=":", label="sensitivity -0.10")
    rng = np.random.default_rng(1)
    for i, src in enumerate(srcs):
        for f, c in col.items():
            g = T[(T["source"] == src) & (T["family"] == f)]
            a1.scatter(g["S4_ticks"], i + 0.25 * rng.uniform(-1, 1, len(g)), s=22, color=c, label=f if i == 0 else None, zorder=3,
                       marker="o" if src == "Test B" else "s")
    a1.set_yticks([0, 1], srcs)
    a1.set_xlabel("S4, impact-free P6 minus TWAP at 30 s (ticks)")
    a1.set_title("Continuation by market (20 passive episodes each)")
    a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    low = T[(T["source"] == "Test B") & T["low_010"]]
    for f, c in col.items():
        g = low[low["family"] == f]
        a2.scatter(g["S4_ticks"], tw.loc[g.index], s=22, color=c, zorder=3)
    a2.axhline(HEADROOM_BP, color="0.3", lw=1, ls="-.", label="headroom 0.5 bp")
    a2.set_xlabel("S4 (ticks)")
    a2.set_ylabel("always-TWAP full-truth regret (bp)")
    a2.set_title("Test B markets with S4 >= -0.10: headroom")
    a2.legend(fontsize=7)
    fig.suptitle("Continuation check (exploratory): stopped at headroom; no simulator arm is shown")
    fig.tight_layout()
    fig.savefig(OUT / "fig_continuation_check.png", dpi=150)
    print("figure written")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    {"timing": timing, "passive": lambda: passive(procs), "classify": classify, "headroom": headroom, "compare": compare, "figure": figure, "stopped": stopped}[cmd]()
