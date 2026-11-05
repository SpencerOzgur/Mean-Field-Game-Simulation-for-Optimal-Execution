"""
Pre-registered analysis of the confirmatory replication (simrel/reports/preregistration_confirmatory.md, H1-H5).

Written before any confirmatory strategy result existed. Run per symbol:

    SIMREL_SYMBOL=CSCO python simrel/experiments/confirm_analyse.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel import slices as S  # noqa: E402
from simrel.configs import slices as C  # noqa: E402

OUT = S.out_dir()
RAW = OUT / "raw"
V = {"Hawkes-original": "hawkes", "Hawkes-queue": "hawkes_sd_q_c", "Hawkes-refill": "hawkes_sd_c_f_m",
     "ABIDES-original": "abides", "ABIDES-calibrated": "abides_quoting"}
N_BOOT = 2000
RNG = np.random.default_rng(20261031)


def episodes() -> pd.DataFrame:
    rows = []
    for p in (RAW / "policy").glob("*.pkl"):
        d = pd.read_pickle(p)
        rows += d["rows"]
    e = pd.DataFrame(rows)
    e = e[e["failed"].isna() | (e["failed"] == False)]  # noqa: E712
    e["size_frac"] = e["size_frac"].fillna(0.10).round(2)  # INTC runs made before the size option existed were all at 10%
    # the pre-registered seeds only (40000-40019); the base versions also ran seeds 40020-40059 because the runner gives
    # them 60 seeds by default, which the pre-registration did not ask for (reported as a deviation, used only as a check)
    e = e[e["seed"] < 40000 + C.N_VERSION_SEEDS] if "--all-seeds" not in sys.argv else e
    return e[e["sim"].isin(V.values()) & e["policy"].isin(C.VERSION_POLICIES)]


def d_table(e: pd.DataFrame, sim: str, size: float) -> pd.Series:
    """TWAP minus immediate per (slice, seed)."""
    d = e[(e["sim"] == sim) & (e["size_frac"] == size)].pivot_table(index=["slice", "seed"], columns="policy", values="is_bps")
    return (d["P0_twap"] - d["P1_immediate"]).dropna()


def boot_mean(x: pd.Series) -> tuple[float, float, float]:
    """Mean over slices of slice means, with a bootstrap that resamples seeds within each slice."""
    groups = [g.to_numpy() for _, g in x.groupby(level=0)]
    stat = np.mean([g.mean() for g in groups])
    bs = [np.mean([RNG.choice(g, len(g)).mean() for g in groups]) for _ in range(N_BOOT)]
    return float(stat), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_diff(x: pd.Series, y: pd.Series, alpha: float = 0.05) -> tuple[float, float, float]:
    """Difference of the two versions' mean-of-slice-means (common slices), resampling seeds within slices."""
    sl = sorted(set(x.index.get_level_values(0)) & set(y.index.get_level_values(0)))
    gx = [x.loc[s].to_numpy() for s in sl]
    gy = [y.loc[s].to_numpy() for s in sl]
    stat = np.mean([a.mean() - b.mean() for a, b in zip(gx, gy)])
    bs = [np.mean([RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for a, b in zip(gx, gy)]) for _ in range(N_BOOT)]
    return float(stat), float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2))), bs


def main() -> None:
    e = episodes()
    real = pd.read_csv(OUT / "realism.csv")
    ob = real.groupby("source")["l1_mean"].mean()
    prof = pd.read_csv(OUT / "realism_profile.csv", index_col=0) if (OUT / "realism_profile.csv").exists() else None
    lines, res = [], {}
    for size in (0.10, 0.30):
        D = {k: d_table(e, v, size) for k, v in V.items()}
        if any(len(x) == 0 for x in D.values()):
            lines.append(f"size {size:.0%}: incomplete ({ {k: len(x) for k, x in D.items()} })")
            continue
        est = {k: boot_mean(x) for k, x in D.items()}
        for k, (m, lo, hi) in est.items():
            lines.append(f"size {size:.0%}  D({k}) = {m:+.2f} bps  [{lo:+.2f}, {hi:+.2f}]  ({len(D[k])} seed-slice pairs)")
        if size == 0.10:
            h1 = est["Hawkes-original"][1] > 0 and est["ABIDES-original"][2] < 0 and est["Hawkes-original"][0] - est["ABIDES-original"][0] >= 2
            dq = boot_diff(D["Hawkes-queue"], D["Hawkes-original"])
            h2 = ob[V["Hawkes-queue"]] < ob[V["Hawkes-original"]] and dq[1] > 0
            res["H1"], res["H2"] = h1, h2
            lines.append(f"H1 {'HOLDS' if h1 else 'does not hold'}: D(Hawkes-original) - D(ABIDES-original) = {est['Hawkes-original'][0] - est['ABIDES-original'][0]:+.2f} bps")
            lines.append(f"H2 {'HOLDS' if h2 else 'does not hold'}: order book L1 queue {ob[V['Hawkes-queue']]:.3f} vs original {ob[V['Hawkes-original']]:.3f}; "
                         f"D(queue) - D(original) = {dq[0]:+.2f} [{dq[1]:+.2f}, {dq[2]:+.2f}]")
        # H3: equivalence within +-1 bp (two one-sided tests at 5% each = 90% interval inside the margin)
        diff, lo90, hi90, _ = boot_diff(D["Hawkes-refill"], D["ABIDES-calibrated"], alpha=0.10)
        h3 = lo90 > -1.0 and hi90 < 1.0
        res[f"H3 {size:.0%}"] = h3
        lines.append(f"H3 {'HOLDS' if h3 else 'does not hold'} at {size:.0%}{' (confirmatory)' if size == 0.10 else ' (exploratory)'}: "
                     f"D(Hawkes-refill) - D(ABIDES-calibrated) = {diff:+.2f}, 90% interval [{lo90:+.2f}, {hi90:+.2f}], margin +-1")
        if size == 0.10:
            vals = pd.DataFrame({"order_book": [ob[v] for v in V.values()], "D": [est[k][0] for k in V]}, index=list(V))
            rho = vals["order_book"].rank().corr(vals["D"].rank())
            res["H5"] = rho < 0
            lines.append(f"H5 {'HOLDS' if rho < 0 else 'does not hold'} (descriptive): Spearman(order-book L1, D) = {rho:+.2f} over the five versions")
    if prof is not None and "impact" in prof:
        floor = prof.loc["real_fit_days", "impact"]
        sims = prof.loc[[v for v in V.values() if v in prof.index], "impact"]
        h4 = bool((sims > 2 * floor).all())
        res["H4"] = h4
        lines.append(f"H4 {'HOLDS' if h4 else 'does not hold'}: impact floor {floor:.2f}; versions " + ", ".join(f"{k} {sims.get(v, np.nan):.2f}" for k, v in V.items()))
    text = f"Confirmatory analysis, {C.SYMBOL}\n" + "\n".join(lines)
    (OUT / "confirmatory_results.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
