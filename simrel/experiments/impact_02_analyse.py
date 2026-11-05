"""
Goal 1 with LOB-Impact: analyses G1.1-G1.3 of simrel/reports/lob_impact_design.md, over the three stocks.

    python simrel/experiments/impact_02_analyse.py

Needs, per stock, the LOB-Impact rollouts (impact_01_run.py run) and the Goal 1 tables (realism_profile.csv,
metaorder_impact.csv, immediate_cost.csv, confirmatory_results.txt).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ["INTC", "CSCO", "MSFT"]
VERSIONS = {"hawkes": "Hawkes-original", "hawkes_sd_q_c": "Hawkes-queue", "hawkes_sd_c_f_m": "Hawkes-refill",
            "abides": "ABIDES-original", "abides_quoting": "ABIDES-calibrated"}
OUT = ROOT / "results" / "simrel" / "lob_impact"
OW_EP = 0.87  # the paper's closed-form end-over-peak of the Obizhaeva-Wang kernel at this schedule (C5 threshold)


def slices_dir(sym: str) -> Path:
    return ROOT / "results" / "simrel" / ("slices" if sym == "INTC" else f"slices_{sym}")


def scores(sym: str) -> pd.DataFrame:
    """LOB-Impact scores of the five versions and of the real stream (runs impact_01_run.py score for the stock)."""
    subprocess.run([sys.executable, str(ROOT / "simrel/experiments/impact_01_run.py"), "score", "--sims", ",".join(VERSIONS)],
                   env=dict(os.environ, SIMREL_SYMBOL=sym), check=True, capture_output=True)
    return pd.read_csv(OUT / sym / "lob_impact_scores.csv")


def ladder(r: pd.Series, real: pd.Series) -> dict:
    """The paper's claims C1-C5 (True pass, False fail, None not decidable)."""
    def fin(x):
        return x is not None and np.isfinite(x)

    c1 = None
    if fin(r.get("p1_gamma_flow")) and fin(r.get("p1_H")) and fin(real.get("p1_gamma_flow")) and fin(real.get("p1_H")):
        c1 = bool(abs(r["p1_gamma_flow"] - real["p1_gamma_flow"]) <= 0.10 and abs(r["p1_H"] - real["p1_H"]) <= 0.02
                  and abs(r["p1_drift_ticks"] - real["p1_drift_ticks"]) <= 5)
    elif fin(real.get("p1_gamma_flow")):
        c1 = False  # the version has no measurable sign memory or diffusion where the real stream has one
    # the paper prints a law only where the peak also clears three standard errors (section 2.1)
    c2 = bool(r.get("p2_identified", False)) and bool(r.get("p2_detected", False))
    c3 = None if not fin(r.get("att_vb")) else bool(r["att_vb"] >= 3)
    c4 = None if not c2 else bool(abs(r["p2_delta"] - 0.5) <= 0.10 and 0.5 <= r["p2_Y"] <= 1.0)
    c5 = None
    if bool(r.get("p3_sig", False)) and fin(r.get("p3_ep")):
        c5 = bool(r["p3_ep"] <= OW_EP and r["p3_slope_per_duration"] <= 0 and abs(r["p1_drift_ticks"]) < 20)
    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}


def mark(v) -> str:
    return "·" if v is None else ("✓" if v else "✗")


def decision(sym: str) -> dict:
    """D (TWAP minus immediate, bps, 10% size) per version from the confirmatory results file."""
    out = {}
    for line in (slices_dir(sym) / "confirmatory_results.txt").read_text().splitlines():
        m = re.match(r"size 10%\s+D\((.+?)\) = ([+-][\d.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main() -> None:
    pd.set_option("display.width", 250)
    rows = []
    for sym in SYMBOLS:
        if not (OUT / sym / "rec").exists():
            print(f"{sym}: no rollouts yet")
            continue
        t = scores(sym)
        real = t[t["sim"] == "real"].iloc[0]
        prof = pd.read_csv(slices_dir(sym) / "realism_profile.csv", index_col=0)
        meta = pd.read_csv(slices_dir(sym) / "metaorder_impact.csv")
        meta = meta[meta["size_frac"].round(2) == 0.10].groupby("version")[["peak_bps", "srl_low_nasdaq_volume", "srl_high_nasdaq_volume"]].mean()
        imm = pd.read_csv(slices_dir(sym) / "immediate_cost.csv")
        imm = imm[imm["size_frac"].round(2) == 0.10]
        D = decision(sym)
        for sim, label in VERSIONS.items():
            r = t[t["sim"] == sim]
            if r.empty:
                continue
            r = r.iloc[0]
            lad = ladder(r, real)
            rows.append({"stock": sym, "version": label, **{k: r.get(k, np.nan) for k in t.columns if k != "sim"},
                         "ladder": "".join(mark(v) for v in lad.values()), **lad,
                         "lobbench_order_book": prof.loc[sim, "order_book"] if sim in prof.index else np.nan,
                         "lobbench_impact": prof.loc[sim, "impact"] if sim in prof.index else np.nan,
                         "twap_peak_bps_10pct": meta.loc[sim, "peak_bps"] if sim in meta.index else np.nan,
                         "immediate_cost_ratio": float(imm[sim].mean() / imm["real_immediate_bps"].mean()) if sim in imm else np.nan,
                         "D_10pct": D.get(label, np.nan)})
        rows.append({"stock": sym, "version": "real stream", **{k: real.get(k, np.nan) for k in t.columns if k.startswith("p1_")}})
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "goal1_lob_impact.csv", index=False)
    cols = ["stock", "version", "p1_gamma_flow", "p1_H", "p1_R100_over_R1", "p1_drift_ticks", "p2_delta", "p2_delta_lo", "p2_delta_hi", "p2_Y",
            "p2_peak_bps", "p3_ep", "att_vb", "ladder", "lobbench_order_book", "twap_peak_bps_10pct", "immediate_cost_ratio", "D_10pct"]
    print("G1.1  LOB-Impact numbers and the claim ladder C1-C5 (✓ pass, ✗ fail, · not decidable)")
    print(g[[c for c in cols if c in g]].round(3).to_string(index=False))
    from scipy.stats import spearmanr

    print("\nG1.2  agreement with our real-data checks, rank correlation over the five versions")
    out = []
    for sym, d in g[g["version"] != "real stream"].groupby("stock"):
        for a, b in [("p2_peak_bps", "twap_peak_bps_10pct"), ("p2_Y", "twap_peak_bps_10pct"), ("p2_peak_bps", "D_10pct"),
                     ("lobbench_impact", "twap_peak_bps_10pct"), ("lobbench_order_book", "D_10pct")]:
            ok = d[[a, b]].dropna()
            out.append({"stock": sym, "x": a, "y": b, "spearman": spearmanr(ok[a], ok[b]).correlation if len(ok) >= 4 else np.nan, "n": len(ok)})
    o = pd.DataFrame(out)
    o.to_csv(OUT / "goal1_agreement.csv", index=False)
    print(o.round(2).to_string(index=False))
    print("\nG1.3  the two versions that give opposite advice, side by side")
    print(g[g["version"].isin(["Hawkes-original", "ABIDES-original"])][["stock", "version", "lobbench_order_book", "p1_H", "p2_delta", "p2_Y", "p2_peak_bps", "p3_ep", "D_10pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
