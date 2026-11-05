"""
Metaorder impact in each simulator against the empirical square-root law.

The decision depends on the impact of the trader's own ten-minute sequence of child orders (a metaorder), which no
realism statistic measures. The square-root law, one of the most robust empirical regularities in market
microstructure (Toth et al. 2011; Bouchaud et al. 2018, ch. 12), says the peak impact of a metaorder of Q shares is

    I(Q) = Y * sigma_daily * sqrt(Q / V_daily),   with Y of order 1 (typically 0.5 to 1 with consolidated volume),

so (a) its size is predictable from volatility and volume, and (b) tripling Q multiplies it by sqrt(3) = 1.73.

From the paired runs we take, for TWAP: the peak impact as the adverse mid displacement against the unperturbed twin
at the end of execution (disp_end_bps), and the average execution impact (impact_bps). sigma_daily and V_daily come
from the real INTC data. Nasdaq carries only part of the consolidated volume, so we report the predicted band with
Nasdaq visible volume (an upper bound on the prediction) and with Nasdaq volume scaled to an assumed consolidated
total (NASDAQ_SHARE).

    python simrel/experiments/slice_14_metaorder_impact.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import slices as C  # noqa: E402
from simrel.data import lobster as L  # noqa: E402
from simrel.experiments import slice_01_run as R  # noqa: E402

NASDAQ_SHARE = 0.2  # assumed share of consolidated volume executed visibly on Nasdaq (stated as an assumption)
Y_BAND = (0.5, 1.0)
VERSIONS = ["hawkes", "hawkes_sd_q_c", "hawkes_sd_c_f_m", "abides", "abides_quoting"]


def real_scale() -> tuple[float, float]:
    """Daily volatility (from 5-second mid returns over the trading day, in bps) and daily visible volume, on fit days."""
    sig, vol = [], []
    for d in C.FIT_DAYS:
        m, b = L.load_day(C.DATA_DIR, C.SYMBOL, d)
        w = L.cut_windows(C.SYMBOL, d, m, b, length_s=6 * 3600, first_s=9 * 3600 + 45 * 60, last_s=9 * 3600 + 46 * 60)[0]
        _, bid, ask = L.grid_quotes(w)
        mid = 0.5 * (bid + ask)
        r = np.diff(np.log(mid)) * 1e4
        sig.append(r.std() * np.sqrt(6.5 * 3600 / 5))
        vol.append(float(m[m[:, 1] == 4, 3].sum()))
    return float(np.mean(sig)), float(np.mean(vol))


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    sigma_d, v_nasdaq = real_scale()
    info = R.slice_info()["counts"]
    e = pd.read_csv(R.OUT / "policy_episodes.csv")
    if "size_frac" not in e:
        e["size_frac"] = 0.10
    e["size_frac"] = e["size_frac"].fillna(0.10).round(2)
    e = e[e["sim"].isin(VERSIONS) & (e["policy"] == "P0_twap") & (e["seed"] < 40000 + C.N_VERSION_SEEDS)]
    rows = []
    for (v, s), d in e.groupby(["sim", "size_frac"]):
        for name, g in d.groupby("slice"):
            q = s * info[name]["volume_visible"]
            rows.append({"version": v, "size_frac": s, "slice": name, "Q": q, "peak_bps": g["disp_end_bps"].mean(),
                         "avg_impact_bps": g["impact_bps"].mean(), "n": len(g)})
    t = pd.DataFrame(rows)
    for lab, vday in [("nasdaq_volume", v_nasdaq), ("consolidated_volume", v_nasdaq / NASDAQ_SHARE)]:
        t[f"srl_low_{lab}"] = Y_BAND[0] * sigma_d * np.sqrt(t["Q"] / vday)
        t[f"srl_high_{lab}"] = Y_BAND[1] * sigma_d * np.sqrt(t["Q"] / vday)
    t.to_csv(R.OUT / "metaorder_impact.csv", index=False)
    summ = t.groupby(["version", "size_frac"])[["Q", "peak_bps", "avg_impact_bps", "srl_low_consolidated_volume", "srl_high_consolidated_volume",
                                                 "srl_low_nasdaq_volume", "srl_high_nasdaq_volume"]].mean()
    print(f"real {C.SYMBOL}: daily volatility {sigma_d:.0f} bps, Nasdaq visible volume {v_nasdaq:,.0f} shares per day")
    print(summ.round(2).to_string())
    if t["size_frac"].nunique() > 1:
        w = t.pivot_table(index=["version", "slice"], columns="size_frac", values="peak_bps")
        ratio = (w[0.3] / w[0.1]).groupby(level=0).median()
        print("\nPeak impact at 30% divided by peak impact at 10% (square-root law: 1.73; linear: 3.0), median over slices:")
        print(ratio.round(2).to_string())
