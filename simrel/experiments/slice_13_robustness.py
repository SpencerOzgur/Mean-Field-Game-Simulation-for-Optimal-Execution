"""
Is each simulator's recommendation robust to how it was fitted or tuned? (part C of the validation)

Families: each main version and its refits or alternatives, with the same four policies and the same 20 seeds per
slice. For every member: TWAP cost minus immediate cost (positive = immediate is cheaper), overall and per slice, and
the best policy per slice. We compare the spread within a family with the gaps between families.

    python simrel/experiments/slice_13_robustness.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import slices as C  # noqa: E402
from simrel.metrics.cost import boot_ci  # noqa: E402

OUT = ROOT / "results" / "simrel" / "slices"
FAMILIES = {
    "Hawkes original": ["hawkes", "hawkes__fast"] + [f"hawkes__lodo{j}" for j in range(5)],
    "Hawkes best (same-day + spread + gap filling + multi-timescale)": ["hawkes_sd_c_f_m"] + [f"hawkes_sd_c_f_m__lodo{j}" for j in range(5)],
    "ABIDES original": ["abides", "abides__alt1", "abides__alt2"],
    "ABIDES calibrated": ["abides_quoting", "abides_quoting__alt1"],
}

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    e = pd.read_csv(OUT / "policy_episodes.csv")
    e = e[e["policy"].isin(C.VERSION_POLICIES) & (e["seed"] < 40000 + C.N_VERSION_SEEDS)]
    e = e[e["failed"].isna() | (e["failed"] == False)]  # noqa: E712
    rows, per_slice = [], []
    for fam, members in FAMILIES.items():
        for v in members:
            d = e[e["sim"] == v]
            w = d.pivot_table(index=["slice", "seed"], columns="policy", values="is_bps")
            x = (w["P0_twap"] - w["P1_immediate"]).dropna()
            lo, hi = boot_ci(x.to_numpy())
            best = d.groupby(["slice", "policy"])["is_bps"].mean().unstack().idxmin(axis=1)
            rows.append({"family": fam, "version": v, "n": len(w), "twap_minus_immediate": x.mean(), "lo": lo, "hi": hi,
                         "slices_best_immediate": int((best == "P1_immediate").sum())})
            for s, xs in x.groupby(level=0):
                per_slice.append({"family": fam, "version": v, "slice": s, "d": xs.mean(), "best": best[s]})
    t = pd.DataFrame(rows)
    ps = pd.DataFrame(per_slice)
    t.to_csv(OUT / "robustness_by_version.csv", index=False)
    ps.to_csv(OUT / "robustness_by_slice.csv", index=False)
    print(t.round(2).to_string(index=False))
    fam = t.groupby("family")["twap_minus_immediate"].agg(["mean", "std", "min", "max"])
    print("\nSpread within each family (TWAP minus immediate, bps):")
    print(fam.round(2).to_string())
    # per slice: does the sign of TWAP-minus-immediate agree with the main version, and does the best policy?
    agree = []
    for f, members in FAMILIES.items():
        main = ps[ps["version"] == members[0]].set_index("slice")
        for v in members[1:]:
            o = ps[ps["version"] == v].set_index("slice")
            agree.append({"family": f, "version": v, "same_sign_slices": int((np.sign(o["d"]) == np.sign(main["d"])).sum()),
                          "same_best_policy_slices": int((o["best"] == main["best"]).sum()),
                          "mean_abs_change_bps": float((o["d"] - main["d"]).abs().mean())})
    a = pd.DataFrame(agree)
    a.to_csv(OUT / "robustness_agreement.csv", index=False)
    print("\nAgreement with the family's main version, per slice (out of 7):")
    print(a.round(2).to_string(index=False))
    m = {f: g["twap_minus_immediate"].mean() for f, g in t.groupby("family")}
    print("\nGap between family means: Hawkes original vs ABIDES original %.1f bps; Hawkes best vs ABIDES calibrated %.1f bps" % (
        m["Hawkes original"] - m["ABIDES original"], m["Hawkes best (same-day + spread + gap filling + multi-timescale)"] - m["ABIDES calibrated"]))
