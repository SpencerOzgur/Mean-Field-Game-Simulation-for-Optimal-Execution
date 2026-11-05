"""
Exploratory: does a modern queue-reactive candidate change the Goal 2 conclusions? (4 November 2026, not pre-registered)

The queue-reactive candidate (simrel/adapters/qr_candidate.py) is fitted to each market's observable log like the 29
pre-registered candidates. Reported, by market group: its regret against always TWAP and against the 29 other
candidates (its rank), its realism and LOB-Impact numbers against the reference market's (does it relax after a
metaorder?), and its error on the decision-relevant quantities. The pre-registered result files are not touched.

    SIMREL_WITH_QR=1 python simrel/experiments/goal2_qr_candidate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("SIMREL_WITH_QR", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_impact as GI  # noqa: E402
from simrel.experiments import goal2_run as G  # noqa: E402

OUT = G.OUT / "exploratory"


def group_of(w: str) -> str:
    if w in G.WELL_WORLDS:
        return "well-specified (Hawkes-built)"
    return "queue-reactive" if w.startswith("R1") else "agent-mix"


def main() -> None:
    sim = pd.read_csv(G.OUT / "goal2_simulators.csv")  # the 29 pre-registered candidates
    rows, li = [], []
    for w in G.worlds_with_results():
        tr = G.truth(w)
        pols = [G.RAW / "policy" / f"{w}_qr_{G.SEED_POL + s}.pkl" for s in range(G.POLICIES_SEEDS)]
        if not all(p.exists() for p in pols):
            continue
        d = pd.DataFrame([r for p in pols for r in G.load(p)])
        d = d[d["failed"].isna() | (d["failed"] == False)] if "failed" in d else d  # noqa: E712
        real = G.load(G.RAW / f"realism_{w}_qr.pkl")
        for size, g in d.groupby("size_frac"):
            size = round(size, 2)
            means = g.groupby("policy")["is_bps"].mean().round(9).sort_index()
            pick = means.idxmin()
            t = tr[tr["size_frac"].round(2) == size].set_index("policy")
            regret = t.loc[pick, "true_cost"] - t["true_cost"].min()
            others = sim[(sim["world"] == w) & (sim["size_frac"].round(2) == size)]
            twap_peak = g[g["policy"] == "P0_twap"]["disp_end_bps"].mean()
            rows.append({"world": w, "group": group_of(w), "size_frac": size, "pick": pick, "regret_qr": regret,
                         "regret_twap": t.loc["P0_twap", "true_cost"] - t["true_cost"].min(), "regret_hawkes_median": others[others["sim"] != "abides"]["regret"].median(),
                         "regret_best_of_29": others["regret"].min(), "rank_among_30": int((others["regret"] < regret - 1e-9).sum()) + 1,
                         "metaorder_error_qr": abs(twap_peak - t.loc["P0_twap", "true_twap_peak"]), "metaorder_error_median_29": others["metaorder_impact_error"].median(),
                         "order_book_qr": real.get("order_book", np.nan), "order_book_median_29": others["order_book"].median()})
        if (GI.BASE_LI / "rec" / w / "qr").exists():
            s, ref = GI.scores(w, "qr"), GI.scores(w, GI.REF)
            li.append({"world": w, "group": group_of(w), "qr_peak_bps": s.get("p2_peak_bps"), "ref_peak_bps": ref.get("p2_peak_bps"),
                       "qr_end_over_peak": s.get("p3_ep"), "ref_end_over_peak": ref.get("p3_ep"), "qr_H": s.get("p1_H"), "ref_H": ref.get("p1_H")})
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "qr_candidate.csv", index=False)
    pd.DataFrame(li).to_csv(OUT / "qr_candidate_lob_impact.csv", index=False)
    pd.set_option("display.width", 220)
    print("mean regret (bps) by market group")
    print(t.groupby("group")[["regret_qr", "regret_twap", "regret_hawkes_median", "regret_best_of_29", "rank_among_30",
                              "metaorder_error_qr", "metaorder_error_median_29", "order_book_qr", "order_book_median_29"]].mean().round(2).to_string())
    print("\nqueue-reactive candidate's picks:", t["pick"].value_counts().to_dict())
    print("\nLOB-Impact: queue-reactive candidate against the market itself")
    print(pd.DataFrame(li).groupby("group").median(numeric_only=True).round(2).to_string())


if __name__ == "__main__":
    main()
