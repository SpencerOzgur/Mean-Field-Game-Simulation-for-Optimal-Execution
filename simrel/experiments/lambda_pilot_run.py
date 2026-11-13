"""
Main run of the lambda pilot (E1 to E4). Run only after simrel/reports/lambda_pilot_rule.md is committed and pushed.

Arms and their world parameters per market come from results/simrel/goal2/lambda_pilot/arms.json, written by the
calibration (lambda_pilot_calib.py) and recorded in the rule file:

    lam0               lambda = 0 (AgentMix, bit for bit); the two latent markets are run in this arm only, as a reference
    all_dose           all-flow lambda 0.25, volatility-matched fund_vol (unjudged dose-response level)
    all_band_low       all-flow lambda 0.6 (judged)
    all_max_feasible   all-flow lambda 1.0 (judged)
    trader_band_low    COMPARISON: only the trader's orders move F, lambda 0.6, the market's own fund_vol
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import reference as C  # noqa: E402
from simrel.experiments import lambda_pilot as P  # noqa: E402

ARMS = ["lam0", "all_dose", "all_band_low", "all_max_feasible", "trader_band_low"]


def cells() -> list:
    """(market, arm, world kwargs, reference volume) of every market and arm that is run."""
    A, M = P.arms(), P.markets()
    V = P.load_json("pilot_markets.json")["reference_volume"]
    out = []
    for m, (fam, _) in M.items():
        for a in ARMS if fam == "am" else ["lam0"]:
            out.append((m, a, A[a][m] if fam == "am" else {}, V[m]))
    return out


def run(procs: int) -> None:
    from simrel.experiments.run_reference import pool_map

    cs = cells()
    P.cached("twins", lambda: pool_map(P.twin_job, [(m, a, kw, C.SEED_HELDOUT + s) for m, a, kw, _ in cs for s in range(P.N_VIAB)], procs, "twins"))
    P.cached("truth", lambda: pool_map(P.truth_job, [(m, a, kw, v, s) for m, a, kw, v in cs for s in range(P.N_TRUTH)], procs, "truth"))
    am = [c for c in cs if P.markets()[c[0]][0] == "am"]
    jobs = [(m, a, kw, P.SEED_E2B + k, "B") for m, a, kw, _ in am if a == "lam0" for k in range(P.N_E2)]
    jobs += [(m, a, kw, P.SEED_E2A + k, "A") for m, a, kw, _ in am for k in range(P.N_E2)]
    P.cached("e2", lambda: pool_map(P.e2_job, jobs, procs, "e2"))


def timing(procs: int) -> None:
    cs = cells()
    pick = [c for c in cs if c[0].endswith("a01") and c[1] in ("lam0", "all_max_feasible")] + [c for c in cs if c[0].endswith("l01")]
    per = {}
    for name, job in [("twins", lambda c: P.twin_job((c[0], c[1], c[2], C.SEED_HELDOUT))), ("truth", lambda c: P.truth_job((c[0], c[1], c[2], c[3], 0))),
                      ("e2", lambda c: P.e2_job((c[0], c[1], c[2], P.SEED_E2A, "A")))]:
        t0 = time.time()
        out = [job(c) for c in pick[:2] if not (name == "e2" and c[0].endswith("l01"))] + ([job(pick[2])] if name != "e2" else [])
        per[name] = (time.time() - t0) / len(out)
        if name == "e2":
            print("  E2 timing samples failed:", [r.get("failed") for r in out])
    n_am = sum(1 for c in cs if P.markets()[c[0]][0] == "am")
    n = {"twins": len(cs) * P.N_VIAB, "truth": len(cs) * P.N_TRUTH, "e2": (n_am + len(P.markets()) - P.N_LATENT) * P.N_E2}
    total = sum(n[k] * per[k] for k in n)
    for k in n:
        print(f"  {k:8s} {n[k]:5d} jobs x {per[k]:6.1f} s = {n[k] * per[k] / 3600:5.2f} core-hours")
    print(f"projected: {total / 3600:.2f} core-hours, about {total / 3600 / procs * 1.25:.2f} hours on {procs} processes (25% overhead allowed)")
