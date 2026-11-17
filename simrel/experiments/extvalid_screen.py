"""
External-validity Stage 0: screen the widened draws on observable statistics (simrel/reports/review_audit.md, 4 and 6).

    python simrel/experiments/extvalid_screen.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_reliability as R  # noqa: E402

G = R.G
N_PROBE, SEED_PROBE = 20, 90_000
BOX = {"R100_over_R1": (1.0, 3.5), "exponent": (0.25, 0.65), "H": (0.45, 0.55)}
OUT = G.BASE / "extvalid"


def probe(name: str) -> dict:
    from simrel.configs import reference as C
    from simrel.core import Task
    from simrel.eventsim import run_twin_set
    from simrel.metrics.cost import episode_metrics
    from simrel.policies.library import candidate_set
    from simrel.reference_worlds.data import make

    path = OUT / "probe" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    rec = {"name": name}
    try:
        w = R.sample_window(name, "market", R.SEED_MKT)
        f = R.features(w)
        rec.update({k: f.get(k) for k in ("R100_over_R1", "H")})
        m = w.message
        ex = (m[:, 1] == 4) & (m[:, 3] > 0)
        vol10 = float(m[ex, 3].sum() / (R.SAMPLE_S / 600.0))  # traded shares per 10 minutes
        rec["ref_volume"] = vol10
        tk = C.TASK
        imp, moves, fails = {}, [], 0
        for size in (0.05, 0.15):
            q = max(int(size * vol10) // 100 * 100, 100)
            task = Task(tk.side, q, tk.horizon_s, tk.n_slices, tk.post_s)
            vals = []
            for s in range(N_PROBE):
                try:
                    recs = run_twin_set(make(name), name, SEED_PROBE + s, task, {"P0_twap": candidate_set(size)["P0_twap"]})
                    e = episode_metrics(recs["P0_twap"], recs["none"])
                    vals.append(e["disp_end_bps"])
                    moves.append(abs(e.get("twin_move_bps", 0.0)))
                except Exception:
                    fails += 1
            imp[size] = float(np.nanmean(vals)) if vals else np.nan
        rec["impact_5"], rec["impact_15"] = imp[0.05], imp[0.15]
        rec["exponent"] = float(np.log(imp[0.15] / imp[0.05]) / np.log(3)) if imp[0.05] > 0 and imp[0.15] > 0 else np.nan
        fail_share = fails / (2 * N_PROBE)
        wild = float(np.mean(np.array(moves) > C.WILD_MOVE_BPS)) if moves else 1.0
        rec["viable"] = bool(fail_share <= C.MAX_FAIL_SHARE and wild <= C.MAX_WILD_SHARE)
        rec["fail_share"], rec["wild_share"] = fail_share, wild
    except Exception as err:
        rec["failed"] = repr(err)[:300]
    rec["accepted"] = bool("failed" not in rec and rec.get("viable") and all(
        rec.get(k) is not None and np.isfinite(rec[k]) and lo <= rec[k] <= hi for k, (lo, hi) in BOX.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return rec


def main() -> None:
    from simrel.reference_worlds import extvalid

    names = list(extvalid.draws())
    accepted, summary = {}, {}
    for prefix in ("EVl_d", "EVa_d"):
        fam = [n for n in names if n.startswith(prefix)]
        acc, seen = [], 0
        for b in range(0, len(fam), 24):  # in draw order, 24 at a time, until 6 are accepted
            batch = fam[b:b + 24]
            G.R.pool_map(probe, batch, 6, f"probe {prefix}")
            for n in batch:
                seen += 1
                r = json.loads((OUT / "probe" / f"{n}.json").read_text())
                if r["accepted"]:
                    acc.append(n)
                if len(acc) == 6:
                    break
            print(f"{prefix}: {seen} draws screened, {len(acc)} accepted", flush=True)
            if len(acc) == 6:
                break
        accepted[prefix] = acc
        recs = [json.loads((OUT / "probe" / f"{n}.json").read_text()) for n in fam[:seen]]
        ok = [r for r in recs if "failed" not in r]
        summary[prefix] = {"screened": seen, "accepted": len(acc), "failed_probes": len(recs) - len(ok), "viable": sum(r.get("viable", False) for r in ok),
                           "in_R_box": sum(1 for r in ok if r.get("R100_over_R1") is not None and BOX["R100_over_R1"][0] <= r["R100_over_R1"] <= BOX["R100_over_R1"][1]),
                           "in_exponent_box": sum(1 for r in ok if r.get("exponent") is not None and np.isfinite(r["exponent"]) and BOX["exponent"][0] <= r["exponent"] <= BOX["exponent"][1]),
                           "median_R": float(np.nanmedian([r.get("R100_over_R1") or np.nan for r in ok])) if ok else None,
                           "median_exponent": float(np.nanmedian([r.get("exponent") if r.get("exponent") is not None else np.nan for r in ok])) if ok else None}
    total = sum(len(v) for v in accepted.values())
    res = {"accepted": accepted, "summary": summary, "total_accepted": total, "stop_rule_met": total < 8}
    (OUT / "stage0.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
