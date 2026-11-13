"""
Goal 2: which kinds of realism predict good execution decisions? (design: simrel/reports/goal2_design.md)

In four synthetic reference markets with known true strategy costs, a family of candidate simulators is fitted to
observable data, scored for realism against held-out reference windows, and asked for its recommended strategy, whose
regret is known.

    python simrel/experiments/goal2_run.py [stage ...] [--procs N] [--worlds W1,W2]

Stages: reference (held-out windows, true costs), fit, realism, policy, analyse. All results are cached under
results/simrel/goal2/raw.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import reference as RC  # noqa: E402
from simrel.data import lobster as L  # noqa: E402
from simrel.experiments import slice_01_run as R  # noqa: E402
from simrel.lob import Book  # noqa: E402

BASE = ROOT / "results" / "simrel" / "goal2"
RAW = BASE / "raw"
# analyses at another decision budget (addendum 7) write their result files under results/simrel/goal2/<tag>; every
# input (fits, policy runs, LOB-Impact rollouts) stays in the base folder
OUT = BASE / os.environ["SIMREL_GOAL2_TAG"] if os.environ.get("SIMREL_GOAL2_TAG") else BASE
REF = ROOT / "results" / "simrel" / "raw"
WORLDS = ["R1a_qr_resilient", "R1b_qr_fragile", "R2a_am_anchored", "R2b_am_trending"]
NEW_WORLDS = ["R1c_qr_thin", "R1d_qr_momentum", "R1e_qr_deep", "R2c_am_anchor", "R2d_am_toxic", "R2e_am_calm"]  # addendum 3


SAMPLED_WORLDS = [f"R1s{k:02d}" for k in range(1, 6)] + [f"R2s{k:02d}" for k in range(1, 6)]  # drawn at random (addendum 4)
LATENT_WORLDS = [f"R3s{k:02d}" for k in range(1, 7)]  # held-out latent-liquidity family (addendum 8)
LATENT_LONG = [w + "_long" for w in LATENT_WORLDS]  # the same markets under the 30-minute task variant (addendum 8)
FIXED_DESIGNS = ["h_k1", "h_c_f_m_k1", "abides", "qr"]  # the four fixed candidate designs of addendum 8, steps 3 and 5
WELL_WORLDS = ["W1_hk1_R1a", "W2_hcfm_R1a", "W3_hk1_R1b", "W4_hcfm_R1b", "W5_hk1_R2a", "W6_hcfm_R2a"]  # well-specified (addendum 5)


def worlds_with_results() -> list:
    """Every market whose reference truth exists and that passed the viability rule of addendum 3."""
    out = []
    # the held-out latent family (addendum 8) is analysed on its own: it joins only when asked for
    held_out = LATENT_WORLDS if os.environ.get("SIMREL_INCLUDE_HELDOUT") else []
    for w in WORLDS + NEW_WORLDS + SAMPLED_WORLDS + WELL_WORLDS + held_out:
        info = REF / f"ref_{w}" / "info.json"
        if (REF / f"ref_{w}" / "rows_reference.pkl").exists() and info.exists() and json.loads(info.read_text()).get("viable", True):
            out.append(w)
    return out
POLICIES_SEEDS = int(os.environ.get("SIMREL_POLICY_SEEDS", 30))  # decision seeds per candidate (100 in addendum 7)
N_WINDOWS = 12
WARMUP_S, WINDOW_S = 600, 600
SEED_REF_WIN, SEED_SIM_WIN, SEED_POL = 60_000, 61_000, 62_000
BETAS = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)  # the second study's grid for these slower markets
MULTI_BETAS = (0.1, 1.0, 10.0)
WEIGHT = {"weight_exponent": 1.0, "weight_cap": 10.0}
FILL_RATE = 200.0


def family() -> dict:
    """name -> options. Structure x impact factor, plus stability caps for two structures, plus ABIDES."""
    out = {}
    for close, fill, multi in itertools.product([False, True], repeat=3):
        base = "h" + ("_c" if close else "") + ("_f" if fill else "") + ("_m" if multi else "")
        for k in (0.5, 1.0, 2.0):
            out[f"{base}_k{k:g}"] = {"close": close, "fill": fill, "multi": multi, "impact": k, "cap": 0.9}
    for cap in (0.8, 0.95):
        out[f"h_cap{cap:g}"] = {"close": False, "fill": False, "multi": False, "impact": 1.0, "cap": cap}
        out[f"h_c_f_m_cap{cap:g}"] = {"close": True, "fill": True, "multi": True, "impact": 1.0, "cap": cap}
    out["abides"] = {}
    if os.environ.get("SIMREL_WITH_QR"):  # exploratory 30th candidate (4 November 2026): a queue-reactive simulator
        out["qr"] = {}
    return out


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pickle.dump(obj, open(tmp, "wb"))
    tmp.replace(path)


def load(path: Path):
    return pickle.load(open(path, "rb"))


def ref_info(world: str) -> dict:
    return json.loads((REF / f"ref_{world}" / "info.json").read_text())


def qty(world: str, size: float) -> int:
    return max(int(size * ref_info(world)["reference_volume"]) // 100 * 100, 100)


# ------------------------------------------------------------------ windows in LOBSTER form
def window_from_model(model, seed_label: str) -> L.Window:
    book = Book()
    book.tape = []
    model.init_book(book)
    model.advance(book, 0.0, WARMUP_S + WINDOW_S, [])
    w = L.tape_to_lobster(book.tape, WARMUP_S, WINDOW_S, price_unit=100)
    return R.two_sided(R.S.visible_window(w))[0]


def reference_windows(world: str) -> list:
    path = RAW / f"refwin_{world}.pkl"
    if path.exists():
        return load(path)
    from simrel.reference_worlds.data import make

    ws = [window_from_model(make(world)(SEED_REF_WIN + k), "ref") for k in range(N_WINDOWS)]
    save(path, ws)
    return ws


def truth(world: str) -> pd.DataFrame:
    rows = pd.DataFrame(load(REF / f"ref_{world}" / "rows_reference.pkl"))
    rows = rows[rows["failed"].isna() | (rows["failed"] == False)]  # noqa: E712
    t = rows.groupby(["size_frac", "policy"]).agg(true_cost=("is_bps", "mean"), true_twap_peak=("disp_end_bps", "mean")).reset_index()
    return t


# ------------------------------------------------------------------ fitted simulators
def fit_job(args) -> str:
    from simrel.adapters import hawkes_fit

    world, kind, cap = args
    path = RAW / f"fit_{world}_{kind}_cap{cap:g}.pkl"
    if path.exists():
        return "cached"
    obs = load(REF / f"ref_{world}" / "fit_data.pkl")
    multi = MULTI_BETAS if kind == "multi" else None
    p, info = hawkes_fit.fit(obs, betas=BETAS, lot=100, empirical=True, max_branching=cap, multi_betas=multi, **WEIGHT)
    save(path, {"params": p, "info": info})
    return f"{world} {kind} cap {cap}: beta {info['beta']}, branching {info['branching_ratio']:.2f}"


def fit_qr_job(world: str) -> str:
    from simrel.adapters import qr_candidate

    path = RAW / f"fit_{world}_qr.pkl"
    if path.exists():
        return "cached"
    obs = load(REF / f"ref_{world}" / "fit_data.pkl")
    p, info = qr_candidate.fit(obs, p0=int(round(ref_info(world)["mean_price_ticks"])))
    save(path, {"params": p, "info": info})
    return f"{world} qr: {info['states_visited']} states visited"


def fit_chawkes_job(world: str) -> str:
    from simrel.adapters import compound_hawkes

    path = RAW / f"fit_{world}_chawkes.pkl"
    if path.exists():
        return "cached"
    p, info = compound_hawkes.fit(load(REF / f"ref_{world}" / "fit_data.pkl"))
    save(path, {"params": p, "info": info})
    return f"{world} chawkes: branching {info['branching_ratio']:.2f}, spread exponent {info['s_exp']}"


def make_factory(world: str, name: str):
    """seed -> simulator model (Hawkes family) built from the fits and the options of this family member."""
    if name == "oracle":  # the market's own generator with its true parameters (addendum 6, positive control)
        from simrel.reference_worlds.data import make

        return make(world)
    if name.startswith("oracle-no-"):  # the true generator with one mechanism switched off (addendum 9)
        from simrel.reference_worlds.data import registry

        cls, params = registry()[world]
        knob = name[len("oracle-no-"):]
        removed = dataclasses.replace(params, **{knob: 0.0})
        return lambda seed: cls(removed, seed)
    if name == "chawkes":  # compound Hawkes baseline (addendum 10, section 6)
        from simrel.adapters.compound_hawkes import CompoundHawkesLOB

        p = load(RAW / f"fit_{world}_chawkes.pkl")["params"]
        return lambda seed: CompoundHawkesLOB(p, seed)
    if name == "qr":
        from simrel.adapters.qr_candidate import QRCandidate

        p = load(RAW / f"fit_{world}_qr.pkl")["params"]
        return lambda seed: QRCandidate(p, seed)
    from simrel.adapters.hawkes import MB, MS
    from simrel.adapters.hawkes_fast import FastHawkesLOB

    o = family()[name]
    fit = load(RAW / f"fit_{world}_{'multi' if o['multi'] else 'single'}_cap{o['cap']:g}.pkl")
    p = fit["params"]
    lam_max = np.array(p.lam_max, dtype=float)
    extra = {}
    if o["close"]:
        rate = fit["info"]["close_rate_per_side"]
        lam_max[:2] += rate
        extra["close_rate"] = rate
    if o["fill"]:
        lam_max[:2] += FILL_RATE
        extra["fill_rate"] = FILL_RATE
    p = dataclasses.replace(p, lam_max=lam_max, market_ratio_q=None, **extra)
    if o["impact"] != 1.0:  # as for INTC: scale the excitation caused by market orders, keeping the branching ratio
        alpha = np.array(p.alpha, dtype=float)
        alpha[:, [MB, MS]] *= o["impact"]
        stack = None if p.alpha_stack is None else np.array(p.alpha_stack, dtype=float)
        if stack is not None:
            stack[:, :, [MB, MS]] *= o["impact"]
        b0 = float(np.max(np.abs(np.linalg.eigvals(np.asarray(p.alpha)))))
        b1 = float(np.max(np.abs(np.linalg.eigvals(alpha))))
        if b1 > b0:
            alpha *= b0 / b1
            if stack is not None:
                stack *= b0 / b1
        p = dataclasses.replace(p, alpha=alpha, alpha_stack=stack)
    p = dataclasses.replace(p, p0=int(round(ref_info(world)["mean_price_ticks"])))
    return lambda seed: FastHawkesLOB(p, seed)


def abides_spec(world: str):
    return load(REF / f"ref_{world}" / "fit_abides.pkl")[0]


# ------------------------------------------------------------------ realism
def realism_job(args) -> str:
    from simrel.metrics import impact_response as IR
    from simrel.metrics import lobbench

    world, name = args
    path = RAW / f"realism_{world}_{name}.pkl"
    if path.exists():
        return "cached"
    R._memory_guard(path, 6.0)
    try:
        if name == "abides":
            spec = abides_spec(world)
            gen = [R.two_sided(R.abides_window(spec, SEED_SIM_WIN + k))[0] for k in range(N_WINDOWS)]
        else:
            f = make_factory(world, name)
            gen = [window_from_model(f(SEED_SIM_WIN + k), name) for k in range(N_WINDOWS)]
        gen = [w for w in gen if len(w.message) > 50]
        ref = reference_windows(world)[: len(gen)]
        rp, gp = [(w.message, w.book) for w in ref], [(w.message, w.book) for w in gen]
        ob = lobbench.score(rp, gp)
        pr = lobbench.score(rp, gp, config=lobbench.price_dynamics_config())
        imp = lobbench.impact_response(rp, gp, tempfile.mkdtemp())
        rr = pd.concat([IR.responses(w.message, w.book, w.book_before) for w in ref])
        rg = pd.concat([IR.responses(w.message, w.book, w.book_before) for w in gen])
        rec = {"order_book": float(ob["l1"].mean()), "price": float(pr["wasserstein"].mean()), "impact": imp["mean"],
               "trade_response_error": float(abs(rg["resp_10s"].mean() - rr["resp_10s"].mean())), "n_windows": len(gen)}
    except Exception as err:
        rec = {"failed": repr(err)[:300]}
    save(path, rec)
    return f"{world} {name}: " + (", ".join(f"{k} {v:.3f}" for k, v in rec.items() if isinstance(v, float)) if "failed" not in rec else rec["failed"][:100])


# ------------------------------------------------------------------ decisions
def policy_job(args) -> str:
    from simrel.core import Task
    from simrel.metrics.cost import episode_metrics
    from simrel.policies.library import candidate_set

    world, name, seed = args
    path = RAW / "policy" / f"{world}_{name}_{seed}.pkl"
    if path.exists():
        return "cached"
    rows = []
    try:
        for size in RC.sizes_for(world):
            tk = RC.task_for(world)
            task = Task(tk.side, qty(world, size), tk.horizon_s, tk.n_slices, tk.post_s)
            pol = candidate_set(size)
            if name == "abides":
                from simbench import abides_world as AW
                from simrel.adapters import abides

                spec = abides_spec(world)
                AW.WORLDS[spec.name] = spec
                recs = abides.run_twin_set(spec.name, seed, task, pol)
            else:
                from simrel.eventsim import run_twin_set

                recs = run_twin_set(make_factory(world, name), f"{world}:{name}", seed, task, pol)
            twin = recs["none"]
            rows += [{"world": world, "sim": name, "size_frac": size, **episode_metrics(r, twin)} for k, r in recs.items() if k != "none"]
    except Exception as err:
        rows = [{"world": world, "sim": name, "seed": seed, "failed": repr(err)[:300]}]
    save(path, rows)
    return f"{world} {name} {seed}"


# ------------------------------------------------------------------ analysis
def analyse() -> None:
    from scipy.stats import spearmanr

    pd.set_option("display.width", 240)
    rows = []
    for w in worlds_with_results():
        tr = truth(w)
        for name in family():
            rp = RAW / f"realism_{w}_{name}.pkl"
            pols = [RAW / "policy" / f"{w}_{name}_{SEED_POL + s}.pkl" for s in range(POLICIES_SEEDS)]
            if os.environ.get("SIMREL_ALLOW_MISSING_SEEDS"):  # interim analyses while runs are still arriving: use the seeds present
                pols = [p for p in pols if p.exists()]
            if not rp.exists() or not pols or not all(p.exists() for p in pols):  # noqa: E501
                continue
            real = load(rp)
            if "failed" in real:
                continue
            df = pd.DataFrame([r for p in pols for r in load(p)])
            df = df[df.get("failed").isna() | (df.get("failed") == False)] if "failed" in df else df  # noqa: E712
            for size, d in df.groupby("size_frac"):
                # exact ties (identical predicted costs) go to the first strategy by name, the same rule in every code path
                means = d.groupby("policy")["is_bps"].mean().round(9).sort_index()
                pick = means.idxmin()
                t = tr[tr["size_frac"] == size].set_index("policy")
                twap_peak = d[d["policy"] == "P0_twap"]["disp_end_bps"].mean()
                rows.append({"world": w, "sim": name, "size_frac": size, "pick": pick, "regret": t.loc[pick, "true_cost"] - t["true_cost"].min(),
                             "metaorder_impact_error": abs(twap_peak - t.loc["P0_twap", "true_twap_peak"]), **{k: real[k] for k in ("order_book", "price", "impact", "trade_response_error")},
                             **{f"pred_{p}": c for p, c in means.items()}})
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "goal2_simulators.csv", index=False)
    measures = ["order_book", "price", "impact", "trade_response_error", "metaorder_impact_error"]
    # A1: rank correlation of regret with each realism measure, ranks taken within world and size
    r = t.copy()
    for c in measures + ["regret"]:
        r[c] = r.groupby(["world", "size_frac"])[c].rank()
    a1 = pd.DataFrame([{"measure": m, "spearman_pooled": spearmanr(r[m], r["regret"]).correlation,
                        **{f"{w[:3]} {s:g}": spearmanr(g[m], g["regret"]).correlation for (w, s), g in t.groupby(["world", "size_frac"])}} for m in measures])
    a1.to_csv(OUT / "goal2_A1_correlations.csv", index=False)
    print("A1  rank correlation between regret and each realism measure (positive = worse realism goes with higher regret)")
    print(a1.round(2).to_string(index=False))
    # A2: pick the simulator that scores best on each measure
    a2 = []
    for (w, s), g in t.groupby(["world", "size_frac"]):
        tr = truth(w)
        twap_regret = float(tr[(tr["size_frac"] == s) & (tr["policy"] == "P0_twap")]["true_cost"].iloc[0] - tr[tr["size_frac"] == s]["true_cost"].min())
        row = {"world": w, "size_frac": s, "median_regret_all": g["regret"].median(), "always_twap": twap_regret}
        for m in measures:  # LOB-Bench's impact score is undefined where no market order moves the price
            ok = g[m].dropna()
            row[f"best_by_{m}"] = g.loc[ok.idxmin(), "regret"] if len(ok) else np.nan
        # A3: ensembles of predicted costs
        pred = [c for c in g.columns if c.startswith("pred_")]
        tc = tr[tr["size_frac"] == s].set_index("policy")["true_cost"]
        for lab, sub in [("ensemble_all", g), ("ensemble_top5_order_book", g.nsmallest(5, "order_book")), ("ensemble_top5_metaorder", g.nsmallest(5, "metaorder_impact_error"))]:
            p = sub[pred].mean().idxmin().removeprefix("pred_")
            row[lab] = tc[p] - tc.min()
        a2.append(row)
    a2 = pd.DataFrame(a2)
    a2.to_csv(OUT / "goal2_A2_A3_selection.csv", index=False)
    print("\nA2/A3  regret (bps) of the recommendation chosen by each rule")
    print(a2.round(2).to_string(index=False))
    print("\nmean over markets and sizes:")
    print(a2.drop(columns=["world", "size_frac"]).mean().round(2).to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    procs = int(args[args.index("--procs") + 1]) if "--procs" in args else 4
    worlds = args[args.index("--worlds") + 1].split(",") if "--worlds" in args else WORLDS
    skip = {str(procs), ",".join(worlds)} | ({args[args.index("--names") + 1]} if "--names" in args else set()) | ({args[args.index("--seeds") + 1]} if "--seeds" in args else set()) | {args[args.index(f) + 1] for f in ("--seed-from", "--shard") if f in args}
    stages = [a for a in args if not a.startswith("--") and a not in skip] or ["reference", "fit", "realism", "policy", "analyse"]
    RAW.mkdir(parents=True, exist_ok=True)
    names = list(family())
    if "--names" in args:  # restrict the stages to some candidates (e.g. the exploratory qr candidate)
        names = args[args.index("--names") + 1].split(",")
    for st in stages:
        print(f"== {st}", flush=True)
        if st == "reference":
            for w in worlds:
                reference_windows(w)
                print(f"  {w}: {len(reference_windows(w))} held-out windows, true costs for {len(truth(w))} (size, strategy) pairs", flush=True)
        elif st == "fit":
            if [n for n in names if n not in ("qr", "chawkes")]:
                R.pool_map(fit_job, [(w, k, c) for w in worlds for k in ("single", "multi") for c in (0.8, 0.9, 0.95)], procs, "fit")
            if "chawkes" in names:
                R.pool_map(fit_chawkes_job, worlds, procs, "chawkes fit")
            if "qr" in names:
                for w in worlds:
                    print("  ", fit_qr_job(w), flush=True)
        elif st == "realism":
            R.pool_map(realism_job, [(w, n) for w in worlds for n in names if not (RAW / f"realism_{w}_{n}.pkl").exists()], procs, "realism")
        elif st == "policy":
            n_seeds = int(args[args.index("--seeds") + 1]) if "--seeds" in args else POLICIES_SEEDS
            first = int(args[args.index("--seed-from") + 1]) if "--seed-from" in args else 0  # e.g. only the added seeds (addendum 7)
            jobs = [(w, n, SEED_POL + s) for w in worlds for n in names for s in range(first, n_seeds)]
            jobs = [j for j in jobs if not (RAW / "policy" / f"{j[0]}_{j[1]}_{j[2]}.pkl").exists()]
            if "--shard" in args:  # i/n: every n-th job from i, so several machines share the work
                i, n = (int(x) for x in args[args.index("--shard") + 1].split("/"))
                jobs = jobs[i::n]
            R.pool_map(policy_job, jobs, procs, "policy")
        elif st == "analyse":
            analyse()
