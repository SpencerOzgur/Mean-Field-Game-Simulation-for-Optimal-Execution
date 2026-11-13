"""
Design addendum 11, round 2: tables, the rule's verdict and one figure.

Uses the pilot's methods unchanged (reactive_pilot_report.e2, e3, e4) on the round-2 stage files, adds the predator's
P&L components, the counterparty shares and the gate statistics, and applies the committed rule
(simrel/reports/addendum11_round2_rule.md) mechanically.

    python simrel/experiments/reactive_round2.py report
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import reactive_pilot as P  # noqa: E402
from simrel.experiments import reactive_pilot_report as PR  # noqa: E402
from simrel.experiments import reactive_round2 as R2  # noqa: E402

OUT, RAW = R2.OUT, R2.RAW
A03 = "XRP_pilot_a03"
KINDS = ["entry", "passive", "flatten", "unwind"]
PILOT_NAME = {"off": "off", "aggressive_medium": "medium", "aggressive_strong": "strong"}


def load(name: str):
    return pickle.load(open(RAW / f"{name}.pkl", "rb"))


def _frame(jobs: list) -> pd.DataFrame:
    """Rows of the truth stage (one list of rows per job, or already flat), with failed = NaN where the run succeeded."""
    rows = [r for x in jobs for r in (x if isinstance(x, list) else [x])]
    df = pd.DataFrame(rows)
    if "failed" not in df:
        df["failed"] = np.nan
    df["failed"] = df["failed"].where(df["failed"].astype(bool), np.nan)
    return df


def _use_round2(arms: list) -> None:
    """Point the pilot's report functions at the round-2 files and arms."""
    PR.RAW, P.ARMS, P.TRUTH_ARMS = RAW, arms, arms


def check_against_pilot(truth: pd.DataFrame) -> pd.DataFrame:
    """The off and aggressive arms must reproduce the pilot's costs exactly (the aggressive predator is unchanged)."""
    path = P.OUT / "raw" / "truth.pkl"
    if not path.exists():
        return pd.DataFrame()
    old = _frame(pickle.load(open(path, "rb")))
    key = ["market", "size_frac", "seed", "policy"]
    rows = []
    for arm, name in PILOT_NAME.items():
        a = truth[(truth["arm"] == arm) & (truth["policy"] != "none")].set_index(key)
        b = old[(old["arm"] == name) & (old["policy"] != "none")].set_index(key)
        j = a[["is_bps", "pred_pnl_ticks"]].join(b[["is_bps", "pred_pnl_ticks"]], rsuffix="_pilot", how="inner")
        rows.append({"arm": arm, "pilot_arm": name, "n": len(j), "max_abs_cost_diff_bps": float((j["is_bps"] - j["is_bps_pilot"]).abs().max()),
                     "max_abs_pnl_diff_ticks": float((j["pred_pnl_ticks"] - j["pred_pnl_ticks_pilot"]).abs().max())})
    return pd.DataFrame(rows)


def predator_account(truth: pd.DataFrame) -> pd.DataFrame:
    """Per market, arm, size: the predator's account against TWAP and over all strategies, in bp of order value."""
    t = truth[truth["failed"].isna() & (truth["policy"] != "none")].copy()
    scale = 1e4 / (t["qty"] * t["arrival_mid"])
    t["pnl_bps"] = t["pred_pnl_ticks"] * scale
    for k in KINDS:
        t[f"{k}_bps"] = t[f"pred_{k}_edge_ticks"] * scale
    t["hold_bps"] = t["pred_hold_pnl_ticks"] * scale
    rows = []
    for which, d0 in (("P0_twap", t[t["policy"] == "P0_twap"]), ("all strategies", t)):
        for (m, a, f), d in d0.groupby(["market", "arm", "size_frac"]):
            buy = d["pred_passive_volume"].sum()
            exits = buy + d["pred_flatten_volume"].sum() + d["pred_unwind_volume"].sum()
            row = {"against": which, "market": m, "arm": a, "size_frac": f, "n": len(d), "pnl_bps": d["pnl_bps"].mean(), "pnl_se": PR.se(d["pnl_bps"]),
                   **{f"{k}_bps": d[f"{k}_bps"].mean() for k in KINDS}, "hold_bps": d["hold_bps"].mean(),
                   "entry_volume_over_qty": (d["pred_entry_volume"] / d["qty"]).mean(),
                   "trader_share_of_passive_buyback": d["fill_by_trader"].sum() / buy if buy else np.nan,
                   "aggressive_share_of_exit_volume": (d["pred_flatten_volume"].sum() + d["pred_unwind_volume"].sum()) / exits if exits else np.nan,
                   "bg_cancel_over_passive_fill": d["bg_cancel"].sum() / buy if buy else np.nan,
                   "gate_block_rate": d["pred_gate_blocks"].sum() / d["pred_gate_checks"].sum() if d["pred_gate_checks"].sum() else np.nan,
                   "share_runs_all_entries_blocked": float(((d["pred_gate_checks"] > 0) & (d["pred_entry_volume"] == 0)).mean()),
                   "share_runs_with_entry": float((d["pred_entry_volume"] > 0).mean()), "trigger_rate": d["trig_task"].mean(),
                   "residual_inventory_share": float((d["pred_inv_end"] != 0).mean())}
            rows.append(row)
    out = pd.DataFrame(rows)
    out["arm"] = pd.Categorical(out["arm"], R2.ARMS, ordered=True)
    return out.sort_values(["against", "market", "size_frac", "arm"])


def pooled(acct: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """Mean over the five markets other than a03 (equal runs per market, so this is the pooled mean), by arm and size."""
    a = acct[(acct["against"] == "P0_twap") & (acct["market"] != A03)]
    s = summ[summ["market"] != A03]
    cols = ["pnl_bps", *[f"{k}_bps" for k in KINDS], "hold_bps", "entry_volume_over_qty", "trader_share_of_passive_buyback",
            "aggressive_share_of_exit_volume", "bg_cancel_over_passive_fill", "gate_block_rate", "share_runs_all_entries_blocked", "share_runs_with_entry"]
    left = a.groupby(["size_frac", "arm"], observed=True)[cols].mean()
    right = s.groupby(["size_frac", "arm"], observed=True)[["twap_cost_minus_off_bps", "twap_regret_bps", "twap_regret_crossfit_bps", "p0r_minus_twap_bps"]].mean()
    return right.join(left).reset_index()


def verdict(truth: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """The committed rule, applied as written. P&L is pooled over all TWAP runs of the five markets at 15%."""
    t = truth[truth["failed"].isna() & (truth["policy"] == "P0_twap") & (truth["size_frac"] == 0.15) & (truth["market"] != A03)].copy()
    t["pnl_bps"] = t["pred_pnl_ticks"] / (t["qty"] * t["arrival_mid"]) * 1e4
    rows = []
    for arm in R2.ARMS[1:]:
        d = t[t["arm"] == arm]
        s = summ[(summ["arm"] == arm) & (summ["size_frac"] == 0.15) & (summ["market"] != A03)].set_index("market")["twap_cost_minus_off_bps"]
        a = float(d["pnl_bps"].mean())
        n_b = int((s >= 1.0).sum())
        judged = arm in R2.RATIONAL_ARMS
        rows.append({"arm": arm, "judged_by_rule": judged, "runs": len(d), "a_pooled_pnl_vs_twap_bps": a, "a_pnl_se": PR.se(d["pnl_bps"]), "a_ok": a >= 0.0,
                     "b_markets_with_twap_rise_ge_1bp": n_b, "b_ok": n_b >= 3, "passes": bool(a >= 0.0 and n_b >= 3),
                     **{f"twap_rise_{m[-3:]}_bps": float(s[m]) for m in s.index}})
    return pd.DataFrame(rows)


def false_triggers() -> pd.DataFrame:
    pa = pd.DataFrame(load("passive"))
    rows = []
    for (m, a), d in pa.groupby(["market", "arm"]):
        buy = d["pred_passive_volume"].sum()
        rows.append({"market": m, "arm": a, "episodes": len(d), "triggers_per_episode": d["pred_triggers"].mean(),
                     "share_episodes_with_trigger": (d["pred_triggers"] > 0).mean(), "share_episodes_with_predator_trade": (d["pred_volume"] > 0).mean(),
                     "pred_volume_share": d["pred_volume"].sum() / d["market_volume"].sum(), "gate_block_rate": d["pred_gate_blocks"].sum() / max(d["pred_gate_checks"].sum(), 1),
                     "aggressive_share_of_exit_volume": d["pred_flatten_volume"].sum() / (buy + d["pred_flatten_volume"].sum()) if buy + d["pred_flatten_volume"].sum() else np.nan,
                     "residual_inventory_share": (d["pred_inv_end"] != 0).mean()})
    out = pd.DataFrame(rows)
    out["arm"] = pd.Categorical(out["arm"], R2.ARMS, ordered=True)
    return out.sort_values(["market", "arm"])


def figure(pool: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = R2.ARMS[1:]
    d = pool[pool["size_frac"] == 0.15].set_index("arm").loc[arms]
    y = np.arange(len(arms))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, col, title, thr in ((axes[0], "pnl_bps", "Predator P&L against TWAP (bp of order value)", 0.0),
                                (axes[1], "twap_cost_minus_off_bps", "TWAP cost minus predator off (bp)", 1.0)):
        ax.barh(y, d[col], height=0.6, color=["#2a78d6" if a in R2.RATIONAL_ARMS else "#a9c7ee" for a in arms])
        ax.axvline(thr, color=PR.INK, linewidth=0.9, linestyle="--" if thr else "-")
        for yi, v in zip(y, d[col]):
            ax.text(v + (0.12 if v >= 0 else -0.12), yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=PR.MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels(arms)
        PR._style(ax, title, "")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.18)
    axes[1].text(1.0, len(arms) - 0.45, " rule: 1.0 bp in 3 of 5 markets", fontsize=7.5, color=PR.MUTED)
    fig.suptitle("Round 2, size 15%, mean over the five markets other than a03 (dark bars: the arms the rule judges)", fontsize=10, color=PR.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_round2_rule.png", dpi=160)
    plt.close(fig)


def main() -> None:
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 300)
    truth = _frame(load("truth"))
    print("failed truth runs:", int(truth["failed"].notna().sum()), "of", len(truth))
    chk = check_against_pilot(truth)
    chk.to_csv(OUT / "check_against_pilot.csv", index=False)
    print(chk.to_string(index=False))

    _use_round2(R2.ARMS)
    e4 = PR.e4()
    e4.to_csv(OUT / "e4_viability.csv", index=False)
    costs, summ = PR.e2(truth)
    costs.to_csv(OUT / "e2_costs.csv", index=False)
    summ.to_csv(OUT / "e2_summary.csv", index=False)
    acct = predator_account(truth)
    acct.to_csv(OUT / "e2_predator_account.csv", index=False)
    pool = pooled(acct, summ)
    pool.to_csv(OUT / "e2_pooled_five_markets.csv", index=False)
    ver = verdict(truth, summ)
    ver.to_csv(OUT / "verdict.csv", index=False)
    false = false_triggers()
    false.to_csv(OUT / "e1_false_triggers_passive.csv", index=False)
    _use_round2(["off"] + R2.NEW_ARMS)
    raw, diffs, aucs, _ = PR.e3()
    raw.to_csv(OUT / "e3_samples.csv", index=False)
    diffs.to_csv(OUT / "e3_feature_differences.csv", index=False)
    aucs.to_csv(OUT / "e3_auc.csv", index=False)
    figure(pool)

    print("\nVERDICT (rule applied as committed)\n", ver.round(3).to_string(index=False))
    print("\npooled, five markets\n", pool.round(3).to_string(index=False))
    a3 = acct[(acct["against"] == "P0_twap") & (acct["market"] == A03)].merge(summ[summ["market"] == A03][["arm", "size_frac", "twap_cost_minus_off_bps", "twap_cost_minus_off_se"]], on=["arm", "size_frac"])
    print("\na03\n", a3[["arm", "size_frac", "pnl_bps", "pnl_se", "twap_cost_minus_off_bps", "twap_cost_minus_off_se", "trader_share_of_passive_buyback", "aggressive_share_of_exit_volume"]].round(2).to_string(index=False))
    print("\nper market, 15%\n", acct[(acct["against"] == "P0_twap") & (acct["size_frac"] == 0.15)][["market", "arm", "pnl_bps", "pnl_se", "trader_share_of_passive_buyback", "aggressive_share_of_exit_volume", "gate_block_rate", "share_runs_with_entry"]].round(2).to_string(index=False))
    print("\nE4\n", e4[["market", "arm", "fail_share", "wild_share", "viable", "volume", "spread_bps", "twin_pred_volume_share"]].round(3).to_string(index=False))
    print("\nfalse triggers\n", false.round(3).to_string(index=False))
    print("\nE3 pooled\n", aucs[aucs["market"] == "pooled"].round(3).to_string(index=False))
    print("\nbest policy\n", summ.pivot_table(index=["market", "size_frac"], columns="arm", values="best_policy", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
