"""
Checks of the ported LOB-Impact estimators (simrel.metrics.lob_impact) on planted laws, as in the paper's validation:
a law of known exponent is recovered (the binned estimator is biased slightly downward), pure noise identifies no law,
and the phase-1 statistics recover known sign memory and diffusion.

    python -m pytest simrel/tests/test_lob_impact.py -q
"""

import numpy as np
import pytest

from simrel.metrics import lob_impact as LI

DAYS = [f"D{k}" for k in range(8)]
DAILY = {d: {"V": 3e6, "range": 0.02} for d in DAYS}


def planted(delta, amp_y, noise_bps, n_roll=64, child=100, seed=0):
    """Records whose impact after child k is amp_y * sigma * (Q_k / V)^delta plus noise (mid 50 USD)."""
    rng = np.random.default_rng(seed)
    sigma = 0.02 / LI.KAPPA_P
    recs = []
    for r in range(n_roll):
        sign = 1 if r % 2 == 0 else -1
        q = child * np.arange(1, 101)
        imp = amp_y * LI.KAPPA_P * sigma * (q / 3e6) ** delta * (q[-1] / 3e6) ** (0.5 - delta)  # Y defined at the data edge
        mid_ref = 500_000.0
        mid = mid_ref * (1 + sign * imp + rng.normal(0, noise_bps * 1e-4, 100))
        recs.append({"day": DAYS[r % 8], "sign": sign, "child_q": np.full(100, float(child)), "mid_ref": mid_ref, "mid_child": mid,
                     "kmid": mid})
    return recs


@pytest.mark.parametrize("delta", [0.5, 0.7, 1.0])
def test_planted_law_is_recovered(delta):
    res = LI.build_up(LI.points(planted(delta, 1.0, 1.0)), DAILY)
    assert res["identified"]
    assert res["delta"] == pytest.approx(delta, abs=0.12)
    assert res["delta_lo"] <= res["delta"] + 1e-9 <= res["delta_hi"] + 0.05


def test_noise_gives_no_law():
    res = LI.build_up(LI.points(planted(0.5, 0.0, 3.0)), DAILY)
    assert not res["identified"]


def test_peak_detection():
    K, _ = LI.k_clock(planted(0.5, 1.0, 1.0), 100, 0)
    p = LI.peak(K, 100)
    assert p["detected"] and p["peak_bps"] > 0


def test_phase1_recovers_memory_and_diffusion():
    rng = np.random.default_rng(1)
    recs = []
    for _ in range(40):
        n = 3000
        # persistent signs: runs with Pareto lengths (long memory); a random-walk mid in ticks (H = 0.5)
        eps, s = [], 1
        while len(eps) < n:
            s = rng.choice((-1, 1))
            eps += [s] * int(np.ceil(rng.pareto(1.5) + 1))
        eps = np.array(eps[:n], dtype=np.int8)
        tmid = 500_000.0 + 100 * np.cumsum(rng.choice((-1, 0, 0, 1), n))
        recs.append({"eps": eps, "tmid": tmid, "n_rows": n, "mid_first": tmid[0], "mid_last": tmid[-1]})
    p1 = LI.phase1(recs)
    assert 0.2 < p1["gamma_flow"] < 1.0
    assert p1["H"] == pytest.approx(0.5, abs=0.05)
