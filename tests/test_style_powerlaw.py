"""Tests for the §3.7.1 per-pair power-law estimator in `_style.py`.

`fit_powerlaw_perpair` is the v0.1.0 main estimator for the F5 /
Table~\\ref{tab:powerlaw} entries, replacing the per-bucket-median
`fit_powerlaw` that the underscored helper retains for F7. The tests
exercise three properties the §3.7.1 prose claims:

  * coefficient recovery on clean log-log data within tight tolerance;
  * $R^{2} \\to 1$ on noiseless data, and degrades smoothly with noise;
  * LRT picks up $k \\neq 1$ and $k \\neq 2$ with high power on
    well-populated cells while declining to reject when $k$ actually
    sits at the null.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from _style import fit_powerlaw_perpair


def _make_cell(
    *,
    n_sats: int = 30,
    pairs_per_sat: int = 16,
    A: float = 0.2,  # noqa: N803 — math notation
    k: float = 1.3,
    noise: float = 0.0,
    seed: int = 0,
    dt_jitter: float = 0.05,
) -> pd.DataFrame:
    """Synthetic cell with known (A, k) and per-sat-jittered offsets.

    Pairs span the four canonical staleness buckets multiplied by a
    small random factor inside the $\\pm 2$ h tolerance, so the
    `actual_dt_sec` spread is realistic. Optional per-sat intercept
    shift (controlled by `noise`) simulates the within-sat correlation
    the satellite-level bootstrap is designed to absorb; the per-pair
    estimator should still recover the population slope.
    """
    rng = np.random.default_rng(seed)
    buckets = np.array([21600, 86400, 259200, 604800], dtype=float)
    rows: list[dict] = []
    for sat_idx in range(n_sats):
        sat_offset = rng.normal(0.0, noise * 0.3)  # per-sat intercept shift
        norad = 44000 + sat_idx
        for _ in range(pairs_per_sat):
            base = float(rng.choice(buckets))
            actual = base * (1.0 + rng.uniform(-dt_jitter, dt_jitter))
            hours = actual / 3600.0
            log_dr = math.log10(A) + k * math.log10(hours) + sat_offset
            log_dr += float(rng.normal(0.0, noise))
            rows.append(
                {
                    "norad_id": norad,
                    "target_dt_sec": int(base),
                    "actual_dt_sec": actual,
                    "dr": 10**log_dr,
                }
            )
    return pd.DataFrame(rows)


class TestCoefficientRecovery:
    def test_clean_recovery(self):
        df = _make_cell(A=0.25, k=1.4, noise=0.0)
        fit = fit_powerlaw_perpair(df, "dr")
        assert fit["A"] == pytest.approx(0.25, rel=1e-6)
        assert fit["k"] == pytest.approx(1.4, rel=1e-6)
        assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)
        assert fit["n_pairs"] == 30 * 16

    def test_noisy_recovery(self):
        df = _make_cell(A=0.1, k=0.9, noise=0.15, seed=11)
        fit = fit_powerlaw_perpair(df, "dr")
        # Slope and intercept recovered within Gaussian noise budget at
        # n = 480 pairs.
        assert fit["k"] == pytest.approx(0.9, abs=0.05)
        assert fit["A"] == pytest.approx(0.1, rel=0.20)
        # R² degrades but stays solidly informative.
        assert 0.6 < fit["r_squared"] < 0.99


class TestLikelihoodRatio:
    def test_lrt_rejects_k1_when_k_is_two(self):
        df = _make_cell(k=2.0, noise=0.05, seed=7)
        fit = fit_powerlaw_perpair(df, "dr")
        # LRT vs. k = 1 should overwhelmingly reject; vs. k = 2 should not.
        assert fit["p_lrt_k1"] < 1e-3
        assert fit["p_lrt_k2"] > 0.1

    def test_lrt_rejects_k2_when_k_is_one(self):
        df = _make_cell(k=1.0, noise=0.05, seed=9)
        fit = fit_powerlaw_perpair(df, "dr")
        assert fit["p_lrt_k2"] < 1e-3
        assert fit["p_lrt_k1"] > 0.1

    def test_lrt_under_null_does_not_reject(self):
        # True slope sits at k = 1.5; both nulls should be rejected
        # cleanly, but at k exactly 1 or 2 the corresponding p-value is
        # not extremely small.  Use a controlled setup where the slope
        # is 1.0 and verify p_lrt_k1 is well above 0.05.
        df = _make_cell(k=1.0, noise=0.05, seed=2)
        fit = fit_powerlaw_perpair(df, "dr")
        assert fit["p_lrt_k1"] > 0.05


class TestEdgeCases:
    def test_drops_nonpositive_dr(self):
        df = _make_cell(A=0.2, k=1.3, noise=0.05, seed=3)
        df.loc[df.index[:50], "dr"] = -1.0  # would explode log10
        fit = fit_powerlaw_perpair(df, "dr")
        # Recovery survives the partial drop.
        assert fit["n_pairs"] == len(df) - 50
        assert fit["k"] == pytest.approx(1.3, abs=0.1)

    def test_raises_on_too_few_pairs(self):
        df = _make_cell(n_sats=2, pairs_per_sat=2)
        with pytest.raises(ValueError):
            fit_powerlaw_perpair(df, "dr", min_pairs=8)

    def test_raises_on_degenerate_dt(self):
        df = _make_cell(n_sats=10, pairs_per_sat=10)
        df["actual_dt_sec"] = 86400.0  # single Δt → zero design variance
        with pytest.raises(ValueError):
            fit_powerlaw_perpair(df, "dr")
