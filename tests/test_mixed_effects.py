"""Tests for `src/scripts/_mixed_effects.py`.

The script fits a per-cell linear mixed-effects model via
`statsmodels.formula.api.mixedlm`; tests confirm the per-cell loop
produces a populated row per (alt_shell × gen_pooled × propagator) and
that synthetic data with a known fixed-effect slope recovers that slope
within the asymptotic Wald CI.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from _mixed_effects import _fit_cell, run_all_cells


def _make_runs(
    *,
    n_sats: int = 12,
    pairs_per_sat: int = 24,
    A: float = 0.2,  # noqa: N803 — math notation
    k: float = 1.3,
    sat_intercept_sd: float = 0.2,
    noise: float = 0.05,
    seed: int = 0,
    shell: str = "550",
    generation: str = "v1.5",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    buckets = np.array([21600, 86400, 259200, 604800], dtype=float)
    rows: list[dict] = []
    for sat_idx in range(n_sats):
        norad = 44000 + sat_idx
        sat_shift = rng.normal(0.0, sat_intercept_sd)
        for _ in range(pairs_per_sat):
            base = float(rng.choice(buckets))
            actual = base * (1.0 + rng.uniform(-0.05, 0.05))
            hours = actual / 3600.0
            log_dr_sgp4 = math.log10(A) + k * math.log10(hours) + sat_shift
            log_dr_sgp4 += float(rng.normal(0.0, noise))
            log_dr_hifi = log_dr_sgp4 + math.log10(1.5)  # offset only
            rows.append(
                {
                    "norad_id": norad,
                    "alt_shell": shell,
                    "generation": generation,
                    "target_dt_sec": int(base),
                    "actual_dt_sec": actual,
                    "dr_sgp4_km": 10**log_dr_sgp4,
                    "dr_hifi_km": 10**log_dr_hifi,
                }
            )
    return pd.DataFrame(rows)


class TestFitCell:
    def test_recovers_slope(self):
        df = _make_runs(k=1.4, seed=2)
        fit = _fit_cell(df, "dr_sgp4_km")
        assert fit is not None
        assert fit["k_fe"] == pytest.approx(1.4, abs=0.10)
        assert fit["k_fe_se"] > 0
        assert fit["n_pairs"] == 12 * 24
        assert fit["n_sats"] == 12

    def test_recovers_random_intercept_sd(self):
        df = _make_runs(sat_intercept_sd=0.3, noise=0.02, seed=4)
        fit = _fit_cell(df, "dr_sgp4_km")
        assert fit is not None
        # mixedlm reports the random-intercept SD on `log_dr`; with
        # n_sats = 12 and the asymptotic estimator there is non-trivial
        # variance, so use a generous tolerance.
        assert fit["re_sd_log10A"] == pytest.approx(0.3, abs=0.15)

    def test_skips_thin_cell(self):
        df = _make_runs(n_sats=3)
        # `_fit_cell` requires ≥ 5 sats; should skip and return None.
        assert _fit_cell(df, "dr_sgp4_km") is None


class TestRunAllCells:
    @pytest.fixture
    def results(self):
        return run_all_cells(_make_runs(n_sats=12, pairs_per_sat=24))

    def test_schema(self, results):
        # One ok row per propagator on the populated 550 × v1.5 cell, with
        # `skipped` rows for the empty (alt_shell × gen_pooled) cells.
        assert {"alt_shell", "gen_pooled", "propagator", "status"} <= set(results.columns)
        ok = results[results["status"] == "ok"]
        assert set(ok["propagator"]) == {"sgp4", "hifi"}
        assert set(ok["alt_shell"]) == {"550"}

    def test_ok_rows_carry_full_payload(self, results):
        ok = results[results["status"] == "ok"]
        for col in ("k_fe", "k_fe_se", "k_fe_ci_lo", "k_fe_ci_hi", "n_pairs", "n_sats"):
            assert ok[col].notna().all(), f"{col} should be populated on ok rows"

    def test_wald_ci_brackets_point(self, results):
        ok = results[results["status"] == "ok"]
        # 95% Wald interval bounds the point estimate by definition.
        assert (ok["k_fe_ci_lo"] <= ok["k_fe"]).all()
        assert (ok["k_fe"] <= ok["k_fe_ci_hi"]).all()
