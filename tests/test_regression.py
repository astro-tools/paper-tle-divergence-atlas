"""Tests for src/scripts/_regression.py.

The module fits OLS via statsmodels, so the meaningful tests are:

  - Coefficient recovery: build synthetic data with a known linear
    relation and confirm the fitted coefficients match within the
    standard-error band.
  - Schema: the output CSV carries one row per coefficient × model ×
    propagator with the documented column set.
  - Plumbing: the corpus join via `per_sat_elements` produces one row
    per sat with finite (sma, ecc, inc).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

# `_regression` lives under src/scripts/; conftest.py adds it to sys.path.
from _regression import (  # noqa: E402
    build_design,
    fit_to_rows,
    per_sat_elements,
    run_all_regressions,
)

# Real-looking but minimal SGP4 line2 (Starlink ~550 km, 53° inc).
_TLE_LINE1 = "1 44713U 19074A   20043.40194444  .00000928  00000-0  78677-4 0  9990"
_TLE_LINE2 = "2 44713  53.0010 113.4983 0001277  85.5125 274.6028 15.05544316 16823"


def _make_corpus(n_sats: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n_sats):
        for j in range(3):
            rows.append(
                {
                    "norad_id": 44000 + i,
                    "epoch_i": pd.Timestamp("2020-02-12", tz="UTC") + pd.Timedelta(days=j),
                    "line1_i": _TLE_LINE1,
                    "line2_i": _TLE_LINE2,
                    "sma_i_km": 6921.0 + 0.1 * i,
                }
            )
    return pd.DataFrame(rows)


def _make_all_runs(n_sats: int = 6, *, dr_floor: float = 0.05) -> pd.DataFrame:
    """Synthetic all_runs with known log-log relations to recover.

    log(dr_sgp4) = -2 + 1.2 * log10(actual_dt_hours) + 0.005 * f107
    so the continuous-Δt regression should recover slopes near
    (1.2, 0.005) for (log_dt, f107).
    """
    rng = np.random.default_rng(42)
    rows = []
    buckets = [21600, 86400, 259200, 604800]
    base_epoch = pd.Timestamp("2020-02-12", tz="UTC")
    for i in range(n_sats):
        norad = 44000 + i
        # Spread sats across both generations so the categorical is full-rank.
        gen = "v1.5" if i % 2 == 0 else "v2-mini"
        shell = "540" if i % 3 == 0 else "550"
        for j in range(8):
            for bucket in buckets:
                f107 = float(rng.uniform(70, 200))
                ap = float(rng.uniform(2, 30))
                hours = bucket / 3600.0
                log_dr = -2.0 + 1.2 * math.log10(hours) + 0.005 * f107
                noise = rng.normal(0, 0.05)
                dr = 10 ** (log_dr + noise)
                rows.append(
                    {
                        "norad_id": norad,
                        "target_dt_sec": bucket,
                        "actual_dt_sec": float(bucket) * float(rng.uniform(0.98, 1.02)),
                        "alt_shell": shell,
                        "generation": gen,
                        "t_i": base_epoch + pd.Timedelta(days=j),
                        "dr_sgp4_km": max(dr, dr_floor),
                        "dr_hifi_km": max(dr * 1.1, dr_floor),
                        "f107": f107,
                        "ap": ap,
                    }
                )
    return pd.DataFrame(rows)


class TestPerSatElements:
    def test_one_row_per_sat(self):
        corpus = _make_corpus(n_sats=4)
        out = per_sat_elements(corpus)
        assert len(out) == 4
        assert set(out["norad_id"]) == {44000, 44001, 44002, 44003}

    def test_finite_orbital_elements(self):
        out = per_sat_elements(_make_corpus())
        assert np.isfinite(out["sma"]).all()
        assert np.isfinite(out["ecc"]).all()
        assert np.isfinite(out["inc"]).all()
        # Starlink-ish inclination.
        assert (out["inc"] > 45).all()
        assert (out["inc"] < 60).all()

    def test_first_epoch_only_used(self):
        """Per-sat selection picks the earliest cached epoch."""
        corpus = _make_corpus(n_sats=2)
        # Surgically change sma on the later row — should not be picked.
        late_mask = corpus["epoch_i"] != corpus["epoch_i"].min()
        corpus.loc[late_mask, "sma_i_km"] = 9999.0
        out = per_sat_elements(corpus).set_index("norad_id")["sma"]
        assert (out < 9999.0).all()


class TestBuildDesign:
    def test_drops_nonpositive_error(self):
        all_runs = _make_all_runs(n_sats=4)
        all_runs.loc[:5, "dr_sgp4_km"] = 0.0
        design = build_design(all_runs, _make_corpus(n_sats=4), "dr_sgp4_km")
        assert (design["dr_sgp4_km"] > 0).all()
        assert "log_dr" in design.columns
        assert "log_dt" in design.columns
        assert "gen_pooled" in design.columns

    def test_log_columns_match_expectations(self):
        all_runs = _make_all_runs(n_sats=4)
        design = build_design(all_runs, _make_corpus(n_sats=4), "dr_sgp4_km")
        np.testing.assert_allclose(
            design["log_dr"].to_numpy(),
            np.log10(design["dr_sgp4_km"].to_numpy()),
        )


class TestRunAllRegressions:
    @pytest.fixture
    def results(self):
        return run_all_regressions(_make_all_runs(n_sats=6), _make_corpus(n_sats=6))

    def test_one_row_per_predictor_per_model_per_propagator(self, results):
        # bucket + continuous, each fit for sgp4 + hifi → 4 fits total
        assert set(results["model"]) == {"bucket", "continuous"}
        assert set(results["propagator"]) == {"sgp4", "hifi"}
        # Same predictor set per (model, propagator) cell.
        per_cell = results.groupby(["model", "propagator"])["predictor"].nunique()
        assert per_cell.nunique() == 2  # one count for bucket, one for continuous

    def test_continuous_recovers_log_dt_slope(self, results):
        """Synthetic data has true log_dt slope = 1.2; recover within tolerance."""
        row = results[
            (results["model"] == "continuous")
            & (results["propagator"] == "sgp4")
            & (results["predictor"] == "log_dt")
        ].iloc[0]
        assert row["coefficient"] == pytest.approx(1.2, abs=0.1)
        assert row["p_value"] < 1e-6

    def test_continuous_recovers_f107_slope(self, results):
        """Synthetic data has true f107 slope = 0.005 in the reference gen."""
        row = results[
            (results["model"] == "continuous")
            & (results["propagator"] == "sgp4")
            & (results["predictor"] == "f107")
        ].iloc[0]
        assert row["coefficient"] == pytest.approx(0.005, abs=0.001)

    def test_r_squared_is_repeated_per_row_within_a_fit(self, results):
        for (_, _), sub in results.groupby(["model", "propagator"]):
            r2 = sub["r_squared"].to_numpy()
            assert (r2 == r2[0]).all()
            assert 0.0 <= r2[0] <= 1.0

    def test_n_obs_positive(self, results):
        assert (results["n_obs"] > 0).all()


class TestFitToRows:
    def test_row_columns(self):
        all_runs = _make_all_runs(n_sats=4)
        design = build_design(all_runs, _make_corpus(n_sats=4), "dr_sgp4_km")
        import statsmodels.formula.api as smf

        fit = smf.ols("log_dr ~ log_dt + f107", data=design).fit()
        rows = fit_to_rows(fit, "test", "sgp4")
        assert len(rows) == 3  # intercept + log_dt + f107
        for r in rows:
            assert set(r.keys()) >= {
                "model",
                "propagator",
                "predictor",
                "coefficient",
                "std_err",
                "p_value",
                "ci_lo",
                "ci_hi",
                "r_squared",
                "n_obs",
            }
