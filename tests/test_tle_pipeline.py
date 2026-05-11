"""Unit tests for sweep.tle_pipeline.

Pure-Python only; no network, no SGP4 propagation. Synthetic TLE lines are
crafted to exercise the mean-motion column parser and the SMA→shell mapper.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sweep.tle_pipeline import (
    DEFAULT_MANEUVER_THRESHOLD_KM,
    EARTH_RADIUS_KM,
    altitude_shell,
    build_corpus,
    build_pairs,
    filter_maneuvers,
    sma_km_from_mean_motion,
    stratified_sample,
)

# A real Starlink line 2 (STARLINK-1007); the parser only inspects columns
# 53..63 (0-indexed 52:63), where mean motion in rev/day is encoded.
LINE2_550KM = "2 44713  53.0540 116.6831 0001247  82.6886 277.4255 15.06405853250789"
LINE2_540KM = "2 44713  53.0540 116.6831 0001247  82.6886 277.4255 15.10000000250789"
LINE2_560KM = "2 44713  53.0540 116.6831 0001247  82.6886 277.4255 15.02000000250789"


def _make_tle_row(norad_id: int, epoch: str, line2: str, line1: str = "1") -> dict:
    return {
        "norad_id": norad_id,
        "epoch": pd.Timestamp(epoch, tz="UTC"),
        "line1": line1,
        "line2": line2,
    }


class TestSmaFromMeanMotion:
    def test_known_starlink_mean_motion_lands_near_550_km_altitude(self) -> None:
        # 15.06 rev/day corresponds to ~6928 km SMA, i.e. ~550 km altitude.
        sma = sma_km_from_mean_motion(15.06405853)
        altitude = sma - EARTH_RADIUS_KM
        assert 545.0 <= altitude <= 555.0, f"expected 550-km shell, got altitude {altitude:.2f} km"

    def test_higher_mean_motion_means_lower_altitude(self) -> None:
        assert sma_km_from_mean_motion(15.10) < sma_km_from_mean_motion(15.06)


class TestAltitudeShell:
    def test_550_band_assignment(self) -> None:
        sma = EARTH_RADIUS_KM + 550.0
        assert altitude_shell(sma) == "550"

    def test_below_lowest_shell_is_unassigned(self) -> None:
        sma = EARTH_RADIUS_KM + 400.0
        assert altitude_shell(sma) is None

    def test_above_highest_shell_is_unassigned(self) -> None:
        sma = EARTH_RADIUS_KM + 700.0
        assert altitude_shell(sma) is None

    def test_shells_are_disjoint(self) -> None:
        boundary = EARTH_RADIUS_KM + 547.0  # right on the 540/550 boundary
        # Lower band is half-open [low, high) so 547.0 falls in "550" by spec.
        assert altitude_shell(boundary) == "550"


class TestBuildPairs:
    def test_three_tles_produce_two_pairs(self) -> None:
        tles = pd.DataFrame(
            [
                _make_tle_row(1, "2026-04-01T00:00:00Z", LINE2_550KM),
                _make_tle_row(1, "2026-04-02T00:00:00Z", LINE2_550KM),
                _make_tle_row(1, "2026-04-03T00:00:00Z", LINE2_550KM),
            ],
        )
        pairs = build_pairs(tles)
        assert len(pairs) == 2

    def test_pairs_respect_per_sat_grouping(self) -> None:
        tles = pd.DataFrame(
            [
                _make_tle_row(1, "2026-04-01T00:00:00Z", LINE2_550KM),
                _make_tle_row(1, "2026-04-02T00:00:00Z", LINE2_550KM),
                _make_tle_row(2, "2026-04-01T00:00:00Z", LINE2_550KM),
                _make_tle_row(2, "2026-04-02T00:00:00Z", LINE2_550KM),
            ],
        )
        pairs = build_pairs(tles)
        assert len(pairs) == 2  # one per sat, not four
        assert set(pairs["norad_id"].unique()) == {1, 2}

    def test_dt_sec_is_positive_and_correct(self) -> None:
        tles = pd.DataFrame(
            [
                _make_tle_row(1, "2026-04-01T00:00:00Z", LINE2_550KM),
                _make_tle_row(1, "2026-04-01T06:00:00Z", LINE2_550KM),
            ],
        )
        pairs = build_pairs(tles)
        assert pairs.iloc[0]["dt_sec"] == pytest.approx(6 * 3600)

    def test_missing_required_column_raises(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            build_pairs(
                pd.DataFrame({"norad_id": [1], "epoch": [pd.Timestamp("2026-04-01", tz="UTC")]})
            )

    def test_sma_jump_zero_for_identical_line2(self) -> None:
        tles = pd.DataFrame(
            [
                _make_tle_row(1, "2026-04-01T00:00:00Z", LINE2_550KM),
                _make_tle_row(1, "2026-04-02T00:00:00Z", LINE2_550KM),
            ],
        )
        pairs = build_pairs(tles)
        assert pairs.iloc[0]["sma_jump_km"] == pytest.approx(0.0, abs=1e-9)


class TestFilterManeuvers:
    def test_default_threshold_keeps_quiet_pair(self) -> None:
        tles = pd.DataFrame(
            [
                _make_tle_row(1, "2026-04-01T00:00:00Z", LINE2_550KM),
                _make_tle_row(1, "2026-04-02T00:00:00Z", LINE2_550KM),
            ],
        )
        kept = filter_maneuvers(build_pairs(tles))
        assert len(kept) == 1

    def test_default_threshold_drops_pair_spanning_an_sma_change(self) -> None:
        # A jump from 550-km mean motion to 540-km mean motion is ~10 km of
        # SMA change — well above the 100-m default threshold.
        tles = pd.DataFrame(
            [
                _make_tle_row(1, "2026-04-01T00:00:00Z", LINE2_550KM),
                _make_tle_row(1, "2026-04-02T00:00:00Z", LINE2_540KM),
            ],
        )
        kept = filter_maneuvers(build_pairs(tles))
        assert len(kept) == 0

    def test_custom_threshold_can_let_a_jump_through(self) -> None:
        tles = pd.DataFrame(
            [
                _make_tle_row(1, "2026-04-01T00:00:00Z", LINE2_550KM),
                _make_tle_row(1, "2026-04-02T00:00:00Z", LINE2_540KM),
            ],
        )
        kept = filter_maneuvers(build_pairs(tles), sma_jump_threshold_km=20.0)
        assert len(kept) == 1


class TestStratifiedSample:
    def test_each_shell_respects_cap(self) -> None:
        # 5 sats per shell across all three; we cap at 2 → 6 sats kept total.
        rows = []
        for shell_idx, line2 in enumerate((LINE2_540KM, LINE2_550KM, LINE2_560KM)):
            for sat in range(5):
                norad = shell_idx * 100 + sat
                rows.append(_make_tle_row(norad, "2026-04-01T00:00:00Z", line2))
                rows.append(_make_tle_row(norad, "2026-04-02T00:00:00Z", line2))
        tles = pd.DataFrame(rows)
        pairs = build_pairs(tles)
        sampled = stratified_sample(pairs, n_per_shell=2, seed=42)
        assert sampled["norad_id"].nunique() == 6  # 2 per shell × 3 shells

    def test_deterministic_under_seed(self) -> None:
        rows = []
        for sat in range(20):
            rows.append(_make_tle_row(sat, "2026-04-01T00:00:00Z", LINE2_550KM))
            rows.append(_make_tle_row(sat, "2026-04-02T00:00:00Z", LINE2_550KM))
        pairs = build_pairs(pd.DataFrame(rows))
        a = stratified_sample(pairs, n_per_shell=5, seed=42)
        b = stratified_sample(pairs, n_per_shell=5, seed=42)
        assert sorted(a["norad_id"].unique()) == sorted(b["norad_id"].unique())

    def test_sats_outside_shells_are_excluded(self) -> None:
        # Mean motion 14.0 rev/day → too low, not in any defined shell.
        line2_too_low = "2 44713  53.0540 116.6831 0001247  82.6886 277.4255 14.00000000250789"
        rows = [
            _make_tle_row(1, "2026-04-01T00:00:00Z", line2_too_low),
            _make_tle_row(1, "2026-04-02T00:00:00Z", line2_too_low),
            _make_tle_row(2, "2026-04-01T00:00:00Z", LINE2_550KM),
            _make_tle_row(2, "2026-04-02T00:00:00Z", LINE2_550KM),
        ]
        pairs = build_pairs(pd.DataFrame(rows))
        sampled = stratified_sample(pairs, n_per_shell=10, seed=42)
        assert set(sampled["norad_id"].unique()) == {2}


class TestBuildCorpus:
    def test_end_to_end_respects_threshold_and_cap(self) -> None:
        # An out-of-shell mean motion (14.0 rev/day → ~700 km altitude) so any
        # sat ending here gives an unambiguous SMA jump from any quiet shell.
        line2_offshell = "2 44713  53.0540 116.6831 0001247  82.6886 277.4255 14.00000000250789"

        rows = []
        norad_counter = 1
        for shell_line2 in (LINE2_540KM, LINE2_550KM, LINE2_560KM):
            for _ in range(3):
                rows.append(_make_tle_row(norad_counter, "2026-04-01T00:00:00Z", shell_line2))
                rows.append(_make_tle_row(norad_counter, "2026-04-02T00:00:00Z", shell_line2))
                norad_counter += 1
            # One maneuvering sat per shell: ends up well outside the shell.
            rows.append(_make_tle_row(norad_counter, "2026-04-01T00:00:00Z", shell_line2))
            rows.append(_make_tle_row(norad_counter, "2026-04-02T00:00:00Z", line2_offshell))
            norad_counter += 1

        corpus = build_corpus(pd.DataFrame(rows), n_per_shell=10)
        # 3 quiet sats per shell × 3 shells = 9 sats survive; maneuvering sats drop.
        assert corpus["norad_id"].nunique() == 9
        assert set(corpus["alt_shell"].unique()) == {"540", "550", "560"}


class TestDefaults:
    def test_default_threshold_matches_documented_value(self) -> None:
        # If this changes, fig_maneuver_filter.py's annotation must change too.
        assert DEFAULT_MANEUVER_THRESHOLD_KM == 0.1
