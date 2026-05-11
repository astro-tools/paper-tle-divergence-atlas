"""Unit tests for sweep.tle_pipeline.

Pure-Python only; no network, no SGP4 propagation. Synthetic TLE lines are
crafted to exercise the mean-motion column parser, the SMA→shell mapper,
maneuver detection, and the multi-Δt pair construction.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from sweep.tle_pipeline import (
    DEFAULT_MANEUVER_THRESHOLD_KM,
    DEFAULT_TARGET_DTS_SEC,
    EARTH_RADIUS_KM,
    altitude_shell,
    build_corpus,
    build_pairs,
    detect_maneuver_epochs,
    filter_maneuvers,
    sample_sats,
    sma_km_from_mean_motion,
    subsample_starting_tles,
)

# A real Starlink line 2 (STARLINK-1007); the parser only inspects columns
# 53..63 (0-indexed 52:63), where mean motion in rev/day is encoded.
LINE2_550KM = "2 44713  53.0540 116.6831 0001247  82.6886 277.4255 15.06405853250789"
LINE2_540KM = "2 44713  53.0540 116.6831 0001247  82.6886 277.4255 15.10000000250789"
LINE2_560KM = "2 44713  53.0540 116.6831 0001247  82.6886 277.4255 15.02000000250789"
LINE2_OFFSHELL = "2 44713  53.0540 116.6831 0001247  82.6886 277.4255 14.00000000250789"


def _tle(norad_id: int, epoch: str, line2: str = LINE2_550KM, line1: str = "1") -> dict:
    return {
        "norad_id": norad_id,
        "epoch": pd.Timestamp(epoch, tz="UTC"),
        "line1": line1,
        "line2": line2,
    }


def _daily_tles(
    norad_id: int, days: int, line2: str = LINE2_550KM, start: str = "2026-04-01T00:00:00Z"
) -> list[dict]:
    """`days` TLEs, one per UTC day starting at `start`."""
    base = pd.Timestamp(start, tz="UTC")
    return [_tle(norad_id, (base + timedelta(days=d)).isoformat(), line2) for d in range(days)]


class TestSmaFromMeanMotion:
    def test_known_starlink_mean_motion_lands_near_550_km_altitude(self) -> None:
        sma = sma_km_from_mean_motion(15.06405853)
        altitude = sma - EARTH_RADIUS_KM
        assert 545.0 <= altitude <= 555.0, f"expected 550-km shell, got altitude {altitude:.2f} km"

    def test_higher_mean_motion_means_lower_altitude(self) -> None:
        assert sma_km_from_mean_motion(15.10) < sma_km_from_mean_motion(15.06)


class TestAltitudeShell:
    def test_550_band_assignment(self) -> None:
        assert altitude_shell(EARTH_RADIUS_KM + 550.0) == "550"

    def test_below_lowest_shell_is_unassigned(self) -> None:
        assert altitude_shell(EARTH_RADIUS_KM + 400.0) is None

    def test_above_highest_shell_is_unassigned(self) -> None:
        assert altitude_shell(EARTH_RADIUS_KM + 700.0) is None

    def test_shells_are_disjoint(self) -> None:
        # 547.0 km sits on the half-open 540/550 boundary; the lower band is
        # [low, high), so 547.0 falls into "550".
        assert altitude_shell(EARTH_RADIUS_KM + 547.0) == "550"


class TestDetectManeuverEpochs:
    def test_quiet_sat_yields_no_events(self) -> None:
        tles = pd.DataFrame(_daily_tles(1, days=5, line2=LINE2_550KM))
        events = detect_maneuver_epochs(tles)
        assert events.empty

    def test_sma_jump_is_flagged_at_t_j(self) -> None:
        # Day 0..2 at 550 km, day 3 jumps to 540 km (~10 km jump).
        rows = _daily_tles(1, days=3, line2=LINE2_550KM)
        rows.append(_tle(1, "2026-04-04T00:00:00Z", LINE2_540KM))
        tles = pd.DataFrame(rows)
        events = detect_maneuver_epochs(tles)
        assert len(events) == 1
        assert events.iloc[0]["maneuver_epoch"] == pd.Timestamp("2026-04-04T00:00:00Z")

    def test_below_threshold_is_not_flagged(self) -> None:
        # Identical line2s → zero SMA jump.
        tles = pd.DataFrame(_daily_tles(1, days=4, line2=LINE2_550KM))
        events = detect_maneuver_epochs(tles, sma_jump_threshold_km=0.001)
        assert events.empty


class TestSubsampleStartingTles:
    def test_one_per_day_default(self) -> None:
        # Six TLEs in one day → keep the first only.
        base = pd.Timestamp("2026-04-01T00:00:00Z")
        rows = [_tle(1, (base + timedelta(hours=4 * i)).isoformat()) for i in range(6)]
        tles = pd.DataFrame(rows)
        out = subsample_starting_tles(tles)
        assert len(out) == 1
        assert out.iloc[0]["epoch"] == base

    def test_multi_day_keeps_one_per_day_per_sat(self) -> None:
        rows = []
        for sat in (1, 2):
            for d in range(3):
                for h in (0, 6, 12):
                    rows.append(
                        _tle(
                            sat,
                            (
                                pd.Timestamp("2026-04-01T00:00:00Z") + timedelta(days=d, hours=h)
                            ).isoformat(),
                        )
                    )
        tles = pd.DataFrame(rows)
        out = subsample_starting_tles(tles)
        assert len(out) == 6  # 2 sats × 3 days
        for sat in (1, 2):
            assert (out["norad_id"] == sat).sum() == 3

    def test_per_day_argument_respected(self) -> None:
        base = pd.Timestamp("2026-04-01T00:00:00Z")
        rows = [_tle(1, (base + timedelta(hours=4 * i)).isoformat()) for i in range(6)]
        out = subsample_starting_tles(pd.DataFrame(rows), per_day=3)
        assert len(out) == 3


class TestBuildPairs:
    def test_target_dt_match_within_tolerance(self) -> None:
        # Starting TLE at day 0; one candidate per default target Δt (6h, 1d, 3d, 7d).
        base = pd.Timestamp("2026-04-01T00:00:00Z")
        offsets = (
            timedelta(0),
            timedelta(hours=6),
            timedelta(days=1),
            timedelta(days=3),
            timedelta(days=7),
        )
        rows = [_tle(1, (base + off).isoformat()) for off in offsets]
        tles = pd.DataFrame(rows)
        starts = tles.iloc[:1]
        pairs = build_pairs(starts, tles)
        assert len(pairs) == len(DEFAULT_TARGET_DTS_SEC)
        assert sorted(pairs["target_dt_sec"].tolist()) == sorted(DEFAULT_TARGET_DTS_SEC)

    def test_missing_target_skipped_when_out_of_tolerance(self) -> None:
        # Only a +1d candidate exists; +3d and +7d should yield no pairs.
        rows = [
            _tle(1, "2026-04-01T00:00:00Z"),
            _tle(1, "2026-04-02T00:00:00Z"),
        ]
        tles = pd.DataFrame(rows)
        pairs = build_pairs(tles.iloc[:1], tles)
        assert len(pairs) == 1
        assert pairs.iloc[0]["target_dt_sec"] == 86_400

    def test_nearest_candidate_within_tolerance_is_chosen(self) -> None:
        # Two candidates near +1d: 23h vs. 25h — pick the closer one.
        rows = [
            _tle(1, "2026-04-01T00:00:00Z"),
            _tle(1, "2026-04-01T23:00:00Z"),
            _tle(1, "2026-04-02T01:00:00Z"),
            # No +3d or +7d candidates.
        ]
        tles = pd.DataFrame(rows)
        pairs = build_pairs(tles.iloc[:1], tles)
        assert len(pairs) == 1
        assert pairs.iloc[0]["epoch_j"] == pd.Timestamp("2026-04-01T23:00:00Z")

    def test_sat_grouping_is_respected(self) -> None:
        # Two sats, each with a +1d candidate; pairs should not cross sats.
        rows = [
            _tle(1, "2026-04-01T00:00:00Z"),
            _tle(1, "2026-04-02T00:00:00Z"),
            _tle(2, "2026-04-01T00:00:00Z"),
            _tle(2, "2026-04-02T00:00:00Z"),
        ]
        tles = pd.DataFrame(rows)
        starts = tles[tles["epoch"] == pd.Timestamp("2026-04-01T00:00:00Z")]
        pairs = build_pairs(starts, tles)
        assert set(pairs["norad_id"]) == {1, 2}


class TestFilterManeuvers:
    def test_no_maneuvers_keeps_all_pairs(self) -> None:
        rows = _daily_tles(1, days=8, line2=LINE2_550KM)
        tles = pd.DataFrame(rows)
        pairs = build_pairs(tles.iloc[:1], tles)
        maneuvers = detect_maneuver_epochs(tles)
        kept = filter_maneuvers(pairs, maneuvers)
        assert len(kept) == len(pairs) > 0

    def test_maneuver_inside_pair_drops_it(self) -> None:
        # Day 0..2 at 550, day 3 jump to 540, day 4..7 at 540.
        rows = _daily_tles(1, days=3, line2=LINE2_550KM)
        rows.append(_tle(1, "2026-04-04T00:00:00Z", LINE2_540KM))
        rows.extend(_daily_tles(1, days=4, line2=LINE2_540KM, start="2026-04-05T00:00:00Z"))
        tles = pd.DataFrame(rows)
        starts = tles.iloc[:1]  # 2026-04-01
        pairs = build_pairs(starts, tles)
        # All three target Δts (+1d, +3d, +7d) might match candidates, but the
        # +3d (2026-04-04) and +7d (2026-04-08) pairs cover the maneuver epoch.
        maneuvers = detect_maneuver_epochs(tles)
        kept = filter_maneuvers(pairs, maneuvers)
        # Only the +1d pair (2026-04-02) survives — no maneuver in (04-01, 04-02].
        assert len(kept) == 1
        assert kept.iloc[0]["target_dt_sec"] == 86_400

    def test_empty_pairs_passes_through(self) -> None:
        empty = pd.DataFrame(columns=["norad_id", "epoch_i", "epoch_j", "sma_i_km"])
        assert filter_maneuvers(empty, pd.DataFrame()).empty


class TestSampleSats:
    def _three_shell_fleet(self, sats_per_shell: int = 5) -> pd.DataFrame:
        rows = []
        norad = 1
        for shell_line2 in (LINE2_540KM, LINE2_550KM, LINE2_560KM):
            for _ in range(sats_per_shell):
                rows.extend(_daily_tles(norad, days=3, line2=shell_line2))
                norad += 1
        return pd.DataFrame(rows)

    def test_each_shell_capped(self) -> None:
        sat_to_shell = sample_sats(self._three_shell_fleet(), n_per_shell=2, seed=42)
        assert len(sat_to_shell) == 6  # 2 per shell × 3 shells
        assert sorted(set(sat_to_shell.values())) == ["540", "550", "560"]

    def test_deterministic_under_seed(self) -> None:
        tles = self._three_shell_fleet()
        a = sample_sats(tles, n_per_shell=2, seed=42)
        b = sample_sats(tles, n_per_shell=2, seed=42)
        assert sorted(a) == sorted(b)

    def test_offshell_sats_are_excluded(self) -> None:
        rows = _daily_tles(1, days=3, line2=LINE2_OFFSHELL)
        rows.extend(_daily_tles(2, days=3, line2=LINE2_550KM))
        sat_to_shell = sample_sats(pd.DataFrame(rows), n_per_shell=10, seed=42)
        assert set(sat_to_shell) == {2}


class TestBuildCorpus:
    def test_end_to_end_quiet_corpus(self) -> None:
        rows = []
        norad = 1
        for shell_line2 in (LINE2_540KM, LINE2_550KM, LINE2_560KM):
            for _ in range(3):
                rows.extend(_daily_tles(norad, days=10, line2=shell_line2))
                norad += 1
        corpus = build_corpus(pd.DataFrame(rows), n_per_shell=10)
        assert corpus["norad_id"].nunique() == 9  # 3 per shell × 3 shells
        assert set(corpus["alt_shell"].unique()) == {"540", "550", "560"}
        # Each sat: 9 starting epochs × 3 target Δts → 27 max, but +3d and +7d
        # are truncated at the window end. Just assert "has pairs".
        assert len(corpus) > 0

    def test_maneuvering_sat_pairs_are_filtered(self) -> None:
        # One quiet sat + one sat that maneuvers mid-window.
        rows = _daily_tles(1, days=10, line2=LINE2_550KM)
        # Sat 2: 550 km for 4 days, then jump to 540 for the rest.
        rows.extend(_daily_tles(2, days=4, line2=LINE2_550KM))
        rows.extend(_daily_tles(2, days=6, line2=LINE2_540KM, start="2026-04-05T00:00:00Z"))
        corpus = build_corpus(pd.DataFrame(rows), n_per_shell=10)
        # Both sats appear in the corpus, but sat 2's pairs that bridge the
        # maneuver (day 1-2 → day 5, day 1-4 → day 8) are dropped. Sat 2 has
        # a non-empty set of pairs only for starting epochs and Δts where the
        # interval does not include the maneuver event.
        assert 1 in corpus["norad_id"].to_numpy()
        sat2_pairs = corpus[corpus["norad_id"] == 2]
        # Any sat 2 pair must NOT span the maneuver at 2026-04-05.
        if not sat2_pairs.empty:
            for _, p in sat2_pairs.iterrows():
                assert not (p["epoch_i"] < pd.Timestamp("2026-04-05T00:00:00Z") <= p["epoch_j"]), (
                    f"sat-2 pair {p['epoch_i']} → {p['epoch_j']} spans the maneuver"
                )


class TestDefaults:
    def test_maneuver_threshold_default(self) -> None:
        # If this changes, fig_maneuver_filter.py's annotation must too.
        assert DEFAULT_MANEUVER_THRESHOLD_KM == 0.1

    def test_target_dts_default(self) -> None:
        # If this changes, the paper's H1 staleness fit horizons change too.
        assert DEFAULT_TARGET_DTS_SEC == (6 * 3600, 86_400, 3 * 86_400, 7 * 86_400)
