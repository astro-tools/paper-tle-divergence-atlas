"""Unit tests for sweep.space_weather.

Covers the text parser, the cache lookup, and the live cache committed
at `src/data/sw_cache.parquet` (so a corpus-window extension that
silently outruns the committed window fails here, not deep in a sweep).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from sweep.space_weather import SwRow, load_sw_cache, lookup_for_epoch, parse_sw_text

# Minimal fixture mirroring CelesTrak's `sw19571001.txt` v1.2 layout: a
# couple of OBSERVED rows, then a couple of DAILY_PREDICTED rows. Values
# match the real file as of the commit so the assertions double as a
# parser-position check against the published FORMAT string.
_FIXTURE = """\
DATATYPE CssiSpaceWeather
VERSION 1.2
UPDATED 2026 May 12 00:00:00 UTC
#
# yy mm dd BSRN ND Kp Kp Kp Kp Kp Kp Kp Kp Sum Ap  Ap  Ap  Ap  Ap  Ap  Ap  Ap  Avg Cp C9 ISN F10.7 Q Ctr81 Lst81 F10.7 Ctr81 Lst81
#
NUM_OBSERVED_POINTS 2
BEGIN OBSERVED
2026 04 01 2627  9 30 20 10 10 30 23 13 23 160  15   7   4   4  15   9   5   9   8 0.5 2 129 141.7 0 125.6 135.9 141.9 125.8 138.8
2026 04 15 2627 23  7 10 10  3 10 20 10 13  83   3   4   4   2   4   7   4   5   4 0.1 0  49 105.5 0 125.3 128.3 104.8 124.5 130.3
END OBSERVED
NUM_DAILY_PREDICTED_POINTS 1
BEGIN DAILY_PREDICTED
2026 05 12 2627 40 17 13 10 17 10 17 10 20 113   6   5   4   6   4   6   4   7   5 0.2 1 100 120.0 2 121.0 122.0 120.0 121.5 122.5
END DAILY_PREDICTED
"""


class TestParseSwText:
    def test_picks_up_both_sections(self) -> None:
        df = parse_sw_text(_FIXTURE)
        assert len(df) == 3
        assert set(df.columns) == {"date", "f107_obs", "f107_avg81", "ap_daily", "is_observed"}

    def test_observed_flag(self) -> None:
        df = parse_sw_text(_FIXTURE)
        assert df["is_observed"].tolist() == [True, True, False]

    def test_known_value_2026_04_01(self) -> None:
        df = parse_sw_text(_FIXTURE)
        row = df[df["date"] == dt.date(2026, 4, 1)].iloc[0]
        # Obs F10.7 = 141.9, Obs Ctr81 = 125.8, Ap Avg = 8 in the fixture row.
        assert row["f107_obs"] == pytest.approx(141.9)
        assert row["f107_avg81"] == pytest.approx(125.8)
        assert row["ap_daily"] == pytest.approx(8.0)

    def test_skips_blank_and_comment_lines(self) -> None:
        text = "BEGIN OBSERVED\n\n# comment\nEND OBSERVED\n"
        assert parse_sw_text(text).empty

    def test_sorted_by_date(self) -> None:
        df = parse_sw_text(_FIXTURE)
        assert df["date"].is_monotonic_increasing


class TestLookupForEpoch:
    def _lookup(self) -> dict[dt.date, SwRow]:
        return {
            dt.date(2026, 4, 1): SwRow(
                f107_obs=141.9, f107_avg81=125.8, ap_daily=8.0, is_observed=True
            ),
        }

    def test_hit_uses_utc_date(self) -> None:
        # Early-UTC and late-UTC epochs on the same UTC day should resolve
        # to the same SW row; a naive local-tz date() would split them.
        early = pd.Timestamp("2026-04-01T00:01:16Z")
        late = pd.Timestamp("2026-04-01T23:59:00Z")
        assert lookup_for_epoch(self._lookup(), early).ap_daily == 8.0
        assert lookup_for_epoch(self._lookup(), late).ap_daily == 8.0

    def test_miss_raises_keyerror(self) -> None:
        # Per the issue: missing date raises a clear error rather than
        # silently NaN-ing — so a corpus window extension that outruns
        # the cache surfaces immediately.
        with pytest.raises(KeyError, match="no space-weather entry for 2027-01-15"):
            lookup_for_epoch(self._lookup(), pd.Timestamp("2027-01-15T12:00:00Z"))


class TestLoadSwCache:
    """End-to-end check against the committed src/data/sw_cache.parquet."""

    CACHE_PATH = Path(__file__).resolve().parents[1] / "src" / "data" / "sw_cache.parquet"

    def test_cache_exists_and_loads(self) -> None:
        assert self.CACHE_PATH.exists(), f"{self.CACHE_PATH} missing — run `make fetch-sw`"
        lookup = load_sw_cache(self.CACHE_PATH)
        assert len(lookup) > 0

    def test_covers_corpus_window(self) -> None:
        # src/data/window.json: April 2026. Cache must cover at least
        # the corpus window, with margin for window extensions.
        lookup = load_sw_cache(self.CACHE_PATH)
        for date in (dt.date(2026, 4, 1), dt.date(2026, 4, 15), dt.date(2026, 4, 30)):
            assert date in lookup, f"{date} missing from sw_cache.parquet"

    def test_known_april_2026_values(self) -> None:
        # Sanity check against the live CelesTrak file for 2026-04-01:
        # Obs F10.7 = 141.9, Obs Ctr81 = 125.8, Ap Avg = 8.
        # If CelesTrak rebases these historically, this test will flag it.
        lookup = load_sw_cache(self.CACHE_PATH)
        row = lookup[dt.date(2026, 4, 1)]
        assert row.f107_obs == pytest.approx(141.9)
        assert row.f107_avg81 == pytest.approx(125.8)
        assert row.ap_daily == pytest.approx(8.0)
        assert row.is_observed is True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
