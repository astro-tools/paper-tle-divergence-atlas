"""Unit tests for sweep.spacecraft_props.

No network — `fetch_gcat_satcat` is exercised only by the manual
`make fetch-satcat` step. Parsing and the per-corpus attach are
verified against a synthetic GCAT-format tsv built per-test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sweep.spacecraft_props import (
    AREA_LARGE_M2,
    AREA_SMALL_M2,
    SPAN_BUCKET_BOUNDARY_M,
    attach_spacecraft_props,
    drag_area_from_span,
    parse_gcat_satcat,
)

# A small slice of GCAT satcat.tsv in the real on-disk format — first line is
# the tab-separated header prefixed with "#", subsequent "#" lines are
# comments, then a few data rows covering v1.0 (Span 9), v1.5 (Span 9),
# v2-mini (Span 29), plus one row with a blank Span column so the parser's
# NaN handling is exercised. The full GCAT header has dozens of columns; we
# include only the subset our parser actually reads.
GCAT_TSV_TEMPLATE = (
    "#JCAT\tSatcat\tLaunch_Tag\tPiece\tType\tName\tPLName\tLDate\tDryMass\tDryFlag\tSpan\tSpanFlag\n"
    "# Updated 2026 May 11 0953:03\n"
    "S44235\t44235\t2019-029\t2019-029A\tP R\tStarlink 31\tStarlink V0.9-01\t2019 May 24\t   248 \t?\t   9.0 \t?\n"
    "S47789\t47789\t2021-018\t2021-018C\tP R\tStarlink 1909\tStarlink V1.0-L20-03\t2021 Mar 04\t   248 \t?\t   9.0 \t?\n"
    "S55444\t55444\t2023-006\t2023-006A\tP R\tStarlink 5917\tStarlink Group 5-2\t2023 Jan 19\t   305 \t?\t   9.0 \t?\n"
    "S60000\t60000\t2025-100\t2025-100A\tP R\tStarlink 30000\tStarlink Group 15-1\t2025 Jul 01\t   700 \t?\t  29.0 \t?\n"
    "S70000\t70000\t2026-001\t2026-001A\tP R\tFuture\tStarlink Group 99-1\t2026 Jan 01\t       \t?\t       \t?\n"
)


class TestDragAreaFromSpan:
    def test_small_span_uses_small_area(self) -> None:
        assert drag_area_from_span(9.0) == AREA_SMALL_M2

    def test_large_span_uses_large_area(self) -> None:
        assert drag_area_from_span(29.0) == AREA_LARGE_M2

    def test_boundary_is_inclusive_low(self) -> None:
        # 15 m exactly belongs to the small-span (v1.x) bucket.
        assert drag_area_from_span(SPAN_BUCKET_BOUNDARY_M) == AREA_SMALL_M2

    def test_just_above_boundary_uses_large_area(self) -> None:
        assert drag_area_from_span(SPAN_BUCKET_BOUNDARY_M + 0.01) == AREA_LARGE_M2


class TestParseGcatSatcat:
    def _write(self, tmp_path: Path, body: str = GCAT_TSV_TEMPLATE) -> Path:
        path = tmp_path / "satcat.tsv"
        path.write_text(body, encoding="utf-8")
        return path

    def test_returns_expected_columns_and_dtypes(self, tmp_path: Path) -> None:
        df = parse_gcat_satcat(self._write(tmp_path))
        assert list(df.columns) == ["satcat_id", "dry_mass_kg", "span_m", "pl_name"]
        assert df["satcat_id"].dtype == int

    def test_drops_comment_lines_and_keeps_data_rows(self, tmp_path: Path) -> None:
        df = parse_gcat_satcat(self._write(tmp_path))
        # Five data rows in the fixture; one has blank mass/span (rows still kept).
        assert len(df) == 5
        assert sorted(df["satcat_id"].tolist()) == [44235, 47789, 55444, 60000, 70000]

    def test_numeric_columns_parsed_through_whitespace_padding(self, tmp_path: Path) -> None:
        df = parse_gcat_satcat(self._write(tmp_path)).set_index("satcat_id")
        assert df.loc[47789, "dry_mass_kg"] == pytest.approx(248.0)
        assert df.loc[55444, "span_m"] == pytest.approx(9.0)
        assert df.loc[60000, "dry_mass_kg"] == pytest.approx(700.0)

    def test_blank_numeric_fields_become_nan(self, tmp_path: Path) -> None:
        df = parse_gcat_satcat(self._write(tmp_path)).set_index("satcat_id")
        assert pd.isna(df.loc[70000, "dry_mass_kg"])
        assert pd.isna(df.loc[70000, "span_m"])


class TestAttachSpacecraftProps:
    def _satcat(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"satcat_id": 1, "dry_mass_kg": 248.0, "span_m": 9.0, "pl_name": "Starlink V1.0"},
                {
                    "satcat_id": 2,
                    "dry_mass_kg": 305.0,
                    "span_m": 9.0,
                    "pl_name": "Starlink Group 4-1",
                },
                {
                    "satcat_id": 3,
                    "dry_mass_kg": 700.0,
                    "span_m": 29.0,
                    "pl_name": "Starlink Group 15-1",
                },
            ],
        )

    def _corpus(self, norad_ids: list[int]) -> pd.DataFrame:
        # Two pair rows per sat so the many-to-one merge is exercised.
        rows = []
        for nid in norad_ids:
            for dt in (86_400, 604_800):
                rows.append({"norad_id": nid, "target_dt_sec": dt, "alt_shell": "550"})
        return pd.DataFrame(rows)

    def test_joins_per_sat_props_to_every_pair(self) -> None:
        corpus = self._corpus([1, 2, 3])
        out = attach_spacecraft_props(corpus, self._satcat())
        assert len(out) == len(corpus)  # no row explosion
        # v1.0 sat (NORAD 1): mass 248, span 9 → drag area 5.0.
        v10 = out[out["norad_id"] == 1].iloc[0]
        assert v10["dry_mass_kg"] == 248.0
        assert v10["drag_area_m2"] == AREA_SMALL_M2
        assert v10["srp_area_m2"] == AREA_SMALL_M2
        # v2-mini sat (NORAD 3): mass 700, span 29 → drag area 3.5.
        v2 = out[out["norad_id"] == 3].iloc[0]
        assert v2["dry_mass_kg"] == 700.0
        assert v2["drag_area_m2"] == AREA_LARGE_M2

    def test_preserves_existing_corpus_columns(self) -> None:
        corpus = self._corpus([1])
        out = attach_spacecraft_props(corpus, self._satcat())
        for col in corpus.columns:
            assert col in out.columns

    def test_pl_name_passes_through(self) -> None:
        out = attach_spacecraft_props(self._corpus([2]), self._satcat())
        assert (out["gcat_pl_name"] == "Starlink Group 4-1").all()

    def test_raises_when_corpus_sat_missing_from_satcat(self) -> None:
        corpus = self._corpus([1, 99])
        with pytest.raises(KeyError, match="missing 1 NORAD IDs"):
            attach_spacecraft_props(corpus, self._satcat())

    def test_raises_on_nan_mass_or_span(self) -> None:
        satcat = self._satcat()
        satcat.loc[satcat["satcat_id"] == 2, "dry_mass_kg"] = float("nan")
        with pytest.raises(ValueError, match="missing mass/span"):
            attach_spacecraft_props(self._corpus([2]), satcat)

    def test_requires_norad_id_column(self) -> None:
        with pytest.raises(ValueError, match="missing 'norad_id'"):
            attach_spacecraft_props(pd.DataFrame({"other": [1]}), self._satcat())
