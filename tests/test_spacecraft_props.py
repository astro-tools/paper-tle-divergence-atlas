"""Unit tests for sweep.spacecraft_props.

No network — `fetch_gcat_satcat` is exercised only by the manual
`make fetch-satcat` step. Parsing, generation classification, and the
per-corpus attach are verified against a synthetic GCAT-format tsv
built per-test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sweep.spacecraft_props import (
    DRAG_AREA_M2,
    SRP_AREA_M2,
    Generation,
    attach_spacecraft_props,
    generation_from_pl_name,
    parse_gcat_satcat,
)

# A small slice of GCAT satcat.tsv in the real on-disk format — first line is
# the tab-separated header prefixed with "#", subsequent "#" lines are
# comments, then a few data rows covering each generation cohort plus one
# row with a blank Span / DryMass column so the parser's NaN handling is
# exercised. The full GCAT header has dozens of columns; we include only
# the subset our parser actually reads.
GCAT_TSV_TEMPLATE = (
    "#JCAT\tSatcat\tLaunch_Tag\tPiece\tType\tName\tPLName\tLDate\tDryMass\tDryFlag\tSpan\tSpanFlag\n"
    "# Updated 2026 May 11 0953:03\n"
    "S44235\t44235\t2019-029\t2019-029A\tP R\tStarlink 31\tStarlink V0.9-01\t2019 May 24\t   248 \t?\t   9.0 \t?\n"
    "S47789\t47789\t2021-018\t2021-018C\tP R\tStarlink 1909\tStarlink V1.0-L20-03\t2021 Mar 04\t   248 \t?\t   9.0 \t?\n"
    "S55444\t55444\t2023-006\t2023-006A\tP R\tStarlink 5917\tStarlink Group 5-2\t2023 Jan 19\t   305 \t?\t   9.0 \t?\n"
    "S60000\t60000\t2025-100\t2025-100A\tP R\tStarlink 30000\tStarlink Group 15-1\t2025 Jul 01\t   700 \t?\t  29.0 \t?\n"
    "S70000\t70000\t2026-001\t2026-001A\tP R\tFuture\tStarlink Group 99-1\t2026 Jan 01\t       \t?\t       \t?\n"
)


class TestGenerationFromPlName:
    def test_v0_9_maps_to_v1_0_bucket(self) -> None:
        assert generation_from_pl_name("Starlink V0.9-01") == Generation.V1_0

    def test_v1_0_launch_pattern(self) -> None:
        assert generation_from_pl_name("Starlink V1.0-L20-03") == Generation.V1_0

    def test_tsp_variant_is_v1_0_bucket(self) -> None:
        assert generation_from_pl_name("Starlink TSP2-03") == Generation.V1_0

    def test_group_2_5_are_v1_5(self) -> None:
        assert generation_from_pl_name("Starlink Group 2-1-15") == Generation.V1_5
        assert generation_from_pl_name("Starlink Group 3-2-7") == Generation.V1_5
        assert generation_from_pl_name("Starlink Group 4-1-3") == Generation.V1_5
        assert generation_from_pl_name("Starlink Group 5-2") == Generation.V1_5

    def test_group_6_plus_are_v2_mini(self) -> None:
        assert generation_from_pl_name("Starlink Group 6-13") == Generation.V2_MINI
        assert generation_from_pl_name("Starlink Group 13-4") == Generation.V2_MINI
        assert generation_from_pl_name("Starlink Group 15-1-22") == Generation.V2_MINI
        assert generation_from_pl_name("Starlink Group 17-6") == Generation.V2_MINI

    def test_handles_whitespace(self) -> None:
        assert generation_from_pl_name("  Starlink   Group 4-1-3  ") == Generation.V1_5

    def test_raises_on_non_starlink(self) -> None:
        with pytest.raises(ValueError, match="unrecognized"):
            generation_from_pl_name("OneWeb 0001")

    def test_raises_on_unparseable_starlink(self) -> None:
        with pytest.raises(ValueError, match="unrecognized"):
            generation_from_pl_name("Starlink Mystery-X")


class TestPerGenerationTables:
    def test_drag_area_covers_every_generation(self) -> None:
        for gen in Generation:
            assert gen in DRAG_AREA_M2
            assert DRAG_AREA_M2[gen] > 0

    def test_srp_area_covers_every_generation(self) -> None:
        for gen in Generation:
            assert gen in SRP_AREA_M2
            assert SRP_AREA_M2[gen] > 0

    def test_v2_mini_drag_area_larger_than_v1_x(self) -> None:
        # Physical sanity: v2-mini bus is structurally bigger than v1.0/v1.5.
        assert DRAG_AREA_M2[Generation.V2_MINI] > DRAG_AREA_M2[Generation.V1_5]
        assert DRAG_AREA_M2[Generation.V2_MINI] > DRAG_AREA_M2[Generation.V1_0]

    def test_srp_area_at_least_drag_area(self) -> None:
        # SRP integrates over panel surface; drag is the much-smaller ram-direction
        # cross-section in nominal attitude. Drag area should never exceed SRP area.
        for gen in Generation:
            assert SRP_AREA_M2[gen] >= DRAG_AREA_M2[gen]


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
                {
                    "satcat_id": 1,
                    "dry_mass_kg": 248.0,
                    "span_m": 9.0,
                    "pl_name": "Starlink V1.0-L20-03",
                },
                {
                    "satcat_id": 2,
                    "dry_mass_kg": 290.0,
                    "span_m": 9.0,
                    "pl_name": "Starlink Group 4-1-3",
                },
                {
                    "satcat_id": 3,
                    "dry_mass_kg": 530.0,
                    "span_m": 29.0,
                    "pl_name": "Starlink Group 15-1-22",
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

    def test_dry_mass_per_sat_drag_and_srp_per_generation(self) -> None:
        corpus = self._corpus([1, 2, 3])
        out = attach_spacecraft_props(corpus, self._satcat())
        assert len(out) == len(corpus)  # many-to-one merge, no row explosion

        v1_0 = out[out["norad_id"] == 1].iloc[0]
        assert v1_0["dry_mass_kg"] == 248.0
        assert v1_0["generation"] == Generation.V1_0.value
        assert v1_0["drag_area_m2"] == DRAG_AREA_M2[Generation.V1_0]
        assert v1_0["srp_area_m2"] == SRP_AREA_M2[Generation.V1_0]

        v1_5 = out[out["norad_id"] == 2].iloc[0]
        assert v1_5["dry_mass_kg"] == 290.0
        assert v1_5["generation"] == Generation.V1_5.value
        assert v1_5["drag_area_m2"] == DRAG_AREA_M2[Generation.V1_5]

        v2_mini = out[out["norad_id"] == 3].iloc[0]
        assert v2_mini["dry_mass_kg"] == 530.0
        assert v2_mini["generation"] == Generation.V2_MINI.value
        assert v2_mini["drag_area_m2"] == DRAG_AREA_M2[Generation.V2_MINI]
        assert v2_mini["srp_area_m2"] == SRP_AREA_M2[Generation.V2_MINI]

    def test_preserves_existing_corpus_columns(self) -> None:
        corpus = self._corpus([1])
        out = attach_spacecraft_props(corpus, self._satcat())
        for col in corpus.columns:
            assert col in out.columns

    def test_pl_name_passes_through(self) -> None:
        out = attach_spacecraft_props(self._corpus([2]), self._satcat())
        assert (out["gcat_pl_name"] == "Starlink Group 4-1-3").all()

    def test_raises_when_corpus_sat_missing_from_satcat(self) -> None:
        corpus = self._corpus([1, 99])
        with pytest.raises(KeyError, match="missing 1 NORAD IDs"):
            attach_spacecraft_props(corpus, self._satcat())

    def test_raises_on_nan_dry_mass(self) -> None:
        satcat = self._satcat()
        satcat.loc[satcat["satcat_id"] == 2, "dry_mass_kg"] = float("nan")
        with pytest.raises(ValueError, match="missing dry mass"):
            attach_spacecraft_props(self._corpus([2]), satcat)

    def test_raises_on_unclassifiable_pl_name(self) -> None:
        satcat = self._satcat()
        satcat.loc[satcat["satcat_id"] == 2, "pl_name"] = "Starlink Mystery"
        with pytest.raises(ValueError, match="unrecognized"):
            attach_spacecraft_props(self._corpus([2]), satcat)

    def test_requires_norad_id_column(self) -> None:
        with pytest.raises(ValueError, match="missing 'norad_id'"):
            attach_spacecraft_props(pd.DataFrame({"other": [1]}), self._satcat())
