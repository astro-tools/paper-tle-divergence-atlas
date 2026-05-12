"""Unit tests for sweep.aggregate."""

from __future__ import annotations

import pandas as pd
import pytest

from sweep.aggregate import CARRIED_COLUMNS, JOIN_KEYS, aggregate


def _per_run_row(run_id: int, norad_id: int, target_dt_sec: int, t_i: pd.Timestamp) -> dict:
    """A row matching `_postprocess_run`'s output schema (17 cols)."""
    return {
        "run_id": run_id,
        "norad_id": norad_id,
        "target_dt_sec": target_dt_sec,
        "t_i": t_i,
        "t_j": t_i + pd.Timedelta(seconds=target_dt_sec),
        "actual_dt_sec": float(target_dt_sec),
        "alt_shell": "550",
        "dr_sgp4_km": 1.0,
        "dr_sgp4_radial_km": 0.0,
        "dr_sgp4_along_km": 1.0,
        "dr_sgp4_cross_km": 0.0,
        "dr_hifi_km": 1.5,
        "dr_hifi_radial_km": 0.1,
        "dr_hifi_along_km": 1.5,
        "dr_hifi_cross_km": 0.0,
        "f107": 130.0,
        "ap": 8.0,
    }


def _corpus_row(
    norad_id: int,
    target_dt_sec: int,
    epoch_i: pd.Timestamp,
    *,
    generation: str = "v1.5",
) -> dict:
    """A row carrying the corpus columns aggregate.py joins (and a few extras)."""
    return {
        "norad_id": norad_id,
        "target_dt_sec": target_dt_sec,
        "epoch_i": epoch_i,
        "alt_shell": "550",
        "generation": generation,
        "dry_mass_kg": 290.0,
        "drag_area_m2": 2.0,
        "srp_area_m2": 5.0,
        # Extra corpus columns aggregate should silently drop on the join side.
        "epoch_j": epoch_i + pd.Timedelta(seconds=target_dt_sec),
        "actual_dt_sec": float(target_dt_sec),
        "line1_i": "1",
        "line2_i": "2",
    }


@pytest.fixture
def smoke_output(tmp_path):
    """Synthesize a tiny outputs/ + corpus that aggregate.py can consume.

    Three runs across two sats × two Δt buckets, plus one corpus row with
    no matching per-run parquet (the "failed run" case).
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    t = pd.Timestamp("2026-04-01T00:00:00Z")
    runs = [
        _per_run_row(0, 44713, 21_600, t),
        _per_run_row(1, 44713, 86_400, t),
        _per_run_row(2, 52464, 21_600, t + pd.Timedelta(hours=1)),
    ]
    for row in runs:
        pd.DataFrame([row]).to_parquet(output_dir / f"run_{row['run_id']}.parquet", index=False)

    corpus = pd.DataFrame(
        [
            _corpus_row(44713, 21_600, t, generation="v1.5"),
            _corpus_row(44713, 86_400, t, generation="v1.5"),
            _corpus_row(52464, 21_600, t + pd.Timedelta(hours=1), generation="v2-mini"),
            # No matching run_<id>.parquet — simulates a failed run.
            _corpus_row(99999, 21_600, t, generation="v1.0"),
        ]
    )
    tles_path = tmp_path / "tles_cache.parquet"
    corpus.to_parquet(tles_path, index=False)

    return output_dir, tles_path


class TestAggregate:
    def test_row_count_equals_n_runs(self, smoke_output):
        output_dir, tles = smoke_output
        df = aggregate(output_dir, tles)
        # 3 per-run parquets exist; the unmatched corpus row is dropped
        # by the left-from-per-run merge direction.
        assert len(df) == 3

    def test_all_carried_columns_present(self, smoke_output):
        output_dir, tles = smoke_output
        df = aggregate(output_dir, tles)
        for col in CARRIED_COLUMNS:
            assert col in df.columns, f"missing carried column: {col}"
            assert df[col].notna().all(), f"NaN in carried column after merge: {col}"

    def test_per_run_schema_preserved(self, smoke_output):
        output_dir, tles = smoke_output
        df = aggregate(output_dir, tles)
        # The 17 per-run columns survive the merge intact, no _x/_y suffixes.
        for col in [
            "run_id",
            "t_i",
            "t_j",
            "actual_dt_sec",
            "alt_shell",
            "dr_sgp4_km",
            "dr_hifi_km",
            "dr_sgp4_radial_km",
            "dr_hifi_cross_km",
            "f107",
            "ap",
        ]:
            assert col in df.columns
        assert "actual_dt_sec_x" not in df.columns
        assert "alt_shell_y" not in df.columns

    def test_generation_carried_correctly(self, smoke_output):
        output_dir, tles = smoke_output
        df = aggregate(output_dir, tles)
        per_sat = dict(zip(df["norad_id"], df["generation"], strict=True))
        assert per_sat[44713] == "v1.5"
        assert per_sat[52464] == "v2-mini"

    def test_run_id_unique(self, smoke_output):
        output_dir, tles = smoke_output
        df = aggregate(output_dir, tles)
        assert df["run_id"].is_unique


class TestAggregateFailures:
    def test_empty_output_dir_raises(self, tmp_path):
        empty = tmp_path / "outputs"
        empty.mkdir()
        corpus = tmp_path / "c.parquet"
        pd.DataFrame([_corpus_row(44713, 21_600, pd.Timestamp("2026-04-01T00:00:00Z"))]).to_parquet(
            corpus, index=False
        )
        with pytest.raises(FileNotFoundError, match="no run_.*parquet"):
            aggregate(empty, corpus)

    def test_unmatched_per_run_raises(self, tmp_path):
        # Per-run row whose (norad_id, target_dt_sec, t_i) doesn't exist in
        # the corpus — common if --output-dir and --tles come from
        # different sweep batches.
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        t = pd.Timestamp("2026-04-01T00:00:00Z")
        pd.DataFrame([_per_run_row(0, 44713, 21_600, t)]).to_parquet(
            output_dir / "run_0.parquet", index=False
        )

        # Corpus row exists but with a different epoch_i.
        corpus = tmp_path / "c.parquet"
        pd.DataFrame([_corpus_row(44713, 21_600, t + pd.Timedelta(days=99))]).to_parquet(
            corpus, index=False
        )

        with pytest.raises(RuntimeError, match="had no corpus match"):
            aggregate(output_dir, corpus)


class TestJoinKeysSurfaceAtModuleLevel:
    """Guard against accidental drift between JOIN_KEYS, CARRIED_COLUMNS, and the
    per-run / corpus schemas. If these tests fail, downstream figure scripts
    that consume `all_runs.parquet` will break silently."""

    def test_join_keys_match_per_run_schema(self):
        assert "t_i" in JOIN_KEYS  # renamed from corpus's epoch_i
        assert "norad_id" in JOIN_KEYS
        assert "target_dt_sec" in JOIN_KEYS

    def test_carried_columns_are_disjoint_from_per_run(self):
        per_run_cols = set(_per_run_row(0, 0, 0, pd.Timestamp("2026-01-01T00:00:00Z")))
        assert per_run_cols.isdisjoint(CARRIED_COLUMNS), (
            "carried columns collide with per-run schema; would force _x/_y suffixes"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
