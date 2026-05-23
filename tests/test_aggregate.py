"""Unit tests for sweep.aggregate."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from sweep.aggregate import CARRIED_COLUMNS, aggregate


def _per_run_row(run_id: int, norad_id: int, target_dt_sec: int, t_i: pd.Timestamp) -> dict:
    """One row matching the hook's `comparison.parquet` schema."""
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
    """A corpus row carrying CARRIED_COLUMNS + a few extras aggregate drops."""
    return {
        "norad_id": norad_id,
        "target_dt_sec": target_dt_sec,
        "epoch_i": epoch_i,
        "alt_shell": "550",
        "generation": generation,
        "dry_mass_kg": 290.0,
        "drag_area_m2": 2.0,
        "srp_area_m2": 5.0,
        # Extras aggregate should silently drop on the join side.
        "epoch_j": epoch_i + pd.Timedelta(seconds=target_dt_sec),
        "actual_dt_sec": float(target_dt_sec),
        "line1_i": "1",
        "line2_i": "2",
    }


def _write_manifest(
    path: Path,
    entries: list[tuple[int, str, Path | None]],
) -> None:
    """Write a minimal manifest with the given (run_id, status, comparison_path) rows.

    For ``status == "ok"`` runs, ``comparison_path`` becomes the
    ``extra_outputs.comparison`` registration. ``None`` is also valid
    (an ok run whose hook ran but registered no `comparison` output);
    ``lazy_extra_outputs`` then treats the run as not-registered.
    """
    header = {
        "schema_version": 1,
        "script_sha256": "0" * 64,
        "gmat_sweep_version": "test",
        "gmat_run_version": "test",
        "gmat_install_version": "test",
        "python_version": "3.12",
        "os_platform": "test",
        "sweep_seed": None,
        "parameter_spec": {"_kind": "explicit", "columns": [], "rows": []},
        "run_count": len(entries),
        "backend": "test",
        "postprocess": "sweep.run_sweep:_postprocess_run",
    }
    now = dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=dt.UTC).isoformat()
    lines = [json.dumps(header, sort_keys=True)]
    for run_id, status, comparison_path in entries:
        extra: dict[str, str] = {}
        postprocess_status = "none"
        if status == "ok" and comparison_path is not None:
            extra["comparison"] = str(comparison_path)
            postprocess_status = "ok"
        entry = {
            "run_id": run_id,
            "overrides": {},
            "context": {},
            "status": status,
            "output_paths": {},
            "extra_outputs": extra,
            "postprocess_status": postprocess_status,
            "solver_paths": {},
            "converged": {},
            "started_at": now,
            "ended_at": now,
            "duration_s": 1.0,
            "stderr": None,
            "log_path": None,
        }
        lines.append(json.dumps(entry, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def smoke_output(tmp_path):
    """Synthesize a manifest + corpus that aggregate.py can consume.

    Three ok runs across two sats × two Δt buckets (each with a real
    comparison.parquet on disk under outputs/run-<id>/), plus one failed
    run with no comparison output.
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    t = pd.Timestamp("2026-04-01T00:00:00Z")
    runs = [
        _per_run_row(0, 44713, 21_600, t),
        _per_run_row(1, 44713, 86_400, t),
        _per_run_row(2, 52464, 21_600, t + pd.Timedelta(hours=1)),
    ]
    comparison_paths: list[Path] = []
    for row in runs:
        run_dir = output_dir / f"run-{row['run_id']}"
        run_dir.mkdir()
        comp_path = run_dir / "comparison.parquet"
        pd.DataFrame([row]).to_parquet(comp_path, index=False)
        comparison_paths.append(comp_path)

    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest_path,
        [
            (0, "ok", comparison_paths[0]),
            (1, "ok", comparison_paths[1]),
            (2, "ok", comparison_paths[2]),
            (3, "failed", None),
        ],
    )

    # The corpus is keyed by row index = run_id (matches
    # `pairs.reset_index(drop=True)` in `sweep.run_sweep.main`). Row 0
    # corresponds to run_id 0, row 1 to run_id 1, etc.
    corpus = pd.DataFrame(
        [
            _corpus_row(44713, 21_600, t, generation="v1.5"),
            _corpus_row(44713, 86_400, t, generation="v1.5"),
            _corpus_row(52464, 21_600, t + pd.Timedelta(hours=1), generation="v2-mini"),
            # Row 3 — the failed manifest entry — no per-run comparison parquet.
            _corpus_row(99999, 21_600, t, generation="v1.0"),
        ]
    )
    tles_path = tmp_path / "tles_cache.parquet"
    corpus.to_parquet(tles_path, index=False)

    return tles_path, manifest_path


class TestAggregate:
    def test_row_count_equals_ok_runs(self, smoke_output):
        tles, manifest = smoke_output
        df = aggregate(tles, manifest)
        # 3 ok manifest entries × 1 row each. The failed entry contributes
        # no comparison output; row 3 in the corpus has no per-run match
        # and is dropped by the left merge on run_id.
        assert len(df) == 3

    def test_all_carried_columns_present(self, smoke_output):
        tles, manifest = smoke_output
        df = aggregate(tles, manifest)
        for col in CARRIED_COLUMNS:
            assert col in df.columns, f"missing carried column: {col}"
            assert df[col].notna().all(), f"NaN in carried column after merge: {col}"

    def test_per_run_schema_preserved(self, smoke_output):
        tles, manifest = smoke_output
        df = aggregate(tles, manifest)
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
        tles, manifest = smoke_output
        df = aggregate(tles, manifest)
        per_sat = dict(zip(df["norad_id"], df["generation"], strict=True))
        assert per_sat[44713] == "v1.5"
        assert per_sat[52464] == "v2-mini"

    def test_run_id_unique(self, smoke_output):
        tles, manifest = smoke_output
        df = aggregate(tles, manifest)
        assert df["run_id"].is_unique


class TestManifestDrivenIteration:
    def test_stale_per_run_parquet_not_in_manifest_is_excluded(self, tmp_path):
        # A stray comparison.parquet from a previous batch lingers under
        # outputs/ but its run_id is not referenced by the current
        # manifest's extra_outputs map. `lazy_extra_outputs` is
        # manifest-driven, so the stray file must not land in the result.
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        t = pd.Timestamp("2026-04-01T00:00:00Z")

        # Two comparison parquets on disk:
        live_dir = output_dir / "run-0"
        live_dir.mkdir()
        live_path = live_dir / "comparison.parquet"
        pd.DataFrame([_per_run_row(0, 44713, 21_600, t)]).to_parquet(live_path, index=False)

        stale_dir = output_dir / "run-999"
        stale_dir.mkdir()
        pd.DataFrame([_per_run_row(999, 99999, 21_600, t)]).to_parquet(
            stale_dir / "comparison.parquet", index=False
        )

        # Manifest references only run_id=0:
        manifest_path = tmp_path / "manifest.jsonl"
        _write_manifest(manifest_path, [(0, "ok", live_path)])

        corpus = tmp_path / "c.parquet"
        pd.DataFrame([_corpus_row(44713, 21_600, t)]).to_parquet(corpus, index=False)

        df = aggregate(corpus, manifest_path)
        # Only run_0 lands in the result; the stray run-999/comparison.parquet
        # is invisible to the manifest-driven aggregator.
        assert len(df) == 1
        assert df["run_id"].tolist() == [0]

    def test_failed_manifest_entries_are_skipped(self, tmp_path):
        # A failed entry has no comparison output; aggregation must not
        # error and the failed run must not appear in the result.
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        t = pd.Timestamp("2026-04-01T00:00:00Z")

        run0_dir = output_dir / "run-0"
        run0_dir.mkdir()
        comp_path = run0_dir / "comparison.parquet"
        pd.DataFrame([_per_run_row(0, 44713, 21_600, t)]).to_parquet(comp_path, index=False)

        manifest_path = tmp_path / "manifest.jsonl"
        _write_manifest(
            manifest_path,
            [(0, "ok", comp_path), (1, "failed", None), (2, "skipped", None)],
        )

        corpus = tmp_path / "c.parquet"
        pd.DataFrame(
            [
                _corpus_row(44713, 21_600, t),
                _corpus_row(99999, 21_600, t),
                _corpus_row(88888, 21_600, t),
            ]
        ).to_parquet(corpus, index=False)

        df = aggregate(corpus, manifest_path)
        assert len(df) == 1
        assert df["run_id"].tolist() == [0]


class TestAggregateFailures:
    def test_no_ok_entries_raises(self, tmp_path):
        manifest_path = tmp_path / "manifest.jsonl"
        _write_manifest(manifest_path, [(0, "failed", None), (1, "skipped", None)])
        corpus = tmp_path / "c.parquet"
        pd.DataFrame([_corpus_row(44713, 21_600, pd.Timestamp("2026-04-01T00:00:00Z"))]).to_parquet(
            corpus, index=False
        )
        # `lazy_extra_outputs` raises SweepConfigError because no run
        # registered the `comparison` key — that bubbles up through
        # aggregate().
        from gmat_sweep import SweepConfigError

        with pytest.raises(SweepConfigError):
            aggregate(corpus, manifest_path)

    def test_unmatched_per_run_raises(self, tmp_path):
        # Per-run row whose run_id has no corresponding corpus row —
        # common if --tles came from a different sweep batch than the
        # manifest.
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        t = pd.Timestamp("2026-04-01T00:00:00Z")
        run5_dir = output_dir / "run-5"
        run5_dir.mkdir()
        comp_path = run5_dir / "comparison.parquet"
        pd.DataFrame([_per_run_row(5, 44713, 21_600, t)]).to_parquet(comp_path, index=False)

        manifest_path = tmp_path / "manifest.jsonl"
        _write_manifest(manifest_path, [(5, "ok", comp_path)])

        # Corpus has only one row (run_id 0), but the manifest says
        # run_id 5 succeeded — the join misses.
        corpus = tmp_path / "c.parquet"
        pd.DataFrame([_corpus_row(44713, 21_600, t)]).to_parquet(corpus, index=False)

        with pytest.raises(RuntimeError, match="had no corpus match"):
            aggregate(corpus, manifest_path)


class TestCarriedColumnsAreDisjointFromPerRun:
    """Guard against carry/per-run column collisions that would force _x/_y suffixes."""

    def test_carried_columns_are_disjoint_from_per_run(self):
        per_run_cols = set(_per_run_row(0, 0, 0, pd.Timestamp("2026-01-01T00:00:00Z")))
        assert per_run_cols.isdisjoint(CARRIED_COLUMNS), (
            "carried columns collide with per-run schema; would force _x/_y suffixes"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
