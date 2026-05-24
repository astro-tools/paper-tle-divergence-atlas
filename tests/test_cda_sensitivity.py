"""Unit tests for sweep.cda_sensitivity and sweep.cda_sensitivity_table.

GMAT-touching code paths are exercised end-to-end through the
`make cda-sensitivity` smoke run, not pytest. Tested here:

- `_scale_drag_area` produces a new spec with `Sat.DragArea` × factor
  and every other override preserved bit-identically.
- `select_v2_mini_subset` reads the subset id file, filters by
  generation, and respects the `reset_index(drop=True)` = run_id
  convention.
- `_aggregate_factor` promotes the `lazy_extra_outputs` run_id index
  back to a column (pins the regression where it was being dropped),
  excludes non-ok manifest entries, and preserves the CdA-only schema
  columns.
- `cda_sensitivity_table.build` produces a valid LaTeX table and a
  JSON summary from a synthetic three-frame input; alignment-mismatch
  cases raise.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest
from gmat_sweep import RunSpec

from sweep import cda_sensitivity_table
from sweep.cda_sensitivity import (
    GENERATION_V2_MINI,
    _aggregate_factor,
    _scale_drag_area,
    select_v2_mini_subset,
)


def _make_spec(drag_area: float = 4.5) -> RunSpec:
    return RunSpec(
        script_path=Path("/tmp/mission.script"),
        overrides={
            "Sat.Epoch": "01 Apr 2026 00:00:00.000",
            "Sat.X": 6800.0,
            "Sat.Y": 0.0,
            "Sat.Z": 0.0,
            "Sat.VX": 0.0,
            "Sat.VY": 7.5,
            "Sat.VZ": 0.0,
            "Sat.DryMass": 800.0,
            "Sat.Cd": 2.2,
            "Sat.DragArea": drag_area,
            "Sat.Cr": 1.5,
            "Sat.SRPArea": 10.0,
            "elapsed_seconds.Value": 86400.0,
        },
        output_dir=Path("/tmp/run_0"),
        run_id=0,
        seed=None,
        run_options={"overwrite": True},
    )


class TestScaleDragArea:
    def test_low_factor_scales_drag_area(self) -> None:
        spec = _make_spec(drag_area=4.5)
        out = _scale_drag_area(spec, 0.8)
        assert out.overrides["Sat.DragArea"] == pytest.approx(3.6)

    def test_high_factor_scales_drag_area(self) -> None:
        spec = _make_spec(drag_area=4.5)
        out = _scale_drag_area(spec, 1.2)
        assert out.overrides["Sat.DragArea"] == pytest.approx(5.4)

    def test_baseline_factor_returns_unchanged_drag_area(self) -> None:
        spec = _make_spec(drag_area=4.5)
        out = _scale_drag_area(spec, 1.0)
        assert out.overrides["Sat.DragArea"] == pytest.approx(4.5)

    def test_other_overrides_unchanged(self) -> None:
        spec = _make_spec(drag_area=4.5)
        out = _scale_drag_area(spec, 0.8)
        # Every key except Sat.DragArea must be bit-identical.
        for key, value in spec.overrides.items():
            if key == "Sat.DragArea":
                continue
            assert out.overrides[key] == value, f"override {key} drifted"

    def test_returns_fresh_runspec(self) -> None:
        # Modifying the returned spec must not affect the input spec.
        spec = _make_spec(drag_area=4.5)
        out = _scale_drag_area(spec, 0.8)
        out.overrides["Sat.DragArea"] = 99.0
        assert spec.overrides["Sat.DragArea"] == pytest.approx(4.5)

    def test_runspec_metadata_preserved(self) -> None:
        spec = _make_spec()
        out = _scale_drag_area(spec, 0.8)
        assert out.script_path == spec.script_path
        assert out.output_dir == spec.output_dir
        assert out.run_id == spec.run_id
        assert out.seed == spec.seed
        assert out.run_options == spec.run_options

    def test_srp_area_unchanged(self) -> None:
        # The issue is explicit: SRP area is unchanged by the factor.
        spec = _make_spec()
        for factor in (0.8, 1.0, 1.2):
            out = _scale_drag_area(spec, factor)
            assert out.overrides["Sat.SRPArea"] == pytest.approx(
                spec.overrides["Sat.SRPArea"],
            )


def _make_corpus() -> pd.DataFrame:
    """Synthetic corpus: 10 rows across two generations and two shells.

    Row index = run_id (the upstream `reset_index(drop=True)` convention).
    """
    rows = []
    for run_id in range(10):
        gen = GENERATION_V2_MINI if run_id % 2 == 0 else "v1.5"
        rows.append(
            {
                "norad_id": 50000 + run_id,
                "target_dt_sec": 86_400,
                "alt_shell": "550",
                "generation": gen,
                "drag_area_m2": 4.5 if gen == GENERATION_V2_MINI else 2.0,
            },
        )
    return pd.DataFrame(rows).reset_index(drop=True)


class TestSelectV2MiniSubset:
    def test_filters_to_v2_mini_only(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("\n".join(str(i) for i in range(10)) + "\n")
        out = select_v2_mini_subset(corpus, ids_path)
        assert (out["generation"] == GENERATION_V2_MINI).all()
        # Even run_ids are v2-mini in the synthetic corpus.
        assert list(out.index) == [0, 2, 4, 6, 8]

    def test_respects_subset_id_list(self, tmp_path: Path) -> None:
        # Only ids {0, 4, 8} (all v2-mini) and {1} (v1.5) are listed.
        corpus = _make_corpus()
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("0\n1\n4\n8\n")
        out = select_v2_mini_subset(corpus, ids_path)
        assert list(out.index) == [0, 4, 8]

    def test_missing_id_file_raises(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        with pytest.raises(FileNotFoundError, match="sensitivity subset id file"):
            select_v2_mini_subset(corpus, tmp_path / "does-not-exist.txt")

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        corpus = _make_corpus()
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("0\n\n2\n   \n4\n")
        out = select_v2_mini_subset(corpus, ids_path)
        assert list(out.index) == [0, 2, 4]


def _synthetic_runs(
    factor: float,
    *,
    medians_km: dict[tuple[str, int], float],
    n_per_cell: int = 10,
    run_id_offset: int = 0,
) -> pd.DataFrame:
    """Build a synthetic per-run frame at the given CdA factor.

    `medians_km` maps (alt_shell, target_dt_sec) to the median |Δr|_hifi
    for that cell. We emit `n_per_cell` rows with `dr_hifi_km` equal to
    that median so the per-cell median in the resulting frame matches
    `medians_km` exactly — handy for asserting on table output.
    """
    rows = []
    run_id = run_id_offset
    for (shell, target_dt_sec), median_km in medians_km.items():
        for _ in range(n_per_cell):
            rows.append(
                {
                    "run_id": run_id,
                    "norad_id": 50000 + run_id,
                    "target_dt_sec": target_dt_sec,
                    "alt_shell": shell,
                    "dr_hifi_km": median_km,
                    "dr_sgp4_km": 0.5,
                    "generation": GENERATION_V2_MINI,
                    "cda_factor": factor,
                },
            )
            run_id += 1
    return pd.DataFrame(rows)


class TestCdaSensitivityTable:
    @pytest.fixture
    def synthetic_inputs(self, tmp_path: Path) -> dict[str, Path]:
        # Same run_id set for all three frames so _validate_alignment passes.
        baseline_medians = {("550", 21_600): 1.5, ("560", 86_400): 8.0}
        low_medians = {("550", 21_600): 1.3, ("560", 86_400): 7.0}
        high_medians = {("550", 21_600): 1.7, ("560", 86_400): 9.0}

        baseline = _synthetic_runs(1.0, medians_km=baseline_medians)
        low = _synthetic_runs(0.8, medians_km=low_medians)
        high = _synthetic_runs(1.2, medians_km=high_medians)

        # The driver loads baseline from `all_runs.parquet` and filters
        # to subset ids; emit the file and the subset list that pass
        # those filters.
        all_runs_path = tmp_path / "all_runs.parquet"
        baseline.to_parquet(all_runs_path, index=False)

        # Off-baseline frames carry the same run_ids; rewrite to point
        # at the same id set.
        low["run_id"] = baseline["run_id"].to_numpy()
        high["run_id"] = baseline["run_id"].to_numpy()
        low_path = tmp_path / "cda_low.parquet"
        high_path = tmp_path / "cda_high.parquet"
        low.to_parquet(low_path, index=False)
        high.to_parquet(high_path, index=False)

        subset_path = tmp_path / "ids.txt"
        subset_path.write_text("\n".join(str(int(r)) for r in baseline["run_id"]))

        return {
            "all_runs": all_runs_path,
            "low": low_path,
            "high": high_path,
            "subset": subset_path,
        }

    def test_build_produces_table_and_summary(
        self,
        synthetic_inputs: dict[str, Path],
    ) -> None:
        table_latex, summary = cda_sensitivity_table.build(
            synthetic_inputs["all_runs"],
            synthetic_inputs["low"],
            synthetic_inputs["high"],
            synthetic_inputs["subset"],
        )
        assert "\\begin{tabular}" in table_latex
        assert "\\bottomrule" in table_latex
        # baseline median for 550 × 6 h is 1.5 km
        assert "1.50" in table_latex
        # relative shifts: low = (1.3 - 1.5) / 1.5 = -13.3%
        assert "-13.3" in table_latex
        # high = (1.7 - 1.5) / 1.5 = +13.3%
        assert "+13.3" in table_latex

    def test_summary_signs_consistent(
        self,
        synthetic_inputs: dict[str, Path],
    ) -> None:
        _, summary = cda_sensitivity_table.build(
            synthetic_inputs["all_runs"],
            synthetic_inputs["low"],
            synthetic_inputs["high"],
            synthetic_inputs["subset"],
        )
        # 0.8× ⇒ smaller drag ⇒ smaller |Δr|_hifi ⇒ negative shift everywhere
        assert summary["sign_low_negative_consistent"] is True
        # 1.2× ⇒ larger drag ⇒ larger |Δr|_hifi ⇒ positive shift everywhere
        assert summary["sign_high_positive_consistent"] is True

    def test_summary_pooled_baseline_matches_median(
        self,
        synthetic_inputs: dict[str, Path],
    ) -> None:
        _, summary = cda_sensitivity_table.build(
            synthetic_inputs["all_runs"],
            synthetic_inputs["low"],
            synthetic_inputs["high"],
            synthetic_inputs["subset"],
        )
        # Median of cell-medians {1.5, 8.0} is 4.75
        assert summary["pooled_baseline_km"] == pytest.approx(4.75)

    def test_misaligned_run_ids_raise(
        self,
        tmp_path: Path,
        synthetic_inputs: dict[str, Path],
    ) -> None:
        # Replace the low frame with one missing a row to trigger the guard.
        low = pd.read_parquet(synthetic_inputs["low"])
        low.iloc[:-1].to_parquet(synthetic_inputs["low"], index=False)
        with pytest.raises(SystemExit, match="run_id counts"):
            cda_sensitivity_table.build(
                synthetic_inputs["all_runs"],
                synthetic_inputs["low"],
                synthetic_inputs["high"],
                synthetic_inputs["subset"],
            )

    def test_disjoint_run_ids_raise(
        self,
        tmp_path: Path,
        synthetic_inputs: dict[str, Path],
    ) -> None:
        # Shift the low frame's run_ids to a disjoint set of same size.
        low = pd.read_parquet(synthetic_inputs["low"])
        low["run_id"] = low["run_id"] + 10_000
        low.to_parquet(synthetic_inputs["low"], index=False)
        with pytest.raises(SystemExit, match="miss baseline run_ids"):
            cda_sensitivity_table.build(
                synthetic_inputs["all_runs"],
                synthetic_inputs["low"],
                synthetic_inputs["high"],
                synthetic_inputs["subset"],
            )

    def test_table_format_shift(self) -> None:
        # Direct test of the relative-shift formatter; protects against
        # sign and precision regressions.
        assert cda_sensitivity_table._fmt_shift(1.2, 1.0) == "+20.0\\%"
        assert cda_sensitivity_table._fmt_shift(0.8, 1.0) == "-20.0\\%"
        assert cda_sensitivity_table._fmt_shift(float("nan"), 1.0) == "---"
        assert cda_sensitivity_table._fmt_shift(1.0, 0.0) == "---"


def _cda_per_run_row(run_id: int, factor: float) -> dict:
    """One row matching `_postprocess_run`'s CdA-arm comparison schema."""
    return {
        "run_id": run_id,
        "norad_id": 50_000 + run_id,
        "target_dt_sec": 86_400,
        "t_i": pd.Timestamp("2026-04-01T00:00:00Z"),
        "t_j": pd.Timestamp("2026-04-02T00:00:00Z"),
        "actual_dt_sec": 86_400.0,
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
        "cda_factor": factor,
        "applied_drag_area_m2": 4.5 * factor,
        "generation": GENERATION_V2_MINI,
    }


def _write_cda_manifest(
    path: Path,
    entries: list[tuple[int, str, Path | None]],
) -> None:
    """Write a minimal manifest with the given (run_id, status, comparison_path) rows.

    Mirrors the `lazy_extra_outputs`-compatible shape used in
    `tests/test_aggregate.py` so the CdA aggregator sees a realistic
    manifest. ``status == "failed"`` entries get no `extra_outputs.comparison`
    registration.
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
        "postprocess": "sweep.cda_sensitivity:_postprocess_run",
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


class TestAggregateFactor:
    """Aggregator unit tests.

    The headline assertion is ``run_id in df.columns``: `lazy_extra_outputs`
    returns a run_id-indexed frame, and `_aggregate_factor` must promote
    that index back to a column (the same idiom `sweep.aggregate` and
    `sweep.maneuver_threshold_sensitivity` already use). The previous
    `reset_index(drop=True)` quietly dropped it, so the top-level
    `all_runs_cda_{low,high}.parquet` shipped without run_id and the
    downstream table builder blew up on the v2-mini join.
    """

    @pytest.fixture
    def cda_manifest(self, tmp_path: Path) -> Path:
        factor = 0.8
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        comparison_paths: list[Path] = []
        for run_id in (10, 11, 12):
            run_dir = output_dir / f"run-{run_id}"
            run_dir.mkdir()
            comp_path = run_dir / "comparison.parquet"
            pd.DataFrame([_cda_per_run_row(run_id, factor)]).to_parquet(
                comp_path,
                index=False,
            )
            comparison_paths.append(comp_path)
        manifest_path = tmp_path / "manifest.jsonl"
        _write_cda_manifest(
            manifest_path,
            [
                (10, "ok", comparison_paths[0]),
                (11, "ok", comparison_paths[1]),
                (12, "ok", comparison_paths[2]),
                (13, "failed", None),
            ],
        )
        return manifest_path

    def test_run_id_is_a_column(self, cda_manifest: Path) -> None:
        df = _aggregate_factor(cda_manifest, factor=0.8)
        assert "run_id" in df.columns
        assert set(df["run_id"].astype(int)) == {10, 11, 12}

    def test_excludes_failed_runs(self, cda_manifest: Path) -> None:
        df = _aggregate_factor(cda_manifest, factor=0.8)
        assert 13 not in set(df["run_id"].astype(int))

    def test_cda_columns_preserved(self, cda_manifest: Path) -> None:
        df = _aggregate_factor(cda_manifest, factor=0.8)
        assert (df["cda_factor"] == 0.8).all()
        assert "applied_drag_area_m2" in df.columns
        assert (df["generation"] == GENERATION_V2_MINI).all()
