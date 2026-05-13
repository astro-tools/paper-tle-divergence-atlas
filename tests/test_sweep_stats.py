"""Unit tests for sweep.sweep_stats."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sweep.sweep_stats import (
    BUCKET_LABELS,
    ManifestSummary,
    compute_bucket_stats,
    compute_manifest_summary,
    compute_shell_gen_stats,
    format_report,
)


def _all_runs_row(
    target_dt_sec: int,
    dr_sgp4: float,
    dr_hifi: float,
    *,
    shell: str = "550",
    generation: str = "v1.5",
) -> dict:
    return {
        "run_id": 0,
        "norad_id": 44713,
        "target_dt_sec": target_dt_sec,
        "actual_dt_sec": float(target_dt_sec),
        "alt_shell": shell,
        "generation": generation,
        "dr_sgp4_km": dr_sgp4,
        "dr_hifi_km": dr_hifi,
    }


@pytest.fixture
def synthetic_all_runs():
    """Two Δt buckets × two (shell, gen) cells × 5 runs each."""
    rng = np.random.default_rng(42)
    rows = []
    for target_dt_sec in (21_600, 86_400):
        for shell, gen in [("540", "v1.5"), ("550", "v2-mini")]:
            for _ in range(5):
                # Synthetic dr scales with bucket so the medians come out ordered.
                base = target_dt_sec / 21_600.0
                rows.append(
                    _all_runs_row(
                        target_dt_sec,
                        base + rng.normal(0, 0.1),
                        1.1 * base + rng.normal(0, 0.1),
                        shell=shell,
                        generation=gen,
                    )
                )
    return pd.DataFrame(rows)


class TestComputeBucketStats:
    def test_one_row_per_bucket(self, synthetic_all_runs):
        df = compute_bucket_stats(synthetic_all_runs)
        assert set(df["target_dt_sec"]) == {21_600, 86_400}
        assert len(df) == 2

    def test_bucket_label_uses_human_labels(self, synthetic_all_runs):
        df = compute_bucket_stats(synthetic_all_runs)
        assert set(df["bucket"]) == {BUCKET_LABELS[21_600], BUCKET_LABELS[86_400]}

    def test_n_matches_group_size(self, synthetic_all_runs):
        df = compute_bucket_stats(synthetic_all_runs)
        # Each bucket has 2 (shell, gen) cells × 5 runs = 10 rows.
        assert (df["n"] == 10).all()

    def test_median_within_iqr_bounds(self, synthetic_all_runs):
        df = compute_bucket_stats(synthetic_all_runs)
        assert (df["dr_sgp4_km_q1"] <= df["dr_sgp4_km_med"]).all()
        assert (df["dr_sgp4_km_med"] <= df["dr_sgp4_km_q3"]).all()
        assert (df["dr_hifi_km_q1"] <= df["dr_hifi_km_med"]).all()
        assert (df["dr_hifi_km_med"] <= df["dr_hifi_km_q3"]).all()

    def test_median_ordering_by_bucket(self, synthetic_all_runs):
        # Synthetic data: dr scales linearly with bucket → 6h median < 1d median.
        df = compute_bucket_stats(synthetic_all_runs).sort_values("target_dt_sec")
        assert df.iloc[0]["dr_sgp4_km_med"] < df.iloc[1]["dr_sgp4_km_med"]

    def test_known_quantiles_for_uniform_input(self):
        # 11 evenly-spaced values 0..10 → q1=2.5, median=5, q3=7.5.
        rows = [
            _all_runs_row(target_dt_sec=21_600, dr_sgp4=float(v), dr_hifi=float(v))
            for v in range(11)
        ]
        df = compute_bucket_stats(pd.DataFrame(rows))
        row = df.iloc[0]
        assert row["dr_sgp4_km_med"] == pytest.approx(5.0)
        assert row["dr_sgp4_km_q1"] == pytest.approx(2.5)
        assert row["dr_sgp4_km_q3"] == pytest.approx(7.5)


class TestComputeShellGenStats:
    def test_one_row_per_combination(self, synthetic_all_runs):
        df = compute_shell_gen_stats(synthetic_all_runs)
        # 2 buckets × 2 (shell, gen) cells = 4 rows.
        assert len(df) == 4

    def test_no_nan_rows(self, synthetic_all_runs):
        df = compute_shell_gen_stats(synthetic_all_runs)
        # Synthetic data has all combinations populated; no NaN medians.
        assert df["dr_sgp4_km_med"].notna().all()
        assert df["dr_hifi_km_med"].notna().all()

    def test_empty_combinations_are_dropped_not_nan(self):
        # Only one (shell, gen) populated → only one row, not a NaN row
        # for the missing v2-mini cell.
        rows = [_all_runs_row(21_600, 1.0, 1.1, shell="540", generation="v1.5") for _ in range(3)]
        df = compute_shell_gen_stats(pd.DataFrame(rows))
        assert len(df) == 1
        assert df.iloc[0]["generation"] == "v1.5"


class TestComputeManifestSummary:
    def _write_manifest(self, path: Path, entries: list[dict]) -> None:
        """Write a tiny manifest with a minimal header + the given entries."""
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
        }
        lines = [json.dumps(header, sort_keys=True)]
        lines.extend(json.dumps(e, sort_keys=True) for e in entries)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _entry(
        self,
        run_id: int,
        status: str,
        *,
        stderr: str | None = None,
    ) -> dict:
        now = dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=dt.UTC).isoformat()
        return {
            "run_id": run_id,
            "overrides": {},
            "status": status,
            "output_paths": {},
            "started_at": now,
            "ended_at": now,
            "duration_s": 1.0,
            "stderr": stderr,
            "log_path": None,
        }

    def test_counts_split_by_status(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        self._write_manifest(
            path,
            [
                self._entry(0, "ok"),
                self._entry(1, "ok"),
                self._entry(2, "failed", stderr="RuntimeError: boom\nstack frame…"),
                self._entry(3, "failed", stderr="RuntimeError: boom\nother frame"),
                self._entry(4, "skipped"),
            ],
        )
        summary = compute_manifest_summary(path)
        assert isinstance(summary, ManifestSummary)
        assert summary.n_total == 5
        assert summary.n_ok == 2
        assert summary.n_failed == 2
        assert summary.n_skipped == 1

    def test_stderr_buckets_share_first_line(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        self._write_manifest(
            path,
            [
                self._entry(0, "failed", stderr="RuntimeError: boom"),
                self._entry(1, "failed", stderr="RuntimeError: boom\nstack"),
                self._entry(2, "failed", stderr="ValueError: nan in state"),
            ],
        )
        summary = compute_manifest_summary(path)
        buckets = summary.failed_stderr_buckets.set_index("stderr_first_line")["count"].to_dict()
        assert buckets["RuntimeError: boom"] == 2
        assert buckets["ValueError: nan in state"] == 1

    def test_no_failed_runs_yields_empty_buckets(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        self._write_manifest(path, [self._entry(0, "ok"), self._entry(1, "ok")])
        summary = compute_manifest_summary(path)
        assert summary.n_failed == 0
        assert summary.failed_stderr_buckets.empty

    def test_none_stderr_is_bucketed_explicitly(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        self._write_manifest(path, [self._entry(0, "failed", stderr=None)])
        summary = compute_manifest_summary(path)
        names = summary.failed_stderr_buckets["stderr_first_line"].tolist()
        assert names == ["(no stderr)"]


class TestFormatReport:
    def test_includes_bucket_and_shell_sections(self, synthetic_all_runs):
        report = format_report(synthetic_all_runs, manifest_path=None)
        assert "Per Δt bucket" in report
        assert "Per (alt_shell, generation, bucket)" in report
        assert "Manifest failure accounting" not in report  # no manifest passed

    def test_includes_manifest_section_when_passed(self, synthetic_all_runs, tmp_path):
        path = tmp_path / "manifest.jsonl"
        header = {
            "schema_version": 1,
            "script_sha256": "0" * 64,
            "gmat_sweep_version": "t",
            "gmat_run_version": "t",
            "gmat_install_version": "t",
            "python_version": "3.12",
            "os_platform": "t",
            "sweep_seed": None,
            "parameter_spec": {"_kind": "explicit", "columns": [], "rows": []},
            "run_count": 0,
            "backend": "t",
        }
        path.write_text(json.dumps(header, sort_keys=True) + "\n", encoding="utf-8")
        report = format_report(synthetic_all_runs, manifest_path=path)
        assert "Manifest failure accounting" in report

    def test_row_count_in_header(self, synthetic_all_runs):
        report = format_report(synthetic_all_runs, manifest_path=None)
        assert f"{len(synthetic_all_runs)} successful run(s)" in report


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
