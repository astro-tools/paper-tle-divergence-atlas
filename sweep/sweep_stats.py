"""First-look statistics over `outputs/all_runs.parquet` + the sweep manifest.

Quick numerical sanity check the human runs after `make sweep` finishes,
before any manuscript figure work begins. Two inputs, one text report:

    all_runs.parquet  ── median/IQR of dr_sgp4_km, dr_hifi_km
            ↓             per Δt bucket; same stratified by
            ↓             (alt_shell, generation); run counts.
            ↓
        report.txt ── + manifest failure tally (ok/failed/skipped)
            ↑           + status='failed' stderr first-line bucketing.
            ↑
      manifest.jsonl

The report is *plain text*. It goes to stdout and (optionally) to a
file. It is not a figure and is not part of the manuscript build —
showyourwork does not see it.

Numbers are deliberately not gated: the H1 signal is that 7-day Δr
grows to tens-to-hundreds of km, so absolute magnitudes do not
indicate "broken." What this script catches: shells or generations
that are systematically empty, infinite/NaN medians, manifest entries
the postprocess step missed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from gmat_sweep import Manifest

BUCKET_LABELS: Final = {
    21_600: "6 h",
    86_400: "1 d",
    259_200: "3 d",
    604_800: "7 d",
}
DR_COLUMNS: Final = ("dr_sgp4_km", "dr_hifi_km")


def _bucket_label(target_dt_sec: int) -> str:
    return BUCKET_LABELS.get(int(target_dt_sec), f"{int(target_dt_sec)} s")


def _median_iqr(values: pd.Series) -> tuple[float, float, float]:
    """Return (median, q1, q3). NaN on empty input."""
    if values.empty:
        return (float("nan"), float("nan"), float("nan"))
    q1, med, q3 = np.quantile(values.to_numpy(), [0.25, 0.5, 0.75])
    return float(med), float(q1), float(q3)


def compute_bucket_stats(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Per-Δt-bucket median/IQR for `dr_sgp4_km` and `dr_hifi_km`, plus n."""
    rows: list[dict] = []
    for target_dt_sec, group in all_runs.groupby("target_dt_sec", sort=True):
        row: dict[str, object] = {
            "target_dt_sec": int(target_dt_sec),
            "bucket": _bucket_label(int(target_dt_sec)),
            "n": int(len(group)),
        }
        for col in DR_COLUMNS:
            med, q1, q3 = _median_iqr(group[col])
            row[f"{col}_med"] = med
            row[f"{col}_q1"] = q1
            row[f"{col}_q3"] = q3
        rows.append(row)
    return pd.DataFrame(rows)


def compute_shell_gen_stats(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Per (alt_shell, generation, Δt-bucket) median/IQR + n.

    Empty cells (no runs for that combination) are omitted rather than
    NaN-filled — they would otherwise mask a real "shell is empty"
    signal as if it were a normal stratum with zero runs.
    """
    rows: list[dict] = []
    keys = ["alt_shell", "generation", "target_dt_sec"]
    for (shell, gen, target_dt_sec), group in all_runs.groupby(keys, sort=True):
        row: dict[str, object] = {
            "alt_shell": str(shell),
            "generation": str(gen),
            "target_dt_sec": int(target_dt_sec),
            "bucket": _bucket_label(int(target_dt_sec)),
            "n": int(len(group)),
        }
        for col in DR_COLUMNS:
            med, q1, q3 = _median_iqr(group[col])
            row[f"{col}_med"] = med
            row[f"{col}_q1"] = q1
            row[f"{col}_q3"] = q3
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    n_total: int
    n_ok: int
    n_failed: int
    n_skipped: int
    failed_stderr_buckets: pd.DataFrame  # cols: stderr_first_line, count


def _classify_stderr(stderr: str | None) -> str:
    """Bucket a stderr blob to its first non-empty line, trimmed.

    Used only for grouping failures; the full stderr is preserved in
    the manifest itself. None and empty stderrs collapse to a single
    "(no stderr)" bucket so the summary always has a defined key.
    """
    if stderr is None:
        return "(no stderr)"
    for line in stderr.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return "(no stderr)"


def compute_manifest_summary(manifest_path: Path) -> ManifestSummary:
    manifest = Manifest.load(manifest_path)
    entries = manifest.entries

    n_total = len(entries)
    by_status: dict[str, int] = {"ok": 0, "failed": 0, "skipped": 0}
    for entry in entries:
        by_status[entry.status] = by_status.get(entry.status, 0) + 1

    failed_stderrs = [_classify_stderr(e.stderr) for e in entries if e.status == "failed"]
    if failed_stderrs:
        counts = pd.Series(failed_stderrs).value_counts()
        buckets = counts.rename_axis("stderr_first_line").reset_index(name="count")
    else:
        buckets = pd.DataFrame(columns=["stderr_first_line", "count"])

    return ManifestSummary(
        n_total=n_total,
        n_ok=by_status.get("ok", 0),
        n_failed=by_status.get("failed", 0),
        n_skipped=by_status.get("skipped", 0),
        failed_stderr_buckets=buckets,
    )


# --- Formatting -----------------------------------------------------------


def _fmt_iqr(med: float, q1: float, q3: float) -> str:
    if not np.isfinite(med):
        return "        NaN"
    return f"{med:7.3f} [{q1:7.3f}, {q3:7.3f}]"


def _fmt_bucket_stats(bucket_stats: pd.DataFrame) -> str:
    lines = [
        "Per Δt bucket (median [IQR] km):",
        "",
        f"  {'bucket':<6}  {'n':>6}  {'dr_sgp4_km':>25}  {'dr_hifi_km':>25}",
        f"  {'-' * 6}  {'-' * 6}  {'-' * 25}  {'-' * 25}",
    ]
    for _, row in bucket_stats.iterrows():
        sgp4 = _fmt_iqr(row["dr_sgp4_km_med"], row["dr_sgp4_km_q1"], row["dr_sgp4_km_q3"])
        hifi = _fmt_iqr(row["dr_hifi_km_med"], row["dr_hifi_km_q1"], row["dr_hifi_km_q3"])
        lines.append(f"  {row['bucket']:<6}  {row['n']:>6}  {sgp4:>25}  {hifi:>25}")
    return "\n".join(lines)


def _fmt_shell_gen_stats(stats: pd.DataFrame) -> str:
    lines = [
        "Per (alt_shell, generation, bucket) (median [IQR] km):",
        "",
        f"  {'shell':<5}  {'gen':<8}  {'bucket':<6}  {'n':>6}  "
        f"{'dr_sgp4_km':>25}  {'dr_hifi_km':>25}",
        f"  {'-' * 5}  {'-' * 8}  {'-' * 6}  {'-' * 6}  {'-' * 25}  {'-' * 25}",
    ]
    for _, row in stats.iterrows():
        sgp4 = _fmt_iqr(row["dr_sgp4_km_med"], row["dr_sgp4_km_q1"], row["dr_sgp4_km_q3"])
        hifi = _fmt_iqr(row["dr_hifi_km_med"], row["dr_hifi_km_q1"], row["dr_hifi_km_q3"])
        lines.append(
            f"  {row['alt_shell']:<5}  {row['generation']:<8}  "
            f"{row['bucket']:<6}  {row['n']:>6}  {sgp4:>25}  {hifi:>25}"
        )
    return "\n".join(lines)


def _fmt_manifest_summary(summary: ManifestSummary) -> str:
    lines = [
        "Manifest failure accounting:",
        "",
        f"  total entries: {summary.n_total:>7}",
        f"  ok:            {summary.n_ok:>7}",
        f"  failed:        {summary.n_failed:>7}",
        f"  skipped:       {summary.n_skipped:>7}",
    ]
    if not summary.failed_stderr_buckets.empty:
        lines.extend(["", "  Failed-run stderr first-line buckets:"])
        for _, row in summary.failed_stderr_buckets.iterrows():
            lines.append(f"    {row['count']:>4}  {row['stderr_first_line']}")
    return "\n".join(lines)


def format_report(
    all_runs: pd.DataFrame,
    manifest_path: Path | None,
) -> str:
    """Build the full text report. `manifest_path=None` skips the failure tally."""
    bucket_stats = compute_bucket_stats(all_runs)
    shell_gen_stats = compute_shell_gen_stats(all_runs)

    sections = [
        f"SWEEP STATS — {len(all_runs)} successful run(s)",
        "=" * 60,
        "",
        _fmt_bucket_stats(bucket_stats),
        "",
        _fmt_shell_gen_stats(shell_gen_stats),
    ]
    if manifest_path is not None:
        sections.extend(["", _fmt_manifest_summary(compute_manifest_summary(manifest_path))])
    return "\n".join(sections) + "\n"


# --- CLI ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-runs",
        type=Path,
        required=True,
        help="Path to outputs/all_runs.parquet (built by sweep.aggregate)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to sweep/manifest.jsonl for the failure tally",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional text file to write the report to (in addition to stdout)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_runs = pd.read_parquet(args.all_runs)
    report = format_report(all_runs, args.manifest)
    sys.stdout.write(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
