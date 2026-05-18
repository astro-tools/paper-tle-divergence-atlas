"""Build the canonical Zenodo deposit bundle.

The canonical bundle is what gets archived at the version DOI for each
tagged release: the aggregated sweep outputs and the small mission-
script / installer / manifest payload a reader needs to regenerate
figures from the cached parquets, plus to re-run the sweep from
scratch if they have GMAT. Per-run directories (`outputs/run_*`,
`outputs/_cda_sensitivity/`, `outputs/_maneuver_threshold_sensitivity/`,
`outputs/_truth_floor/run_*`) are intermediate and not bundled here; if
a raw-run consumer ever needs them, they live in a separate "raw"
deposit.

Layout inside the zip is flat (every entry at the archive root) to
match the showyourwork `datasets:` `contents:` mapping that fetches by
basename. Missing files are a hard error — Zenodo deposits are
immutable per version, so an incomplete bundle is not recoverable
after the fact.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Final

# Canonical bundle file list (paths relative to repo root). Update this
# list when the deposit composition changes; everything else in the
# module is generic.
BUNDLE_FILES: Final[tuple[str, ...]] = (
    "outputs/all_runs.parquet",
    "outputs/all_runs_cda_low.parquet",
    "outputs/all_runs_cda_high.parquet",
    "outputs/all_runs_maneuver_augment.parquet",
    "outputs/tles_cache_50m.parquet",
    "outputs/tles_cache_200m.parquet",
    "outputs/propagator_wins.json",
    "outputs/h3_regression.json",
    "outputs/cda_sensitivity_summary.json",
    "outputs/maneuver_threshold_summary.json",
    "outputs/mixed_effects_results.csv",
    "outputs/sweep_stats.txt",
    "outputs/_truth_floor.parquet",
    "outputs/_truth_floor.json",
    "sweep/manifest.jsonl",
    "sweep/mission.script",
    "sweep/install_egm2008.py",
)


def build_bundle(
    repo_root: Path,
    out_path: Path,
    files: tuple[str, ...] = BUNDLE_FILES,
) -> Path:
    """Zip `files` under `repo_root` into `out_path` with a flat layout.

    Each entry is added under its basename — no subdirectories inside
    the archive. Refuses to write if any source file is missing.
    """
    resolved = [(rel, repo_root / rel) for rel in files]
    missing = [rel for rel, src in resolved if not src.is_file()]
    if missing:
        listing = "\n  ".join(missing)
        raise FileNotFoundError(f"bundle source file(s) missing under {repo_root}:\n  {listing}")

    basenames = [src.name for _, src in resolved]
    duplicates = sorted({n for n in basenames if basenames.count(n) > 1})
    if duplicates:
        # A flat layout collides if two source paths share a basename.
        # Surface it as a hard error rather than silently letting one
        # entry overwrite the other in the archive.
        raise ValueError(f"flat-layout bundle would collide on basenames: {duplicates}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for _, src in resolved:
            zf.write(src, arcname=src.name)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of sweep/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output zip path (defaults to <repo-root>/bundle.zip).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = args.out if args.out is not None else args.repo_root / "bundle.zip"
    try:
        written = build_bundle(args.repo_root, out_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    size_mb = written.stat().st_size / (1024 * 1024)
    print(f"wrote {written} ({size_mb:.1f} MB, {len(BUNDLE_FILES)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
