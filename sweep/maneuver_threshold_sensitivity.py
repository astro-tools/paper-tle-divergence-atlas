"""Maneuver-filter-threshold sensitivity sweep (issue #31, v3 Theme F).

The baseline 100 m maneuver-detection threshold is empirically calibrated
against the bimodal $|\\Delta a|$ histogram of the full Starlink TLE
archive (Appendix A, Figure F8). Two reviewers flagged that the OD-noise
mode and the maneuver mode overlap in the 50--200 m region, asking for a
quantitative defence of the choice at the threshold endpoints.

This driver underwrites that defence with new GMAT data on the *augment*
set --- pairs that the 200 m filter retains but the 100 m filter
rejects. Together with a filter-only view of the existing main-sweep
``outputs/all_runs.parquet`` (the 50 m result is a strict subset of the
100 m corpus and needs no new propagation), the augment lets the
``sweep.maneuver_threshold_table`` emitter compare per-cell medians at
{50 m, 100 m, 200 m} against the 100 m baseline's sat-level bootstrap CIs.

Pipeline::

    augment_pairs = read(200 m corpus) \\ read(100 m corpus)        # set diff
    augment_pairs.reset_index(drop=True) + AUGMENT_RUN_ID_OFFSET    # fresh run_ids
    pre = _preprocess_all(augment_pairs, sw_lookup)                # TEME -> MJ2000Eq
    specs = [_build_run_spec(p, mission, scratch) for p in pre]    # GMAT RunSpecs
    Sweep(specs).run()                                             # 8-worker parallel
    rows = [_postprocess_run(pre, report) for entry in manifest]   # per-run frames
    aggregate + join corpus columns                                # main-sweep schema
    -> outputs/all_runs_maneuver_augment.parquet

The augment frame's per-row schema matches the joined
``outputs/all_runs.parquet`` produced by ``sweep.aggregate``: the 17
``_postprocess_run`` columns plus the ``CARRIED_COLUMNS`` (``generation``,
``dry_mass_kg``, ``drag_area_m2``, ``srp_area_m2``) from the 200 m
corpus, so the downstream table emitter can concat the baseline and
augment frames without extra alignment.

Run IDs are offset by ``AUGMENT_RUN_ID_OFFSET`` (1{,}000{,}000) to
guarantee they never collide with main-sweep IDs (0..24{,}640) when the
two frames are concatenated.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Final

import pandas as pd
from gmat_sweep import LocalJoblibPool, Sweep, lazy_extra_outputs

from sweep.run_sweep import _build_run_spec, _preprocess_all
from sweep.space_weather import load_sw_cache

DEFAULT_WORKERS: Final = 8
DEFAULT_BASELINE_THRESHOLD_KM: Final = 0.1
DEFAULT_AUGMENT_THRESHOLD_KM: Final = 0.2

PAIR_KEY: Final = ("norad_id", "epoch_i", "epoch_j")
CARRIED_COLUMNS: Final = ("generation", "dry_mass_kg", "drag_area_m2", "srp_area_m2")
AUGMENT_RUN_ID_OFFSET: Final = 1_000_000


def compute_augment(
    baseline_corpus: pd.DataFrame,
    high_threshold_corpus: pd.DataFrame,
) -> pd.DataFrame:
    """Pairs in `high_threshold_corpus` that are not in `baseline_corpus`.

    Matched by ``(norad_id, epoch_i, epoch_j)``. The two corpora must
    have been built from the same raw TLE cache and the same sat sample
    (i.e., identical ``--per-shell`` / ``--seed`` arguments) for the set
    difference to be meaningful. ``epoch_i`` and ``epoch_j`` are
    tz-aware Timestamps; matching is exact, not within tolerance.

    The augment is the pair set the 100 m filter rejects but the 200 m
    filter retains --- i.e., pairs whose ``(t_i, t_j]`` interval covers
    an SMA jump in the 100--200 m bin. These pairs are the ones the
    Appendix B "Maneuver-threshold sensitivity" subsection needs new
    GMAT data for, since the 50 m corpus is a strict subset of the
    100 m corpus and the 100 m corpus is a strict subset of the 200 m
    corpus (inclusion verified empirically at corpus-build time).
    """
    keys = list(PAIR_KEY)
    baseline_set = set(map(tuple, baseline_corpus[keys].itertuples(index=False, name=None)))
    high_keys = high_threshold_corpus[keys].apply(tuple, axis=1)
    augment_mask = ~high_keys.isin(baseline_set)
    return high_threshold_corpus[augment_mask].copy()


def _assign_augment_run_ids(augment_pairs: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic, offset run_ids to the augment pairs.

    The augment set is sorted by ``(norad_id, epoch_i, target_dt_sec)``
    so that the run_id is stable across reruns of this driver --- a
    reviewer regenerating the augment frame against the same corpora
    will see exactly the same run_ids and the same outputs.

    ``_preprocess_all`` reads the run_id from the DataFrame's row index
    via ``iterrows()``; we set that index to the offset value so the
    augment IDs are 1_000_000..1_000_000+N-1 in sort order.
    """
    sort_cols = ["norad_id", "epoch_i", "target_dt_sec"]
    sorted_pairs = augment_pairs.sort_values(sort_cols).reset_index(drop=True)
    sorted_pairs.index = pd.RangeIndex(
        AUGMENT_RUN_ID_OFFSET,
        AUGMENT_RUN_ID_OFFSET + len(sorted_pairs),
    )
    return sorted_pairs


def _aggregate_augment(
    manifest_path: Path,
    augment_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Stream every ok manifest entry's comparison output and join corpus columns.

    The per-run comparison schema is produced by the main sweep's
    postprocess hook (`sweep.run_sweep:_postprocess_run`) — there is
    no augment-specific hook, since the augment frame's per-run schema
    is identical to the main sweep's. The join attaches
    ``CARRIED_COLUMNS`` from the 200 m augment corpus (``generation``
    and the spacecraft-property columns) so the augment frame matches
    the joined ``outputs/all_runs.parquet`` schema and the table
    emitter can concat without extra alignment.

    The augment_pairs index already carries the offset run_id (set by
    `_assign_augment_run_ids`), so we lift it into a column for the merge.
    """
    frame = lazy_extra_outputs(manifest_path, "comparison")
    ok = frame[frame["__status"] == "ok"].drop(columns="__status")
    if ok.empty:
        raise RuntimeError("no augment runs were postprocessed")
    ok = ok.reset_index()

    corpus_cols = augment_pairs[list(CARRIED_COLUMNS)].copy()
    corpus_cols["run_id"] = corpus_cols.index.astype(int)
    return ok.merge(corpus_cols, on="run_id", how="left")


def run_augment_sweep(
    augment_pairs: pd.DataFrame,
    mission: Path,
    sw_lookup: dict,
    output_root: Path,
    gmat_sw_file: Path,
    workers: int,
) -> pd.DataFrame:
    """Preprocess, dispatch GMAT, postprocess, and aggregate the augment set.

    Pure orchestration on top of the imported ``run_sweep`` helpers. The
    manifest is unlinked at entry so each call is a clean dispatch;
    there is no resume path for this single-shot diagnostic. The scratch
    dir conventionally lives at ``outputs/_maneuver_threshold_sensitivity/``.
    """
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    manifest_path.unlink(missing_ok=True)
    mission = mission.resolve()

    print(
        f"=== Maneuver-threshold augment: preprocessing {len(augment_pairs)} pair(s) ===",
        file=sys.stderr,
    )
    t0 = time.monotonic()
    preprocessed = _preprocess_all(augment_pairs, sw_lookup)
    print(
        f"    {len(preprocessed)}/{len(augment_pairs)} pair(s) preprocessed in "
        f"{time.monotonic() - t0:.1f} s",
        file=sys.stderr,
    )
    if not preprocessed:
        raise RuntimeError("no pairs survived preprocessing")

    specs = [_build_run_spec(p, mission, output_root, gmat_sw_file) for p in preprocessed]

    print(
        f"=== Maneuver-threshold augment: dispatching {len(specs)} GMAT run(s), "
        f"{workers} worker(s) ===",
        file=sys.stderr,
    )
    t0 = time.monotonic()
    with LocalJoblibPool(max_workers=workers) as pool:
        Sweep(
            runs=specs,
            backend=pool,
            manifest_path=manifest_path,
            output_dir=output_root,
            script_path=mission,
            sweep_seed=None,
            progress=True,
        ).run()
    print(f"    sweep finished in {time.monotonic() - t0:.1f} s", file=sys.stderr)

    df = _aggregate_augment(manifest_path, augment_pairs)
    print(
        f"=== Maneuver-threshold augment: aggregated {len(df)} run(s) ===",
        file=sys.stderr,
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        type=Path,
        default=Path("sweep/mission.script"),
        help="Path to the GMAT mission .script (same script the main sweep uses)",
    )
    parser.add_argument(
        "--baseline-corpus",
        type=Path,
        default=Path("src/static/tles_cache.parquet"),
        help="Path to the 100 m baseline corpus (main sweep's input).",
    )
    parser.add_argument(
        "--augment-corpus",
        type=Path,
        default=Path("outputs/tles_cache_200m.parquet"),
        help="Path to the 200 m corpus from which augment pairs are extracted.",
    )
    parser.add_argument(
        "--sw-cache",
        type=Path,
        default=Path("src/static/sw_cache.parquet"),
        help="Path to space-weather cache parquet (built by `make fetch-sw`)",
    )
    parser.add_argument(
        "--gmat-sw-file",
        type=Path,
        default=Path("src/static/SpaceWeather-All-v1.2.txt"),
        help="Path to the CssiSpaceWeather text file passed to GMAT via "
        "FM.Drag.CSSISpaceWeatherFile.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/_maneuver_threshold_sensitivity"),
        help="Directory for per-run GMAT scratch and the augment manifest "
        "(gitignored under outputs/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/all_runs_maneuver_augment.parquet"),
        help="Output Parquet path for the aggregated augment frame",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If given, only sweep the first N augment pairs (smoke test).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    baseline = pd.read_parquet(args.baseline_corpus)
    augment_corpus = pd.read_parquet(args.augment_corpus)
    augment_pairs = compute_augment(baseline, augment_corpus)
    if augment_pairs.empty:
        raise SystemExit(
            "no augment pairs: 200 m corpus does not exceed 100 m corpus. "
            "Did the two corpora get built at the same threshold?",
        )
    augment_pairs = _assign_augment_run_ids(augment_pairs)
    if args.limit is not None and args.limit > 0:
        augment_pairs = augment_pairs.iloc[: args.limit]
        print(f"--limit {args.limit}: smoke-truncated augment", file=sys.stderr)
    print(
        f"computed augment: {len(augment_pairs)} pair(s) in 200 m corpus but not 100 m corpus",
        file=sys.stderr,
    )
    print(
        "  per (alt_shell, target_dt_sec, generation):\n"
        f"{augment_pairs.groupby(['alt_shell', 'target_dt_sec', 'generation']).size().to_string()}",
        file=sys.stderr,
    )

    sw_lookup = load_sw_cache(args.sw_cache)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    args.mission = args.mission.resolve()
    gmat_sw_file = args.gmat_sw_file.resolve()

    df = run_augment_sweep(
        augment_pairs,
        args.mission,
        sw_lookup,
        args.output_root,
        gmat_sw_file,
        args.workers,
    )
    df.to_parquet(args.out, index=False)
    print(
        f"wrote {args.out}: {len(df)} row(s), {len(df.columns)} col(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
