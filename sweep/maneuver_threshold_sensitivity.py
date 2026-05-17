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

import numpy as np
import pandas as pd
from gmat_sweep import LocalJoblibPool, Manifest, Sweep

from sweep.run_sweep import (
    _OVERRIDE_COLUMNS,
    _RESOLVED_SCRIPT_NAME,
    _build_run_spec,
    _decompose_rsw,
    _final_gmat_state,
    _preprocess_all,
    _Preprocessed,
    _resolve_mission_script,
)
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


def _postprocess_run(
    pre: _Preprocessed,
    report_path: Path,
    out_path: Path,
) -> dict:
    """Per-run comparison row in the main-sweep schema.

    Differs from `sweep.run_sweep._postprocess_run` only by living next
    to the augment driver --- the column list is identical, so the
    augment frame concatenates cleanly with the main-sweep frame in the
    downstream table emitter.
    """
    r_hifi_mj, _ = _final_gmat_state(report_path)
    dr_sgp4 = pre.r_sgp4_pred_mj_km - pre.r_truth_mj_km
    dr_hifi = r_hifi_mj - pre.r_truth_mj_km

    sgp4_r, sgp4_a, sgp4_c = _decompose_rsw(dr_sgp4, pre.r_truth_mj_km, pre.v_truth_mj_km_s)
    hifi_r, hifi_a, hifi_c = _decompose_rsw(dr_hifi, pre.r_truth_mj_km, pre.v_truth_mj_km_s)

    row = {
        "run_id": pre.run_id,
        "norad_id": pre.norad_id,
        "target_dt_sec": pre.target_dt_sec,
        "t_i": pre.epoch_i,
        "t_j": pre.epoch_j,
        "actual_dt_sec": pre.actual_dt_sec,
        "alt_shell": pre.alt_shell,
        "dr_sgp4_km": float(np.linalg.norm(dr_sgp4)),
        "dr_sgp4_radial_km": sgp4_r,
        "dr_sgp4_along_km": sgp4_a,
        "dr_sgp4_cross_km": sgp4_c,
        "dr_hifi_km": float(np.linalg.norm(dr_hifi)),
        "dr_hifi_radial_km": hifi_r,
        "dr_hifi_along_km": hifi_a,
        "dr_hifi_cross_km": hifi_c,
        "f107": pre.f107_obs,
        "ap": pre.ap_daily,
    }
    pd.DataFrame([row]).to_parquet(out_path, index=False)
    return row


def _aggregate_augment(
    preprocessed: list[_Preprocessed],
    manifest_path: Path,
    output_root: Path,
    augment_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Walk the manifest, postprocess each ok entry, join corpus columns.

    The join attaches ``CARRIED_COLUMNS`` from the 200 m corpus
    (``generation`` and the spacecraft-property columns) so the augment
    frame matches the joined ``outputs/all_runs.parquet`` schema and the
    table emitter can concat without extra alignment.
    """
    by_run_id = {p.run_id: p for p in preprocessed}
    manifest = Manifest.load(manifest_path)
    rows: list[dict] = []
    for entry in manifest.entries:
        if entry.status != "ok":
            continue
        pre = by_run_id.get(entry.run_id)
        if pre is None:
            print(
                f"  run_id={entry.run_id}: ok in manifest but no preprocess data; skipping",
                file=sys.stderr,
            )
            continue
        report_path_obj = entry.output_paths.get("report__FinalState")
        if report_path_obj is None or not Path(report_path_obj).exists():
            print(
                f"  run_id={entry.run_id}: no FinalState report; skipping postprocess",
                file=sys.stderr,
            )
            continue
        out_path = output_root / f"run_{pre.run_id}.parquet"
        rows.append(_postprocess_run(pre, Path(report_path_obj), out_path))

    if not rows:
        raise RuntimeError("no augment runs were postprocessed")

    frame = pd.DataFrame(rows)
    # Join corpus columns by `run_id`. The augment_pairs index already
    # carries the offset run_id (set by `_assign_augment_run_ids`), so we
    # join on that index against the per-run rows.
    corpus_cols = augment_pairs[list(CARRIED_COLUMNS)].copy()
    corpus_cols["run_id"] = corpus_cols.index.astype(int)
    return frame.merge(corpus_cols, on="run_id", how="left")


def run_augment_sweep(
    augment_pairs: pd.DataFrame,
    mission: Path,
    sw_lookup: dict,
    output_root: Path,
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

    specs = [_build_run_spec(p, mission, output_root) for p in preprocessed]

    parameter_spec = {
        "_kind": "explicit",
        "columns": list(_OVERRIDE_COLUMNS),
        "rows": [[s.overrides[c] for c in _OVERRIDE_COLUMNS] for s in specs],
    }

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
            parameter_spec=parameter_spec,
            sweep_seed=None,
            progress=True,
        ).run()
    print(f"    sweep finished in {time.monotonic() - t0:.1f} s", file=sys.stderr)

    df = _aggregate_augment(preprocessed, manifest_path, output_root, augment_pairs)
    print(
        f"=== Maneuver-threshold augment: postprocessed {len(df)} run(s) ===",
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

    # Same script-templating step as the main sweep — see
    # sweep.run_sweep._resolve_mission_script for the rationale.
    args.mission = args.mission.resolve()
    resolved_mission = args.mission.parent / _RESOLVED_SCRIPT_NAME
    _resolve_mission_script(args.mission, args.gmat_sw_file.resolve(), resolved_mission)
    print(
        f"resolved mission script -> {resolved_mission} "
        f"(SW file baked: {args.gmat_sw_file.resolve()})",
        file=sys.stderr,
    )

    df = run_augment_sweep(
        augment_pairs,
        resolved_mission,
        sw_lookup,
        args.output_root,
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
