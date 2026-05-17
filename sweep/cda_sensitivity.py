"""Drag-area sensitivity sweep on the v2-mini subset (issue #28, v3 Theme E).

Reruns the v2-mini slice of the 1,000-pair stratified sensitivity subset
(from `sweep.sensitivity_subset`) at CdA scaling factors {0.8, 1.2}. The
CdA = 1.0 baseline is reused verbatim from `outputs/all_runs.parquet`;
this driver only computes the two off-baseline frames. Output:

    outputs/all_runs_cda_low.parquet     (CdA factor 0.8 — drag_area × 0.8)
    outputs/all_runs_cda_high.parquet    (CdA factor 1.2 — drag_area × 1.2)

Per-run schema mirrors `outputs/run_<id>.parquet` plus three columns:
`cda_factor` (the multiplier applied), `applied_drag_area_m2` (the value
actually fed to GMAT), and `generation` (always `v2-mini` for these
frames). The baseline column carry from `sweep.aggregate` is bypassed —
the v2-mini-only filter makes the join unnecessary.

The driver mirrors the per-pair preprocessing, RunSpec construction, and
postprocessing of `sweep.run_sweep` so the two sweeps are
bit-comparable: the only deliberate difference per pair is the
`Sat.DragArea` override. Reusing the imported helpers avoids any
silent divergence between this sensitivity arm and the main sweep
hi-fid arm.

Output layout (gitignored, deposited to Zenodo alongside the main sweep
bundle as part of the v0.1.0 release cut):

    outputs/_cda_sensitivity/
        cda_0.8/
            manifest.jsonl
            run_<id>/        # GMAT scratch per run
            run_<id>.parquet # one-row comparison frame
        cda_1.2/
            ...same shape...

Resume is not supported: this is a single-shot diagnostic at ~800 runs,
and the existing manifest is unlinked at start so the run is
deterministic against the current mission script. Re-run from scratch
to refresh.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from gmat_sweep import LocalJoblibPool, Manifest, RunSpec, Sweep

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
DEFAULT_CDA_FACTORS: Final = (0.8, 1.2)
GENERATION_V2_MINI: Final = "v2-mini"

_FACTOR_LABELS: Final = {0.8: "low", 1.2: "high"}


def _scale_drag_area(spec: RunSpec, factor: float) -> RunSpec:
    """Return a new RunSpec with `Sat.DragArea` multiplied by `factor`.

    Every other override is preserved verbatim. `factor == 1.0` is a no-op
    that still returns a fresh RunSpec instance, so the caller can blindly
    map across all factors without special-casing the baseline.
    """
    new_overrides = dict(spec.overrides)
    new_overrides["Sat.DragArea"] = float(spec.overrides["Sat.DragArea"]) * float(factor)
    return RunSpec(
        script_path=spec.script_path,
        overrides=new_overrides,
        output_dir=spec.output_dir,
        run_id=spec.run_id,
        seed=spec.seed,
        run_options=spec.run_options,
    )


def select_v2_mini_subset(corpus: pd.DataFrame, subset_ids_path: Path) -> pd.DataFrame:
    """Load the sensitivity-subset run_ids and filter to v2-mini rows.

    `subset_ids_path` is one run_id per line, sorted ascending (the
    contract from `sweep.sensitivity_subset.write_pair_ids`). The corpus
    must use the same `reset_index(drop=True)` row-index = run_id
    convention as `sweep.run_sweep.main`.
    """
    if not subset_ids_path.exists():
        raise FileNotFoundError(
            f"sensitivity subset id file not found: {subset_ids_path}. "
            f"Build it with `make build-sensitivity-subset`.",
        )
    ids = [int(x) for x in subset_ids_path.read_text().splitlines() if x.strip()]
    corpus = corpus.reset_index(drop=True)
    sub = corpus.loc[ids]
    return sub[sub["generation"] == GENERATION_V2_MINI].copy()


def _postprocess_run(
    pre: _Preprocessed,
    report_path: Path,
    out_path: Path,
    cda_factor: float,
    applied_drag_area_m2: float,
) -> dict:
    """Per-run comparison row. Mirrors `run_sweep._postprocess_run`.

    Additions over the main-sweep schema: `cda_factor`,
    `applied_drag_area_m2`, and `generation` (always `v2-mini` for this
    sensitivity arm). Carries the per-pair pre-computed
    SGP4 prediction and truth state through unchanged — those terms do
    not depend on the CdA factor, so the SGP4-vs-truth Δr column is
    identical to the corresponding baseline row, which is the intent.
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
        "cda_factor": float(cda_factor),
        "applied_drag_area_m2": float(applied_drag_area_m2),
        "generation": GENERATION_V2_MINI,
    }
    pd.DataFrame([row]).to_parquet(out_path, index=False)
    return row


def _aggregate_factor(
    preprocessed: list[_Preprocessed],
    manifest_path: Path,
    factor_dir: Path,
    factor: float,
) -> pd.DataFrame:
    """Walk the manifest, postprocess each ok entry, return aggregated frame."""
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
        out_path = factor_dir / f"run_{pre.run_id}.parquet"
        applied = pre.drag_area_m2 * float(factor)
        rows.append(
            _postprocess_run(pre, Path(report_path_obj), out_path, factor, applied),
        )
    if not rows:
        raise RuntimeError(f"no runs were postprocessed at CdA factor {factor:g}")
    return pd.DataFrame(rows)


def run_one_factor(
    pairs: pd.DataFrame,
    mission: Path,
    sw_lookup: dict,
    factor: float,
    output_root: Path,
    workers: int,
) -> pd.DataFrame:
    """Run the GMAT sweep at a single CdA factor; return the aggregated frame.

    Per-run scratch and the factor-scoped manifest land under
    ``output_root / f"cda_{factor:g}"``. The manifest is unlinked at
    entry so each call is a clean dispatch — there is no resume path
    for this diagnostic.
    """
    factor_dir = (output_root / f"cda_{factor:g}").resolve()
    factor_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = factor_dir / "manifest.jsonl"
    manifest_path.unlink(missing_ok=True)
    mission = mission.resolve()

    print(
        f"=== CdA factor {factor:g}: preprocessing {len(pairs)} v2-mini pair(s) ===",
        file=sys.stderr,
    )
    t0 = time.monotonic()
    preprocessed = _preprocess_all(pairs, sw_lookup)
    print(
        f"    {len(preprocessed)}/{len(pairs)} pair(s) preprocessed in "
        f"{time.monotonic() - t0:.1f} s",
        file=sys.stderr,
    )
    if not preprocessed:
        raise RuntimeError(f"no pairs survived preprocessing at CdA factor {factor:g}")

    base_specs = [_build_run_spec(p, mission, factor_dir) for p in preprocessed]
    scaled_specs = [_scale_drag_area(s, factor) for s in base_specs]

    parameter_spec = {
        "_kind": "explicit",
        "columns": list(_OVERRIDE_COLUMNS),
        "rows": [[s.overrides[c] for c in _OVERRIDE_COLUMNS] for s in scaled_specs],
    }

    print(
        f"=== CdA factor {factor:g}: dispatching {len(scaled_specs)} GMAT run(s), "
        f"{workers} worker(s) ===",
        file=sys.stderr,
    )
    t0 = time.monotonic()
    with LocalJoblibPool(max_workers=workers) as pool:
        Sweep(
            runs=scaled_specs,
            backend=pool,
            manifest_path=manifest_path,
            output_dir=factor_dir,
            script_path=mission,
            parameter_spec=parameter_spec,
            sweep_seed=None,
            progress=True,
        ).run()
    print(f"    sweep finished in {time.monotonic() - t0:.1f} s", file=sys.stderr)

    df = _aggregate_factor(preprocessed, manifest_path, factor_dir, factor)
    print(
        f"=== CdA factor {factor:g}: postprocessed {len(df)} run(s) ===",
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
        "--tles",
        type=Path,
        default=Path("src/static/tles_cache.parquet"),
        help="Path to the cached TLE-pair corpus",
    )
    parser.add_argument(
        "--subset-ids",
        type=Path,
        default=Path("src/static/sensitivity_subset_pair_ids.txt"),
        help="Path to the sensitivity-subset run_id file from sweep.sensitivity_subset",
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
        default=Path("outputs/_cda_sensitivity"),
        help="Directory for per-factor sub-sweep scratch (gitignored)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for the final aggregated all_runs_cda_{low,high}.parquet frames",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--factor",
        type=float,
        nargs="+",
        default=list(DEFAULT_CDA_FACTORS),
        help=(
            "CdA scaling factor(s) to run. Default runs both ±20% factors "
            f"({DEFAULT_CDA_FACTORS}); pass a single value to run just one."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    corpus = pd.read_parquet(args.tles)
    pairs = select_v2_mini_subset(corpus, args.subset_ids)
    if pairs.empty:
        raise SystemExit(
            "no v2-mini pairs in the sensitivity subset; check that "
            "build-sensitivity-subset ran against the current corpus.",
        )
    print(
        f"selected {len(pairs)} v2-mini pair(s) from the sensitivity subset",
        file=sys.stderr,
    )
    print(
        f"  per (alt_shell, target_dt_sec):\n"
        f"{pairs.groupby(['alt_shell', 'target_dt_sec']).size().to_string()}",
        file=sys.stderr,
    )

    sw_lookup: dict = load_sw_cache(args.sw_cache)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    for factor in args.factor:
        df = run_one_factor(
            pairs,
            resolved_mission,
            sw_lookup,
            factor,
            output_root,
            args.workers,
        )
        label = _FACTOR_LABELS.get(float(factor))
        if label is None:
            # Defensive: not one of the two documented factors; persist as
            # cda_<factor> so the file at least lands somewhere readable
            # rather than overwriting one of the headline outputs.
            out_path = args.output_dir / f"all_runs_cda_{factor:g}.parquet"
        else:
            out_path = args.output_dir / f"all_runs_cda_{label}.parquet"
        df.to_parquet(out_path, index=False)
        print(
            f"wrote {out_path}: {len(df)} row(s), {len(df.columns)} col(s)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
