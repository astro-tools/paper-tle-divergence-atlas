"""Run the full TLE-divergence-atlas sweep.

Drives gmat-sweep over the corpus of Starlink TLE pairs. Each run propagates
a satellite forward from t_i to t_j using both SGP4 (from TLE_i) and GMAT
high-fidelity force models, then compares both predictions against the
operator's next-TLE truth (SGP4(TLE_j, Δt=0)).

Pipeline (two phases, single process for the driver):

    1. Preprocess each pair (Python, no GMAT):
       - Evaluate SGP4(TLE_i, 0) and SGP4(TLE_i, dt) in TEME.
       - Evaluate SGP4(TLE_j, 0) — the operator's next-TLE truth — in TEME.
       - Rotate every state TEME → MJ2000Eq via Vallado's IAU 1976/1980
         procedure (precession × nutation × equation-of-equinox, composed
         from `erfa` primitives), so the state we hand to GMAT lives in
         the same inertial frame GMAT propagates in.
       - Stash the per-pair payload the postprocess hook needs
         (truth state, SGP4 prediction, SW values, identifiers) on
         each `RunSpec.context`.

    2. GMAT sweep: build one RunSpec per pair with field overrides
       (`Sat.Epoch`, `Sat.X..VZ`, `elapsed_seconds.Value`,
       `FM.Drag.CSSISpaceWeatherFile`) and dispatch through
       `gmat_sweep.Sweep` over a `LocalJoblibPool` with the per-run
       postprocess hook `sweep.run_sweep:_postprocess_run` registered.
       Each worker writes its ReportFile to
       `outputs/run-<id>/report__FinalState.parquet`, then the hook
       reads it, computes Δr_hifi and Δr_sgp4 against truth, decomposes
       both into radial/along/cross relative to the truth state's RSW
       frame, and emits `outputs/run-<id>/comparison.parquet` — a
       one-row comparison frame recorded as the manifest entry's
       `extra_outputs.comparison`. `sweep.aggregate` stitches them into
       `outputs/all_runs.parquet` via `gmat_sweep.lazy_extra_outputs`.

`f107` (daily observed F10.7, sfu) and `ap` (planetary daily Ap) are
joined from `src/static/sw_cache.parquet` by `date(epoch_i)` UTC. The
cache is built by `sweep.space_weather`; a missing date raises rather
than silently NaN-ing so a corpus window extension that outruns the
cached SW window fails loudly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import erfa
import numpy as np
import pandas as pd
from astropy.time import Time
from gmat_sweep import LocalJoblibPool, Manifest, RunOutcome, RunSpec, Sweep
from sgp4.api import Satrec, jday

from sweep.space_weather import (
    SwRow,
    load_sw_cache,
    lookup_for_epoch,
    verify_gmat_sw_coverage,
)
from sweep.spacecraft_props import CD, CR

DEFAULT_SMOKE_N: Final = 8  # split evenly across the Δt buckets
DEFAULT_SMOKE_SEED: Final = 42
DEFAULT_WORKERS: Final = 8

# --- TEME → MJ2000Eq rotation (Vallado IAU 1976/1980, via erfa) -----------


def _teme_to_mj2000_matrix(epoch: pd.Timestamp) -> np.ndarray:
    """3×3 rotation matrix r_MJ2000 = M @ r_TEME at `epoch`.

    Composed from erfa primitives following Vallado Algorithm 24:
    TEME → TOD (equation of equinoxes), TOD → MOD (IAU 1980 nutation),
    MOD → J2000 (IAU 1976 precession). All evaluated at TT split from
    the UTC `epoch` via astropy's IERS-aware time conversion.

    The same matrix is used for the velocity rotation. The frame-rate
    term dM/dt (precession ≲ 50″/yr, nutation ≲ 9″ amplitude) yields a
    velocity-component correction ≲ 1e-5 mm/s at LEO — negligible vs.
    the km-scale Δr the paper is characterising.
    """
    t = Time(epoch.tz_convert("UTC").to_pydatetime(), scale="utc")
    jd_tt1, jd_tt2 = t.tt.jd1, t.tt.jd2

    precession = erfa.pmat76(jd_tt1, jd_tt2)  # MJ2000Eq → MOD
    dpsi, deps = erfa.nut80(jd_tt1, jd_tt2)
    obliquity = erfa.obl80(jd_tt1, jd_tt2)
    nutation = erfa.numat(obliquity, dpsi, deps)  # MOD → TOD
    eq_equinox = erfa.eqeq94(jd_tt1, jd_tt2)
    cz, sz = np.cos(eq_equinox), np.sin(eq_equinox)
    r_z = np.array([[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]])  # TOD → TEME

    return precession.T @ nutation.T @ r_z.T


def _teme_to_mj2000(
    r_teme_km: np.ndarray,
    v_teme_km_s: np.ndarray,
    epoch: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _teme_to_mj2000_matrix(epoch)
    return matrix @ r_teme_km, matrix @ v_teme_km_s


# --- SGP4 helpers ----------------------------------------------------------


def _jd_fr_from_epoch(epoch: pd.Timestamp) -> tuple[float, float]:
    e = epoch.tz_convert("UTC")
    return jday(e.year, e.month, e.day, e.hour, e.minute, e.second + e.microsecond / 1e6)


def _sgp4_state_teme(line1: str, line2: str, jd: float, fr: float) -> tuple[np.ndarray, np.ndarray]:
    sat = Satrec.twoline2rv(line1, line2)
    err, r, v = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error code {err}")
    return np.asarray(r, dtype=float), np.asarray(v, dtype=float)


# --- Along/cross/radial decomposition --------------------------------------


def _decompose_rsw(
    delta_km: np.ndarray,
    r_truth_km: np.ndarray,
    v_truth_km_s: np.ndarray,
) -> tuple[float, float, float]:
    """Decompose `delta` in the truth state's RSW frame.

    Returns ``(radial, along, cross)`` in km. Basis:
        e_r = r̂_truth
        e_h = (r × v)̂  — orbital angular momentum direction (cross-track)
        e_t = e_h × e_r  — completes the right-handed triad (along-track,
                           ≈ velocity direction for a near-circular orbit)
    """
    e_r = r_truth_km / np.linalg.norm(r_truth_km)
    angular_momentum = np.cross(r_truth_km, v_truth_km_s)
    e_h = angular_momentum / np.linalg.norm(angular_momentum)
    e_t = np.cross(e_h, e_r)
    return float(e_r @ delta_km), float(e_t @ delta_km), float(e_h @ delta_km)


# --- Per-pair preprocessing ------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Preprocessed:
    """Everything we need per pair before GMAT runs and after GMAT finishes."""

    run_id: int
    norad_id: int
    target_dt_sec: int
    epoch_i: pd.Timestamp
    epoch_j: pd.Timestamp
    actual_dt_sec: float
    alt_shell: str
    r_init_mj_km: np.ndarray
    v_init_mj_km_s: np.ndarray
    r_sgp4_pred_mj_km: np.ndarray
    r_truth_mj_km: np.ndarray
    v_truth_mj_km_s: np.ndarray
    dry_mass_kg: float
    drag_area_m2: float
    srp_area_m2: float
    f107_obs: float
    ap_daily: float


def _preprocess_pair(
    run_id: int,
    pair: pd.Series,
    sw_lookup: dict[dt.date, SwRow],
) -> _Preprocessed:
    jd_i, fr_i = _jd_fr_from_epoch(pair["epoch_i"])
    r_init_teme, v_init_teme = _sgp4_state_teme(pair["line1_i"], pair["line2_i"], jd_i, fr_i)
    r_init_mj, v_init_mj = _teme_to_mj2000(r_init_teme, v_init_teme, pair["epoch_i"])

    jd_j, fr_j = _jd_fr_from_epoch(pair["epoch_j"])
    r_pred_teme, v_pred_teme = _sgp4_state_teme(pair["line1_i"], pair["line2_i"], jd_j, fr_j)
    r_pred_mj, _ = _teme_to_mj2000(r_pred_teme, v_pred_teme, pair["epoch_j"])

    r_truth_teme, v_truth_teme = _sgp4_state_teme(pair["line1_j"], pair["line2_j"], jd_j, fr_j)
    r_truth_mj, v_truth_mj = _teme_to_mj2000(r_truth_teme, v_truth_teme, pair["epoch_j"])

    sw = lookup_for_epoch(sw_lookup, pair["epoch_i"])

    return _Preprocessed(
        run_id=run_id,
        norad_id=int(pair["norad_id"]),
        target_dt_sec=int(pair["target_dt_sec"]),
        epoch_i=pair["epoch_i"],
        epoch_j=pair["epoch_j"],
        actual_dt_sec=float(pair["actual_dt_sec"]),
        alt_shell=str(pair["alt_shell"]),
        r_init_mj_km=r_init_mj,
        v_init_mj_km_s=v_init_mj,
        r_sgp4_pred_mj_km=r_pred_mj,
        r_truth_mj_km=r_truth_mj,
        v_truth_mj_km_s=v_truth_mj,
        dry_mass_kg=float(pair["dry_mass_kg"]),
        drag_area_m2=float(pair["drag_area_m2"]),
        srp_area_m2=float(pair["srp_area_m2"]),
        f107_obs=sw.f107_obs,
        ap_daily=sw.ap_daily,
    )


# --- RunSpec build ---------------------------------------------------------


POSTPROCESS_HOOK: Final = "sweep.run_sweep:_postprocess_run"


# JSON-safe per-pair payload the postprocess hook reads back off
# RunSpec.context. The GMAT-input fields (initial state, masses, areas)
# live only in RunSpec.overrides — no duplication. Numpy arrays become
# lists; tz-aware Timestamps become ISO-8601 strings so the dict round-trips
# through the manifest's JSON encoding cleanly.
def _gmat_epoch_string(epoch: pd.Timestamp) -> str:
    """Format as GMAT UTCGregorian: '01 Apr 2026 12:34:56.789'."""
    e = epoch.tz_convert("UTC")
    return f"{e.strftime('%d %b %Y %H:%M:%S')}.{e.microsecond // 1000:03d}"


def _context_from_preprocessed(pre: _Preprocessed) -> dict[str, object]:
    return {
        "run_id": int(pre.run_id),
        "norad_id": int(pre.norad_id),
        "target_dt_sec": int(pre.target_dt_sec),
        "epoch_i": pre.epoch_i.isoformat(),
        "epoch_j": pre.epoch_j.isoformat(),
        "actual_dt_sec": float(pre.actual_dt_sec),
        "alt_shell": str(pre.alt_shell),
        "r_sgp4_pred_mj_km": [float(x) for x in pre.r_sgp4_pred_mj_km],
        "r_truth_mj_km": [float(x) for x in pre.r_truth_mj_km],
        "v_truth_mj_km_s": [float(x) for x in pre.v_truth_mj_km_s],
        "f107": float(pre.f107_obs),
        "ap": float(pre.ap_daily),
    }


def _build_run_spec(
    pre: _Preprocessed,
    mission_path: Path,
    output_root: Path,
    gmat_sw_file: Path,
) -> RunSpec:
    return RunSpec(
        script_path=mission_path,
        overrides={
            "Sat.Epoch": _gmat_epoch_string(pre.epoch_i),
            "Sat.X": float(pre.r_init_mj_km[0]),
            "Sat.Y": float(pre.r_init_mj_km[1]),
            "Sat.Z": float(pre.r_init_mj_km[2]),
            "Sat.VX": float(pre.v_init_mj_km_s[0]),
            "Sat.VY": float(pre.v_init_mj_km_s[1]),
            "Sat.VZ": float(pre.v_init_mj_km_s[2]),
            "Sat.DryMass": float(pre.dry_mass_kg),
            "Sat.Cd": float(CD),
            "Sat.DragArea": float(pre.drag_area_m2),
            "Sat.Cr": float(CR),
            "Sat.SRPArea": float(pre.srp_area_m2),
            "elapsed_seconds.Value": float(pre.actual_dt_sec),
            "FM.Drag.CSSISpaceWeatherFile": str(gmat_sw_file),
        },
        output_dir=output_root / f"run-{pre.run_id}",
        run_id=pre.run_id,
        seed=None,
        run_options={},
        postprocess=POSTPROCESS_HOOK,
        context=_context_from_preprocessed(pre),
    )


# --- Per-run postprocess ---------------------------------------------------


def _final_gmat_state(report_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(report_path)
    if "time" in df.columns:
        df = df.sort_values("time")
    last = df.iloc[-1]
    r = np.array([last["Sat.X"], last["Sat.Y"], last["Sat.Z"]], dtype=float)
    v = np.array([last["Sat.VX"], last["Sat.VY"], last["Sat.VZ"]], dtype=float)
    return r, v


def _postprocess_run(spec: RunSpec, outcome: RunOutcome) -> dict[str, Path]:
    """gmat-sweep postprocess hook: per-run SGP4-vs-hifi-vs-truth comparison.

    Reads the GMAT FinalState report parquet the worker produced, joins
    against the per-pair payload stashed on ``spec.context`` (truth state,
    SGP4 prediction, identifiers, SW values), decomposes both deltas in
    the truth state's RSW frame, and writes a one-row Parquet next to
    the worker's other artefacts. The returned ``{"comparison": path}``
    becomes the manifest entry's ``extra_outputs.comparison`` and feeds
    :func:`sweep.aggregate.aggregate` via
    :func:`gmat_sweep.lazy_extra_outputs`.

    Module-level / importable: the worker re-imports this in each
    subprocess via ``spec.postprocess`` (``sweep.run_sweep:_postprocess_run``).
    """
    report_path = Path(outcome.output_paths["report__FinalState"])
    r_hifi_mj, _ = _final_gmat_state(report_path)

    ctx = spec.context
    r_sgp4_pred = np.asarray(ctx["r_sgp4_pred_mj_km"], dtype=float)
    r_truth = np.asarray(ctx["r_truth_mj_km"], dtype=float)
    v_truth = np.asarray(ctx["v_truth_mj_km_s"], dtype=float)

    dr_sgp4 = r_sgp4_pred - r_truth
    dr_hifi = r_hifi_mj - r_truth

    sgp4_r, sgp4_a, sgp4_c = _decompose_rsw(dr_sgp4, r_truth, v_truth)
    hifi_r, hifi_a, hifi_c = _decompose_rsw(dr_hifi, r_truth, v_truth)

    row = {
        "run_id": int(ctx["run_id"]),
        "norad_id": int(ctx["norad_id"]),
        "target_dt_sec": int(ctx["target_dt_sec"]),
        "t_i": pd.Timestamp(ctx["epoch_i"]),
        "t_j": pd.Timestamp(ctx["epoch_j"]),
        "actual_dt_sec": float(ctx["actual_dt_sec"]),
        "alt_shell": str(ctx["alt_shell"]),
        "dr_sgp4_km": float(np.linalg.norm(dr_sgp4)),
        "dr_sgp4_radial_km": sgp4_r,
        "dr_sgp4_along_km": sgp4_a,
        "dr_sgp4_cross_km": sgp4_c,
        "dr_hifi_km": float(np.linalg.norm(dr_hifi)),
        "dr_hifi_radial_km": hifi_r,
        "dr_hifi_along_km": hifi_a,
        "dr_hifi_cross_km": hifi_c,
        "f107": float(ctx["f107"]),
        "ap": float(ctx["ap"]),
    }
    out_path = spec.output_dir / "comparison.parquet"
    pd.DataFrame([row]).to_parquet(out_path, index=False)
    return {"comparison": out_path}


# --- CLI / driver ----------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        type=Path,
        required=True,
        help="Path to the GMAT mission .script",
    )
    parser.add_argument(
        "--tles",
        type=Path,
        required=True,
        help="Path to the cached TLE-pair Parquet (built by sweep/tle_pipeline.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-run subdirs (GMAT reports) and the per-run "
        "comparison Parquets (run_<id>.parquet)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to write the reproducibility manifest (JSONL)",
    )
    parser.add_argument(
        "--smoke",
        type=int,
        nargs="?",
        const=DEFAULT_SMOKE_N,
        default=None,
        metavar="N",
        help=f"Run a stratified-random subset of ~N pairs instead of the full "
        f"corpus (default {DEFAULT_SMOKE_N} when --smoke given alone). N is "
        f"split evenly across the Δt buckets (N // n_buckets per bucket; "
        f"remainder truncates), seed=42 for reproducibility.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel workers (joblib backend)",
    )
    parser.add_argument(
        "--sw-cache",
        type=Path,
        required=True,
        help="Path to space-weather cache parquet (built by `make fetch-sw`)",
    )
    parser.add_argument(
        "--gmat-sw-file",
        type=Path,
        required=True,
        help="Path to the CssiSpaceWeather text file passed to GMAT via "
        "FM.Drag.CSSISpaceWeatherFile (committed at "
        "src/static/SpaceWeather-All-v1.2.txt; refresh with `make fetch-gmat-sw`)",
    )
    parser.add_argument(
        "--window",
        type=Path,
        default=Path("src/static/window.json"),
        help="Path to the corpus window JSON used to verify the GMAT SW file's "
        "observed-data horizon covers the analysis epochs.",
    )
    return parser.parse_args()


def _preprocess_all(
    pairs: pd.DataFrame,
    sw_lookup: dict[dt.date, SwRow],
) -> list[_Preprocessed]:
    out: list[_Preprocessed] = []
    for run_id, pair in pairs.iterrows():
        try:
            out.append(_preprocess_pair(int(run_id), pair, sw_lookup))
        except RuntimeError as exc:
            print(
                f"  skip run_id={run_id} sat={pair['norad_id']}: {exc}",
                file=sys.stderr,
            )
    return out


def _resume_compatible(manifest_path: Path, n_corpus_pairs: int) -> bool:
    """True if an existing manifest matches the current corpus and we should resume.

    Refuses to resume when the manifest's `run_count` differs from the
    corpus pair count — typically a smoke→sweep transition (8 vs 24,641)
    or a corpus rebuild between sessions. Caller is expected to clear
    the manifest in that case.
    """
    if not manifest_path.exists():
        return False
    manifest = Manifest.load(manifest_path)
    if manifest.run_count != n_corpus_pairs:
        raise SystemExit(
            f"existing manifest at {manifest_path} has run_count={manifest.run_count} "
            f"but the corpus has {n_corpus_pairs} pair(s). The manifest is from a "
            f"different sweep configuration. Delete it (`rm {manifest_path}`) to "
            f"start a fresh sweep, or point --tles at the corpus the manifest came "
            f"from to resume."
        )
    return True


def _make_context_provider(
    pairs: pd.DataFrame,
    sw_lookup: dict[dt.date, SwRow],
) -> object:
    """Lazy preprocess for runs the manifest forgot.

    `Sweep.from_manifest` restores each entry's `context` from disk — so a
    resumed sweep does not re-preprocess pairs whose entries already
    landed. The only run_ids without a stored context are the ones that
    never produced a manifest entry at all (a cataclysmic mid-dispatch
    interrupt before the first worker wrote). For those, gmat-sweep calls
    this callback with the run's integer `run_id`; we look the pair up by
    its corpus row index and run `_preprocess_pair` on demand.
    """
    pairs_by_run_id = pairs.set_index(pairs.index.astype(int))

    def provider(run_id: int) -> dict[str, object]:
        pair = pairs_by_run_id.loc[run_id]
        pre = _preprocess_pair(int(run_id), pair, sw_lookup)
        return _context_from_preprocessed(pre)

    return provider


def _dispatch_sweep(
    preprocessed: list[_Preprocessed],
    mission: Path,
    output_dir: Path,
    manifest_path: Path,
    gmat_sw_file: Path,
    workers: int,
    resume: bool,
    context_provider: object | None = None,
) -> None:
    run_specs = [_build_run_spec(p, mission, output_dir, gmat_sw_file) for p in preprocessed]
    with LocalJoblibPool(max_workers=workers) as pool:
        if resume:
            Sweep.from_manifest(
                manifest_path,
                mission,
                backend=pool,
                output_dir=output_dir,
                context_provider=context_provider,
                progress=True,
            ).resume()
        else:
            Sweep(
                runs=run_specs,
                backend=pool,
                manifest_path=manifest_path,
                output_dir=output_dir,
                script_path=mission,
                sweep_seed=None,
                postprocess=POSTPROCESS_HOOK,
                progress=True,
            ).run()


def _summarise_manifest(manifest_path: Path) -> tuple[int, int]:
    """Count ok-with-comparison-output vs failed/missing for the end-of-run summary."""
    manifest = Manifest.load(manifest_path)
    ok = 0
    failed = 0
    for entry in manifest.entries:
        if entry.status == "ok" and "comparison" in entry.extra_outputs:
            ok += 1
        else:
            failed += 1
    return ok, failed


def main() -> int:
    args = parse_args()

    pairs = pd.read_parquet(args.tles).reset_index(drop=True)
    if args.smoke is not None:
        # Stratified-random by Δt bucket so every horizon (6 h / 1 d / 3 d /
        # 7 d) is exercised; sats and altitude shells fall out randomly within
        # each stratum so a sat-loop or shell-specific bug surfaces here
        # rather than escaping to the full sweep. Seeded for reproducibility.
        n_buckets = pairs["target_dt_sec"].nunique()
        per_bucket = max(1, args.smoke // n_buckets)
        pairs = (
            pairs.groupby("target_dt_sec", group_keys=False)
            .sample(n=per_bucket, random_state=DEFAULT_SMOKE_SEED)
            .reset_index(drop=True)
        )
    print(f"loaded {len(pairs)} pair(s) from {args.tles}", file=sys.stderr)

    sw_lookup = load_sw_cache(args.sw_cache)
    print(
        f"loaded {len(sw_lookup)} space-weather day(s) from {args.sw_cache}",
        file=sys.stderr,
    )

    # GMAT resolves a relative working_dir against its installed OUTPUT_PATH
    # (e.g. /home/.../gmat-R2026a/output/), producing paths like
    # /home/.../gmat-R2026a/output/outputs/run_0/. Absolutise so per-run
    # files land where the caller pointed.
    args.output_dir = args.output_dir.resolve()
    args.manifest = args.manifest.resolve()
    args.mission = args.mission.resolve()
    args.tles = args.tles.resolve()
    args.gmat_sw_file = args.gmat_sw_file.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    window_end = dt.datetime.fromisoformat(json.loads(args.window.read_text())["end"]).date()
    end = verify_gmat_sw_coverage(args.gmat_sw_file, window_end)
    print(
        f"GMAT space-weather file {args.gmat_sw_file} covers observed through "
        f"{end.isoformat()} (corpus window ends {window_end.isoformat()})",
        file=sys.stderr,
    )

    # Resume if an existing manifest matches the current corpus; otherwise
    # fresh dispatch. The fresh path preprocesses every pair and dispatches
    # through Sweep(...).run(); the resume path delegates to
    # Sweep.from_manifest(...).resume(), which restores each entry's
    # `context` from the manifest and re-dispatches only the failed and
    # missing runs — no driver-side preprocess needed.
    resume = _resume_compatible(args.manifest, len(pairs))
    if resume:
        print(f"existing manifest at {args.manifest}; resuming sweep", file=sys.stderr)

    preprocessed: list[_Preprocessed]
    context_provider: object | None
    if resume:
        # Provider runs only for run_ids whose entry never landed — see
        # `_make_context_provider`. The common case is zero invocations.
        preprocessed = []
        context_provider = _make_context_provider(pairs, sw_lookup)
    else:
        print("phase 1/2: preprocess", file=sys.stderr)
        t0 = time.monotonic()
        preprocessed = _preprocess_all(pairs, sw_lookup)
        print(
            f"  preprocessed {len(preprocessed)}/{len(pairs)} pair(s) in "
            f"{time.monotonic() - t0:.1f} s",
            file=sys.stderr,
        )
        if not preprocessed:
            print("no pairs survived preprocessing; aborting", file=sys.stderr)
            return 1
        context_provider = None

    n_to_dispatch = "tbd via manifest" if resume else len(preprocessed)
    print(
        f"phase 2/2: GMAT sweep ({n_to_dispatch} runs to dispatch, "
        f"{args.workers} workers; postprocess runs inline as a hook)",
        file=sys.stderr,
    )
    t0 = time.monotonic()
    _dispatch_sweep(
        preprocessed,
        args.mission,
        args.output_dir,
        args.manifest,
        args.gmat_sw_file,
        args.workers,
        resume=resume,
        context_provider=context_provider,
    )
    print(f"  sweep finished in {time.monotonic() - t0:.1f} s", file=sys.stderr)

    ok, failed = _summarise_manifest(args.manifest)
    print(f"DONE: {ok} run(s) ok, {failed} failed/missing", file=sys.stderr)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
