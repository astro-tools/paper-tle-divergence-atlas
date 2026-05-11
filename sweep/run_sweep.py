"""Run the full TLE-divergence-atlas sweep.

Drives gmat-sweep over the corpus of Starlink TLE pairs. Each run propagates
a satellite forward from t_i to t_j using both SGP4 (from TLE_i) and GMAT
high-fidelity force models, then compares both predictions against the
operator's next-TLE truth (SGP4(TLE_j, Δt=0)).

Pipeline (three phases, single process for the driver):

    1. Preprocess each pair (Python, no GMAT):
       - Evaluate SGP4(TLE_i, 0) and SGP4(TLE_i, dt) in TEME.
       - Evaluate SGP4(TLE_j, 0) — the operator's next-TLE truth — in TEME.
       - Rotate every state TEME → MJ2000Eq via Vallado's IAU 1976/1980
         procedure (precession × nutation × equation-of-equinox, composed
         from `erfa` primitives), so the state we hand to GMAT lives in
         the same inertial frame GMAT propagates in.

    2. GMAT sweep: build one RunSpec per pair with field overrides
       (`Sat.Epoch`, `Sat.X..VZ`, `elapsed_seconds`) and dispatch through
       `gmat_sweep.Sweep` over a `LocalJoblibPool`. Each run writes its
       ReportFile as `outputs/run_<id>/report__FinalState.parquet` and
       the sweep manifest lands at `--manifest` (default `sweep/manifest.jsonl`,
       outside `outputs/` so it can be committed).

    3. Postprocess each successful run: read the run's report parquet,
       take the final integration step (state at t_j in MJ2000Eq),
       compute Δr_hifi vs. truth, recover Δr_sgp4 from the precomputed
       SGP4 prediction, decompose both into radial/along/cross relative
       to the truth state's RSW frame, and emit a one-row
       `outputs/run_<id>.parquet` with the comparison columns.

`f107` and `ap` columns are written as NaN; space-weather data fetch and
join is deferred to a follow-up before figure F7 (Day 6).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import erfa
import numpy as np
import pandas as pd
from astropy.time import Time
from gmat_sweep import LocalJoblibPool, Manifest, RunSpec, Sweep
from sgp4.api import Satrec, jday

DEFAULT_SMOKE_PER_BUCKET: Final = 2  # 4 Δt buckets × 2 → N=8 total
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


def _preprocess_pair(run_id: int, pair: pd.Series) -> _Preprocessed:
    jd_i, fr_i = _jd_fr_from_epoch(pair["epoch_i"])
    r_init_teme, v_init_teme = _sgp4_state_teme(pair["line1_i"], pair["line2_i"], jd_i, fr_i)
    r_init_mj, v_init_mj = _teme_to_mj2000(r_init_teme, v_init_teme, pair["epoch_i"])

    jd_j, fr_j = _jd_fr_from_epoch(pair["epoch_j"])
    r_pred_teme, v_pred_teme = _sgp4_state_teme(pair["line1_i"], pair["line2_i"], jd_j, fr_j)
    r_pred_mj, _ = _teme_to_mj2000(r_pred_teme, v_pred_teme, pair["epoch_j"])

    r_truth_teme, v_truth_teme = _sgp4_state_teme(pair["line1_j"], pair["line2_j"], jd_j, fr_j)
    r_truth_mj, v_truth_mj = _teme_to_mj2000(r_truth_teme, v_truth_teme, pair["epoch_j"])

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
    )


# --- RunSpec build ---------------------------------------------------------


def _gmat_epoch_string(epoch: pd.Timestamp) -> str:
    """Format as GMAT UTCGregorian: '01 Apr 2026 12:34:56.789'."""
    e = epoch.tz_convert("UTC")
    return f"{e.strftime('%d %b %Y %H:%M:%S')}.{e.microsecond // 1000:03d}"


def _build_run_spec(pre: _Preprocessed, mission_path: Path, output_root: Path) -> RunSpec:
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
            # GMAT Variables override via the .Value pseudo-field: bare name
            # alone fails gmat-run's "Resource.Field" path validation.
            "elapsed_seconds.Value": float(pre.actual_dt_sec),
        },
        output_dir=output_root / f"run_{pre.run_id}",
        run_id=pre.run_id,
        seed=None,
        run_options={},
    )


# --- Per-run postprocess ---------------------------------------------------


_REPORT_COLUMNS: Final = ("Sat.X", "Sat.Y", "Sat.Z", "Sat.VX", "Sat.VY", "Sat.VZ")


def _final_gmat_state(report_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(report_path)
    if "time" in df.columns:
        df = df.sort_values("time")
    last = df.iloc[-1]
    r = np.array([last["Sat.X"], last["Sat.Y"], last["Sat.Z"]], dtype=float)
    v = np.array([last["Sat.VX"], last["Sat.VY"], last["Sat.VZ"]], dtype=float)
    return r, v


def _postprocess_run(pre: _Preprocessed, report_path: Path, out_path: Path) -> dict:
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
        # SW data fetch deferred to a follow-up before F7 (Day 6).
        "f107": float("nan"),
        "ap": float("nan"),
    }
    pd.DataFrame([row]).to_parquet(out_path, index=False)
    return row


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
        action="store_true",
        help="Run a small stratified-random subset (2 pairs per Δt bucket, "
        "seed=42) instead of the full corpus — for pipeline validation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel workers (joblib backend)",
    )
    return parser.parse_args()


def _preprocess_all(pairs: pd.DataFrame) -> list[_Preprocessed]:
    out: list[_Preprocessed] = []
    for run_id, pair in pairs.iterrows():
        try:
            out.append(_preprocess_pair(int(run_id), pair))
        except RuntimeError as exc:
            print(
                f"  skip run_id={run_id} sat={pair['norad_id']}: {exc}",
                file=sys.stderr,
            )
    return out


def _dispatch_sweep(
    preprocessed: list[_Preprocessed],
    mission: Path,
    output_dir: Path,
    manifest_path: Path,
    workers: int,
) -> None:
    run_specs = [_build_run_spec(p, mission, output_dir) for p in preprocessed]
    parameter_spec = {
        "_kind": "explicit",
        "columns": [
            "Sat.Epoch",
            "Sat.X",
            "Sat.Y",
            "Sat.Z",
            "Sat.VX",
            "Sat.VY",
            "Sat.VZ",
            "elapsed_seconds.Value",
        ],
        "rows": [
            [
                spec.overrides[c]
                for c in (
                    "Sat.Epoch",
                    "Sat.X",
                    "Sat.Y",
                    "Sat.Z",
                    "Sat.VX",
                    "Sat.VY",
                    "Sat.VZ",
                    "elapsed_seconds.Value",
                )
            ]
            for spec in run_specs
        ],
    }
    with LocalJoblibPool(max_workers=workers) as pool:
        Sweep(
            runs=run_specs,
            backend=pool,
            manifest_path=manifest_path,
            output_dir=output_dir,
            script_path=mission,
            parameter_spec=parameter_spec,
            sweep_seed=None,
            progress=True,
        ).run()


def _postprocess_all(
    preprocessed: list[_Preprocessed],
    manifest_path: Path,
    output_dir: Path,
) -> tuple[int, int]:
    by_run_id = {p.run_id: p for p in preprocessed}
    manifest = Manifest.load(manifest_path)
    ok = 0
    failed = 0
    for entry in manifest.entries:
        pre = by_run_id.get(entry.run_id)
        if pre is None:
            continue  # not from this batch (resumed sweep)
        if entry.status != "ok":
            failed += 1
            continue
        report_path = entry.output_paths.get("report__FinalState")
        if report_path is None or not Path(report_path).exists():
            print(
                f"  run_id={entry.run_id}: no FinalState report; skipping postprocess",
                file=sys.stderr,
            )
            failed += 1
            continue
        out_path = output_dir / f"run_{pre.run_id}.parquet"
        _postprocess_run(pre, Path(report_path), out_path)
        ok += 1
    return ok, failed


def main() -> int:
    args = parse_args()

    pairs = pd.read_parquet(args.tles).reset_index(drop=True)
    if args.smoke:
        # Stratified-random by Δt bucket so all four horizons (6 h / 1 d / 3 d
        # / 7 d) are exercised; sats and altitude shells fall out randomly
        # within each stratum, so a single bad-sat or shell-specific bug
        # surfaces here rather than escaping to the full sweep. Seeded for
        # bit-reproducible smoke runs.
        pairs = (
            pairs.groupby("target_dt_sec", group_keys=False)
            .sample(n=DEFAULT_SMOKE_PER_BUCKET, random_state=DEFAULT_SMOKE_SEED)
            .reset_index(drop=True)
        )
    print(f"loaded {len(pairs)} pair(s) from {args.tles}", file=sys.stderr)

    print("phase 1/3: preprocess", file=sys.stderr)
    t0 = time.monotonic()
    preprocessed = _preprocess_all(pairs)
    print(
        f"  preprocessed {len(preprocessed)}/{len(pairs)} pair(s) in {time.monotonic() - t0:.1f} s",
        file=sys.stderr,
    )
    if not preprocessed:
        print("no pairs survived preprocessing; aborting", file=sys.stderr)
        return 1

    # GMAT resolves a relative working_dir against its installed OUTPUT_PATH
    # (e.g. /home/.../gmat-R2026a/output/), producing paths like
    # /home/.../gmat-R2026a/output/outputs/run_0/. Absolutise so per-run
    # files land where the caller pointed.
    args.output_dir = args.output_dir.resolve()
    args.manifest = args.manifest.resolve()
    args.mission = args.mission.resolve()
    args.tles = args.tles.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"phase 2/3: GMAT sweep ({len(preprocessed)} runs, {args.workers} workers)",
        file=sys.stderr,
    )
    t0 = time.monotonic()
    _dispatch_sweep(
        preprocessed,
        args.mission,
        args.output_dir,
        args.manifest,
        args.workers,
    )
    print(f"  sweep finished in {time.monotonic() - t0:.1f} s", file=sys.stderr)

    print("phase 3/3: postprocess", file=sys.stderr)
    t0 = time.monotonic()
    ok, failed = _postprocess_all(preprocessed, args.manifest, args.output_dir)
    print(
        f"  postprocessed {ok} run(s), {failed} failed/missing, {time.monotonic() - t0:.1f} s",
        file=sys.stderr,
    )

    print(f"DONE: {ok} run(s) ok, {failed} failed/missing", file=sys.stderr)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
