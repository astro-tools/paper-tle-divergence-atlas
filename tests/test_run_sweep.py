"""Unit tests for sweep.run_sweep — pure-Python helpers only.

The full driver loop touches gmat-sweep, gmat-run, and the GMAT engine, which
require subprocess dispatch and a GMAT install; those paths exercise via the
N=8 smoke run, not pytest. What's tested here:

  - The TEME→MJ2000Eq rotation matrix is a proper rotation (orthogonal,
    det = +1), round-trips, and agrees with astropy's geocentric GCRS chain
    at the LEO scale our paper cares about.
  - The RSW (radial/along/cross) decomposition is correct on canonical
    perturbations of a circular orbit and is norm-preserving.
  - Per-pair preprocessing yields finite LEO-scale states with the right
    shape.
  - The GMAT epoch string and RunSpec wiring are well-formed.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import GCRS, TEME, CartesianDifferential, CartesianRepresentation
from astropy.time import Time

from sweep.run_sweep import (
    _build_run_spec,
    _decompose_rsw,
    _filter_pairs_needing_postprocess,
    _gmat_epoch_string,
    _postprocess_all,
    _postprocess_run,
    _preprocess_pair,
    _Preprocessed,
    _resume_compatible,
    _resume_dispatch,
    _teme_to_mj2000,
    _teme_to_mj2000_matrix,
)
from sweep.space_weather import SwRow

# Real STARLINK-1007 TLE (same sat used by issue #1's validation script).
LINE1_A = "1 44713U 19074A   26091.50000000  .00001234  00000-0  12345-3 0  9991"
LINE2_A = "2 44713  53.0540 116.6831 0001247  82.6886 277.4255 15.06405853250789"
# Same sat, slightly later epoch (≈ 1 day later) for pair construction.
LINE1_B = "1 44713U 19074A   26092.50000000  .00001234  00000-0  12345-3 0  9991"
LINE2_B = "2 44713  53.0540 116.6831 0001247  82.6886 277.4255 15.06405853250799"


class TestTemeToMj2000Matrix:
    def test_orthogonal(self) -> None:
        epoch = pd.Timestamp("2026-04-01T00:00:00Z")
        m = _teme_to_mj2000_matrix(epoch)
        assert m.shape == (3, 3)
        np.testing.assert_allclose(m.T @ m, np.eye(3), atol=1e-12)

    def test_proper_rotation(self) -> None:
        epoch = pd.Timestamp("2026-04-15T12:34:56Z")
        m = _teme_to_mj2000_matrix(epoch)
        assert np.isclose(np.linalg.det(m), 1.0, atol=1e-12)

    def test_round_trip_state(self) -> None:
        epoch = pd.Timestamp("2026-04-01T00:00:00Z")
        m = _teme_to_mj2000_matrix(epoch)
        r_teme = np.array([6800.0, 100.0, -50.0])
        r_back = m.T @ (m @ r_teme)
        np.testing.assert_allclose(r_back, r_teme, atol=1e-9)

    def test_agrees_with_astropy_gcrs_within_frame_bias(self) -> None:
        # GCRS and MJ2000Eq differ by the IAU frame-bias matrix — ≈ 17 mas
        # total, or ≈ 0.6 m at 7000 km radius. Our rotation should land in
        # that ballpark when compared against astropy's TEME→GCRS chain.
        epoch = pd.Timestamp("2026-04-01T00:00:00Z")
        r_teme = np.array([6800.0, 0.0, 0.0])
        v_teme = np.array([0.0, 7.5, 0.5])
        r_mj, v_mj = _teme_to_mj2000(r_teme, v_teme, epoch)

        obstime = Time(epoch.tz_convert("UTC").to_pydatetime(), scale="utc")
        rep = CartesianRepresentation(
            x=r_teme[0] * u.km,
            y=r_teme[1] * u.km,
            z=r_teme[2] * u.km,
            differentials=CartesianDifferential(
                d_x=v_teme[0] * u.km / u.s,
                d_y=v_teme[1] * u.km / u.s,
                d_z=v_teme[2] * u.km / u.s,
            ),
        )
        gcrs = TEME(rep, obstime=obstime).transform_to(GCRS(obstime=obstime))
        r_gcrs = gcrs.cartesian.xyz.to_value(u.km)

        # < 2 m disagreement at LEO between IAU-1976 MJ2000Eq and IAU-2006 GCRS.
        assert np.linalg.norm(r_mj - r_gcrs) < 2.0e-3  # km


class TestDecomposeRsw:
    # A circular orbit at periapsis: r along +x, v along +y at the orbit radius.
    R_CIRC = np.array([7000.0, 0.0, 0.0])
    V_CIRC = np.array([0.0, 7.546, 0.0])  # rough LEO speed

    def test_pure_radial(self) -> None:
        delta = np.array([0.5, 0.0, 0.0])
        radial, along, cross = _decompose_rsw(delta, self.R_CIRC, self.V_CIRC)
        assert np.isclose(radial, 0.5)
        assert np.isclose(along, 0.0, atol=1e-12)
        assert np.isclose(cross, 0.0, atol=1e-12)

    def test_pure_along_track(self) -> None:
        # Along-track is +ê_t = ê_h × ê_r. For ê_r = +x, ê_h = +z, ê_t = +y.
        delta = np.array([0.0, 1.5, 0.0])
        radial, along, cross = _decompose_rsw(delta, self.R_CIRC, self.V_CIRC)
        assert np.isclose(radial, 0.0, atol=1e-12)
        assert np.isclose(along, 1.5)
        assert np.isclose(cross, 0.0, atol=1e-12)

    def test_pure_cross_track(self) -> None:
        # Cross-track is +ê_h = (r × v)̂. For r=+x, v=+y, ê_h = +z.
        delta = np.array([0.0, 0.0, 2.0])
        radial, along, cross = _decompose_rsw(delta, self.R_CIRC, self.V_CIRC)
        assert np.isclose(radial, 0.0, atol=1e-12)
        assert np.isclose(along, 0.0, atol=1e-12)
        assert np.isclose(cross, 2.0)

    def test_norm_preserved(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            delta = rng.normal(size=3)
            radial, along, cross = _decompose_rsw(delta, self.R_CIRC, self.V_CIRC)
            assert np.isclose(radial**2 + along**2 + cross**2, float(np.dot(delta, delta)))


_SW_FIXTURE = {
    dt.date(2026, 4, 1): SwRow(f107_obs=141.9, f107_avg81=125.8, ap_daily=8.0, is_observed=True),
}


class TestPreprocessPair:
    def _pair(self) -> pd.Series:
        return pd.Series(
            {
                "norad_id": 44713,
                "target_dt_sec": 86_400,
                "epoch_i": pd.Timestamp("2026-04-01T12:00:00Z"),
                "epoch_j": pd.Timestamp("2026-04-02T12:00:00Z"),
                "actual_dt_sec": 86_400.0,
                "alt_shell": "550",
                "line1_i": LINE1_A,
                "line2_i": LINE2_A,
                "line1_j": LINE1_B,
                "line2_j": LINE2_B,
                "dry_mass_kg": 248.0,
                "drag_area_m2": 5.0,
                "srp_area_m2": 5.0,
            }
        )

    def test_props_flow_through_to_preprocessed(self) -> None:
        pre = _preprocess_pair(0, self._pair(), _SW_FIXTURE)
        assert pre.dry_mass_kg == 248.0
        assert pre.drag_area_m2 == 5.0
        assert pre.srp_area_m2 == 5.0

    def test_returns_leo_scale_states(self) -> None:
        pre = _preprocess_pair(0, self._pair(), _SW_FIXTURE)
        # Initial state magnitude ~6800–6900 km at 550 km altitude.
        assert 6700.0 < np.linalg.norm(pre.r_init_mj_km) < 7000.0
        # Orbital speed ~7.5 km/s.
        assert 7.0 < np.linalg.norm(pre.v_init_mj_km_s) < 8.0

    def test_prediction_differs_from_initial_state(self) -> None:
        # Propagating TLE_i forward by Δt ≈ 1 day (≈ 15.06 revs) lands the
        # satellite at a different orbital phase from its own TLE epoch.
        # The 0.06-rev fractional component ≈ 23° on a 7000-km orbit ≈
        # thousands of km of separation in inertial coordinates.
        pre = _preprocess_pair(0, self._pair(), _SW_FIXTURE)
        assert np.linalg.norm(pre.r_sgp4_pred_mj_km - pre.r_init_mj_km) > 100.0

    def test_all_states_finite(self) -> None:
        pre = _preprocess_pair(7, self._pair(), _SW_FIXTURE)
        for arr in (
            pre.r_init_mj_km,
            pre.v_init_mj_km_s,
            pre.r_sgp4_pred_mj_km,
            pre.r_truth_mj_km,
            pre.v_truth_mj_km_s,
        ):
            assert np.all(np.isfinite(arr))
            assert arr.shape == (3,)

    def test_sw_values_attached_from_lookup(self) -> None:
        pre = _preprocess_pair(0, self._pair(), _SW_FIXTURE)
        assert pre.f107_obs == 141.9
        assert pre.ap_daily == 8.0

    def test_missing_sw_date_raises_keyerror(self) -> None:
        # A pair whose epoch_i isn't in the SW cache must fail loudly,
        # not silently NaN: per issue #14's acceptance criteria.
        pair = self._pair()
        pair["epoch_i"] = pd.Timestamp("2027-01-15T12:00:00Z")
        with pytest.raises(KeyError, match="no space-weather entry"):
            _preprocess_pair(0, pair, _SW_FIXTURE)


class TestGmatEpochString:
    def test_format_matches_gmat_utcgregorian(self) -> None:
        ts = pd.Timestamp("2026-04-01T12:34:56.789000Z")
        assert _gmat_epoch_string(ts) == "01 Apr 2026 12:34:56.789"

    def test_microsecond_truncation_is_floor(self) -> None:
        ts = pd.Timestamp("2026-04-01T00:00:00.123456Z")
        assert _gmat_epoch_string(ts).endswith(".123")


class TestBuildRunSpec:
    def _pre(self, run_id: int = 3) -> _Preprocessed:
        return _Preprocessed(
            run_id=run_id,
            norad_id=44713,
            target_dt_sec=86_400,
            epoch_i=pd.Timestamp("2026-04-01T00:00:00Z"),
            epoch_j=pd.Timestamp("2026-04-02T00:00:00Z"),
            actual_dt_sec=86_400.0,
            alt_shell="550",
            r_init_mj_km=np.array([6800.0, 0.0, 0.0]),
            v_init_mj_km_s=np.array([0.0, 7.5, 0.5]),
            r_sgp4_pred_mj_km=np.array([0.0, 0.0, 0.0]),
            r_truth_mj_km=np.array([0.0, 0.0, 0.0]),
            v_truth_mj_km_s=np.array([0.0, 0.0, 0.0]),
            dry_mass_kg=305.0,
            drag_area_m2=5.0,
            srp_area_m2=5.0,
            f107_obs=141.9,
            ap_daily=8.0,
        )

    def test_overrides_complete(self) -> None:
        from pathlib import Path

        spec = _build_run_spec(self._pre(), Path("mission.script"), Path("outputs"))
        assert set(spec.overrides) == {
            "Sat.Epoch",
            "Sat.X",
            "Sat.Y",
            "Sat.Z",
            "Sat.VX",
            "Sat.VY",
            "Sat.VZ",
            "Sat.DryMass",
            "Sat.Cd",
            "Sat.DragArea",
            "Sat.Cr",
            "Sat.SRPArea",
            "elapsed_seconds.Value",
        }

    def test_per_sat_props_propagate_into_overrides(self) -> None:
        from pathlib import Path

        from sweep.spacecraft_props import CD, CR

        spec = _build_run_spec(self._pre(), Path("m.script"), Path("outputs"))
        assert spec.overrides["Sat.DryMass"] == 305.0
        assert spec.overrides["Sat.DragArea"] == 5.0
        assert spec.overrides["Sat.SRPArea"] == 5.0
        assert spec.overrides["Sat.Cd"] == CD
        assert spec.overrides["Sat.Cr"] == CR

    def test_output_dir_nests_run_id(self) -> None:
        from pathlib import Path

        spec = _build_run_spec(self._pre(run_id=7), Path("m.script"), Path("outputs"))
        assert spec.output_dir == Path("outputs/run_7")
        assert spec.run_id == 7

    def test_run_options_set_overwrite_true(self) -> None:
        # mission.run(overwrite=True) lets resumed/retried runs clear
        # stale partial outputs left by a Ctrl-C'd worker. Without it,
        # every retry of an interrupted run_id fails with "working_dir
        # already contains output files".
        from pathlib import Path

        spec = _build_run_spec(self._pre(), Path("m.script"), Path("outputs"))
        assert spec.run_options == {"overwrite": True}

    def test_override_values_are_json_safe(self) -> None:
        from pathlib import Path

        spec = _build_run_spec(self._pre(), Path("m.script"), Path("outputs"))
        # RunSpec.overrides values must be JSON-encodable for the manifest.
        # Floats and strings; no numpy scalars.
        for value in spec.overrides.values():
            assert isinstance(value, (str, float)), (
                f"{type(value).__name__} is not JSON-safe in RunSpec.overrides"
            )


class TestPostprocessRunSchema:
    """End-to-end check that SW values land in the per-run parquet."""

    def _pre(self) -> _Preprocessed:
        return _Preprocessed(
            run_id=42,
            norad_id=44713,
            target_dt_sec=86_400,
            epoch_i=pd.Timestamp("2026-04-01T00:00:00Z"),
            epoch_j=pd.Timestamp("2026-04-02T00:00:00Z"),
            actual_dt_sec=86_400.0,
            alt_shell="550",
            r_init_mj_km=np.array([6800.0, 0.0, 0.0]),
            v_init_mj_km_s=np.array([0.0, 7.5, 0.0]),
            r_sgp4_pred_mj_km=np.array([6800.1, 0.0, 0.0]),
            r_truth_mj_km=np.array([6800.0, 0.0, 0.0]),
            v_truth_mj_km_s=np.array([0.0, 7.5, 0.0]),
            dry_mass_kg=248.0,
            drag_area_m2=5.0,
            srp_area_m2=5.0,
            f107_obs=141.9,
            ap_daily=8.0,
        )

    def test_f107_and_ap_are_real_floats_not_nan(self, tmp_path) -> None:
        # Synthesise a GMAT FinalState report parquet with a single row.
        report = tmp_path / "report__FinalState.parquet"
        pd.DataFrame(
            {
                "time": [0.0],
                "Sat.X": [6800.0],
                "Sat.Y": [0.0],
                "Sat.Z": [0.0],
                "Sat.VX": [0.0],
                "Sat.VY": [7.5],
                "Sat.VZ": [0.0],
            }
        ).to_parquet(report, index=False)

        out = tmp_path / "run_42.parquet"
        _postprocess_run(self._pre(), report, out)

        df = pd.read_parquet(out)
        assert df["f107"].dtype.kind == "f"
        assert df["ap"].dtype.kind == "f"
        assert df["f107"].iloc[0] == pytest.approx(141.9)
        assert df["ap"].iloc[0] == pytest.approx(8.0)
        assert np.isfinite(df["f107"].iloc[0])
        assert np.isfinite(df["ap"].iloc[0])


class TestFilterPairsNeedingPostprocess:
    """Resume helper: keep pairs that don't already have a comparison parquet."""

    def _pairs(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({"norad_id": list(range(n)), "x": [0] * n}).reset_index(drop=True)

    def test_no_parquets_returns_all(self, tmp_path) -> None:
        pairs = self._pairs(5)
        result = _filter_pairs_needing_postprocess(pairs, tmp_path)
        assert len(result) == 5

    def test_drops_pairs_with_existing_parquet(self, tmp_path) -> None:
        pairs = self._pairs(5)
        (tmp_path / "run_0.parquet").write_bytes(b"x")
        (tmp_path / "run_3.parquet").write_bytes(b"x")
        result = _filter_pairs_needing_postprocess(pairs, tmp_path)
        assert list(result.index) == [1, 2, 4]

    def test_all_done_returns_empty(self, tmp_path) -> None:
        pairs = self._pairs(3)
        for i in range(3):
            (tmp_path / f"run_{i}.parquet").write_bytes(b"x")
        assert len(_filter_pairs_needing_postprocess(pairs, tmp_path)) == 0


class TestResumeCompatible:
    """Resume gate: refuse when the manifest doesn't match the current corpus."""

    def _write_manifest(self, path: Path, *, run_count: int) -> None:
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
            "run_count": run_count,
            "backend": "t",
        }
        path.write_text(json.dumps(header, sort_keys=True) + "\n", encoding="utf-8")

    def test_missing_manifest_is_fresh(self, tmp_path) -> None:
        assert _resume_compatible(tmp_path / "missing.jsonl", n_corpus_pairs=10) is False

    def test_matching_run_count_resumes(self, tmp_path) -> None:
        path = tmp_path / "manifest.jsonl"
        self._write_manifest(path, run_count=100)
        assert _resume_compatible(path, n_corpus_pairs=100) is True

    def test_mismatched_run_count_refuses(self, tmp_path) -> None:
        # Smoke (8 rows) → sweep (24641 rows) is the canonical case here.
        path = tmp_path / "manifest.jsonl"
        self._write_manifest(path, run_count=8)
        with pytest.raises(SystemExit, match="run_count=8.*24641"):
            _resume_compatible(path, n_corpus_pairs=24_641)


class TestResumeDispatchScriptHashCheck:
    """Resume safety: refuse when the mission script changed since the manifest."""

    def _write_manifest(self, path: Path, *, script_sha256: str) -> None:
        header = {
            "schema_version": 1,
            "script_sha256": script_sha256,
            "gmat_sweep_version": "t",
            "gmat_run_version": "t",
            "gmat_install_version": "t",
            "python_version": "3.12",
            "os_platform": "t",
            "sweep_seed": None,
            "parameter_spec": {"_kind": "explicit", "columns": [], "rows": []},
            "run_count": 1,
            "backend": "t",
        }
        path.write_text(json.dumps(header, sort_keys=True) + "\n", encoding="utf-8")

    def test_script_hash_drift_raises_systemexit(self, tmp_path) -> None:
        # A drifted manifest must refuse resume — otherwise the resumed
        # runs would load a different script than the already-ok runs
        # and the aggregated frame would be a mongrel.
        script = tmp_path / "mission.script"
        script.write_text("Create Spacecraft Sat\n", encoding="utf-8")

        manifest = tmp_path / "manifest.jsonl"
        self._write_manifest(manifest, script_sha256="b" * 64)  # not the real hash

        with pytest.raises(SystemExit, match="script hash drifted"):
            _resume_dispatch(
                preprocessed=[],
                mission=script,
                output_dir=tmp_path,
                manifest_path=manifest,
                workers=1,
            )


class TestPostprocessAllSkipsExistingParquets:
    """Idempotency: re-running postprocess does not rewrite finished parquets."""

    def _entry(self, run_id: int, status: str, report_path: Path | None) -> dict:
        now = dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=dt.UTC).isoformat()
        return {
            "run_id": run_id,
            "overrides": {},
            "status": status,
            "output_paths": {"report__FinalState": str(report_path)} if report_path else {},
            "started_at": now,
            "ended_at": now,
            "duration_s": 1.0,
            "stderr": None,
            "log_path": None,
        }

    def _write_manifest(self, path: Path, entries: list[dict]) -> None:
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
            "run_count": len(entries),
            "backend": "t",
        }
        lines = [json.dumps(header, sort_keys=True)]
        lines.extend(json.dumps(e, sort_keys=True) for e in entries)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_existing_parquet_is_not_rewritten(self, tmp_path) -> None:
        # An ok manifest entry whose run_<id>.parquet already exists must
        # not be re-postprocessed, even when its _Preprocessed is in the
        # batch (e.g. preprocessing ran defensively over all corpus rows).
        out_path = tmp_path / "run_42.parquet"
        sentinel = b"already postprocessed"
        out_path.write_bytes(sentinel)

        report_path = tmp_path / "report__FinalState.parquet"
        report_path.write_bytes(b"")  # presence only; never read

        manifest_path = tmp_path / "manifest.jsonl"
        self._write_manifest(manifest_path, [self._entry(42, "ok", report_path)])

        pre = _Preprocessed(
            run_id=42,
            norad_id=44713,
            target_dt_sec=86_400,
            epoch_i=pd.Timestamp("2026-04-01T00:00:00Z"),
            epoch_j=pd.Timestamp("2026-04-02T00:00:00Z"),
            actual_dt_sec=86_400.0,
            alt_shell="550",
            r_init_mj_km=np.array([6800.0, 0.0, 0.0]),
            v_init_mj_km_s=np.array([0.0, 7.5, 0.0]),
            r_sgp4_pred_mj_km=np.array([6800.1, 0.0, 0.0]),
            r_truth_mj_km=np.array([6800.0, 0.0, 0.0]),
            v_truth_mj_km_s=np.array([0.0, 7.5, 0.0]),
            dry_mass_kg=248.0,
            drag_area_m2=5.0,
            srp_area_m2=5.0,
            f107_obs=141.9,
            ap_daily=8.0,
        )

        ok, failed = _postprocess_all([pre], manifest_path, tmp_path)
        assert ok == 1 and failed == 0
        # The sentinel bytes are intact — postprocess did not rewrite.
        assert out_path.read_bytes() == sentinel


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
