"""Empirical truth-floor for next-TLE-as-truth.

Estimates the order-of-magnitude size of `Δ_OD(epoch_j)` -- the operator's
own orbit-determination residual at the truth epoch -- by propagating
`TLE_{j-N}` forward to `t_j` with the same GMAT high-fidelity force model
the main sweep uses, and comparing to `SGP4(TLE_j, 0)`. Short arcs minimise
the integrated propagation contribution, so the residual is dominated by
the operator's `Δ_OD(epoch_j)` and a smaller `Δ_OD(epoch_{j-N})` carried
forward by Φ.

For the densest corpus satellite in the raw cache (NORAD 53675 by default,
a v1.5 in the 550 km shell with 131 TLEs over the analysis window), we
sweep `N ∈ {1, 2, 3, 6}` -- arc lengths of order 5 / 10 / 15 / 30 hours.
Maneuver-bearing arcs are filtered with the same 100 m SMA-jump rule the
corpus pipeline uses.

Outputs land under `outputs/_truth_floor/` (per-arc GMAT scratch) and
`outputs/_truth_floor.{parquet,json,png}` -- gitignored, just like the
peer `_diagnostic_sweep_scatter.py`. The `.json` carries the headline
medians so §2.1 prose can quote without re-running this script.

Usage (from repo root, GMAT-enabled env active):
    python src/scripts/_truth_floor.py \\
        --raw src/data/tles_raw.parquet \\
        --cache src/static/tles_cache.parquet \\
        --mission sweep/mission.script \\
        --output-root outputs/_truth_floor \\
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Running `python src/scripts/_truth_floor.py` puts only `src/scripts/`
# on sys.path; the `sweep` namespace package lives at the repo root, so
# add it explicitly. Mirrors the `tests/conftest.py` pattern.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from gmat_sweep import LocalJoblibPool, Manifest, RunSpec, Sweep  # noqa: E402

# We deliberately reach into sweep.run_sweep for the rotation + epoch
# helpers rather than copy-pasting them, so the diagnostic stays
# bit-consistent with the sweep that produced the corpus medians it is
# trying to characterise.
from sweep.run_sweep import (  # noqa: E402, PLC2701
    _gmat_epoch_string,
    _jd_fr_from_epoch,
    _sgp4_state_teme,
    _teme_to_mj2000,
)
from sweep.spacecraft_props import CD, CR  # noqa: E402
from sweep.tle_pipeline import detect_maneuver_epochs  # noqa: E402

ARC_OFFSETS_N: tuple[int, ...] = (1, 2, 3, 6)


@dataclass(frozen=True, slots=True)
class _ArcSpec:
    arc_id: int
    n: int  # j - i index offset
    j: int  # TLE_j index in the per-sat history (the truth-side TLE)
    epoch_i: pd.Timestamp  # = epoch of TLE_{j-N}
    epoch_j: pd.Timestamp
    actual_dt_sec: float
    line1_i: str
    line2_i: str
    line1_j: str
    line2_j: str


def _pick_densest_corpus_sat(raw: pd.DataFrame, cache: pd.DataFrame) -> int:
    """Return the NORAD ID with the longest TLE history that is also in the corpus.

    Restricting to corpus sats ensures the diagnostic's
    `(dry_mass, drag_area, srp_area)` exactly match what the main sweep
    propagated; an off-corpus sat would force us to re-derive props.
    """
    corpus_ids = set(cache["norad_id"].unique())
    counts = (
        raw[raw["norad_id"].isin(corpus_ids)]
        .groupby("norad_id")
        .size()
        .sort_values(ascending=False)
    )
    if counts.empty:
        raise SystemExit("no corpus satellites appear in the raw TLE cache")
    return int(counts.index[0])


def _build_arc_specs(
    sat_tles: pd.DataFrame,
    maneuver_epochs: pd.DatetimeIndex,
    offsets: tuple[int, ...],
) -> list[_ArcSpec]:
    """For each `N ∈ offsets`, emit one arc per `j ≥ N` with no maneuver inside (t_{j-N}, t_j]."""
    sat_tles = sat_tles.sort_values("epoch").reset_index(drop=True)
    specs: list[_ArcSpec] = []
    arc_id = 0
    for n in offsets:
        for j in range(n, len(sat_tles)):
            i_row = sat_tles.iloc[j - n]
            j_row = sat_tles.iloc[j]
            window = (i_row["epoch"], j_row["epoch"])
            if (
                maneuver_epochs.size
                and ((maneuver_epochs > window[0]) & (maneuver_epochs <= window[1])).any()
            ):
                continue
            specs.append(
                _ArcSpec(
                    arc_id=arc_id,
                    n=n,
                    j=j,
                    epoch_i=i_row["epoch"],
                    epoch_j=j_row["epoch"],
                    actual_dt_sec=(j_row["epoch"] - i_row["epoch"]).total_seconds(),
                    line1_i=i_row["line1"],
                    line2_i=i_row["line2"],
                    line1_j=j_row["line1"],
                    line2_j=j_row["line2"],
                ),
            )
            arc_id += 1
    return specs


def _build_run_spec(
    arc: _ArcSpec,
    mission_path: Path,
    output_root: Path,
    props: dict,
) -> RunSpec:
    """Build the GMAT RunSpec for a single arc. Mirrors run_sweep._build_run_spec."""
    jd_i, fr_i = _jd_fr_from_epoch(arc.epoch_i)
    r_teme, v_teme = _sgp4_state_teme(arc.line1_i, arc.line2_i, jd_i, fr_i)
    r_mj, v_mj = _teme_to_mj2000(r_teme, v_teme, arc.epoch_i)
    return RunSpec(
        script_path=mission_path,
        overrides={
            "Sat.Epoch": _gmat_epoch_string(arc.epoch_i),
            "Sat.X": float(r_mj[0]),
            "Sat.Y": float(r_mj[1]),
            "Sat.Z": float(r_mj[2]),
            "Sat.VX": float(v_mj[0]),
            "Sat.VY": float(v_mj[1]),
            "Sat.VZ": float(v_mj[2]),
            "Sat.DryMass": float(props["dry_mass_kg"]),
            "Sat.Cd": float(CD),
            "Sat.DragArea": float(props["drag_area_m2"]),
            "Sat.Cr": float(CR),
            "Sat.SRPArea": float(props["srp_area_m2"]),
            "elapsed_seconds.Value": float(arc.actual_dt_sec),
        },
        output_dir=output_root / f"run_{arc.arc_id}",
        run_id=arc.arc_id,
        seed=None,
        run_options={"overwrite": True},
    )


def _final_state_mj(report_path: Path) -> np.ndarray:
    df = pd.read_parquet(report_path)
    if "time" in df.columns:
        df = df.sort_values("time")
    last = df.iloc[-1]
    return np.array([last["Sat.X"], last["Sat.Y"], last["Sat.Z"]], dtype=float)


def _truth_state_mj(arc: _ArcSpec) -> np.ndarray:
    jd_j, fr_j = _jd_fr_from_epoch(arc.epoch_j)
    r_teme, v_teme = _sgp4_state_teme(arc.line1_j, arc.line2_j, jd_j, fr_j)
    r_mj, _ = _teme_to_mj2000(r_teme, v_teme, arc.epoch_j)
    return r_mj


def _aggregate(
    specs: list[_ArcSpec],
    manifest_path: Path,
    output_root: Path,
) -> pd.DataFrame:
    by_id = {spec.arc_id: spec for spec in specs}
    manifest = Manifest.load(manifest_path)
    rows = []
    for entry in manifest.entries:
        if entry.status != "ok":
            continue
        arc = by_id.get(entry.run_id)
        if arc is None:
            continue
        report_path = entry.output_paths.get("report__FinalState")
        if report_path is None or not Path(report_path).exists():
            print(f"  arc {entry.run_id}: no FinalState report", file=sys.stderr)
            continue
        r_hifi = _final_state_mj(Path(report_path))
        r_truth = _truth_state_mj(arc)
        rows.append(
            {
                "arc_id": arc.arc_id,
                "n": arc.n,
                "t_iN": arc.epoch_i,
                "t_j": arc.epoch_j,
                "actual_dt_h": arc.actual_dt_sec / 3600.0,
                "dr_km": float(np.linalg.norm(r_hifi - r_truth)),
            },
        )
    if not rows:
        raise SystemExit("no arcs survived the sweep; cannot aggregate")
    return pd.DataFrame(rows)


def _emit_summary(df: pd.DataFrame, sat_props: dict, out_json: Path) -> dict:
    by_n = {}
    for n, sub in df.groupby("n"):
        by_n[int(n)] = {
            "n_arcs": int(len(sub)),
            "median_dt_h": float(sub["actual_dt_h"].median()),
            "median_dr_km": float(sub["dr_km"].median()),
            "q25_dr_km": float(sub["dr_km"].quantile(0.25)),
            "q75_dr_km": float(sub["dr_km"].quantile(0.75)),
        }
    summary = {
        "satellite": sat_props,
        "by_n": by_n,
        "headline_median_dr_km": by_n[min(by_n)]["median_dr_km"],
        "headline_n": min(by_n),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    return summary


def _plot(df: pd.DataFrame, summary: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = sorted(df["n"].unique())
    positions = list(range(len(ns)))
    box_data = [df.loc[df["n"] == n, "dr_km"] for n in ns]
    ax.boxplot(box_data, positions=positions, widths=0.6, showfliers=False)
    rng = np.random.default_rng(seed=0)
    for i, _n in enumerate(ns):
        ax.scatter(
            np.full(len(box_data[i]), i) + rng.uniform(-0.12, 0.12, len(box_data[i])),
            box_data[i],
            s=8,
            alpha=0.35,
            color="#1f77b4",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"N={n}\n(~{summary['by_n'][n]['median_dt_h']:.1f} h)" for n in ns],
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Vert$GMAT$_{j-N\rightarrow j}-$SGP4$(\mathrm{TLE}_j, 0)\Vert$ (km)")
    ax.set_xlabel("arc offset")
    ax.set_title(
        f"Truth-floor diagnostic — NORAD {summary['satellite']['norad_id']} "
        f"({summary['satellite']['generation']}, {summary['satellite']['alt_shell']} km)",
    )
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("src/data/tles_raw.parquet"))
    parser.add_argument("--cache", type=Path, default=Path("src/static/tles_cache.parquet"))
    parser.add_argument("--mission", type=Path, default=Path("sweep/mission.script"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/_truth_floor"))
    parser.add_argument(
        "--norad-id",
        type=int,
        default=None,
        help="Override the densest-sat picker. Useful for testing.",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    raw = pd.read_parquet(args.raw)
    cache = pd.read_parquet(args.cache)
    norad_id = args.norad_id or _pick_densest_corpus_sat(raw, cache)
    sat_tles = raw[raw["norad_id"] == norad_id].copy()
    props_row = cache[cache["norad_id"] == norad_id].iloc[0]
    sat_props = {
        "norad_id": int(norad_id),
        "generation": str(props_row["generation"]),
        "alt_shell": str(props_row["alt_shell"]),
        "dry_mass_kg": float(props_row["dry_mass_kg"]),
        "drag_area_m2": float(props_row["drag_area_m2"]),
        "srp_area_m2": float(props_row["srp_area_m2"]),
        "n_tles_in_window": int(len(sat_tles)),
    }
    print(f"sat {norad_id}: {sat_props}", file=sys.stderr)

    # Reuse the corpus pipeline's maneuver detector so the diagnostic's
    # rejection rule matches the paper's (100 m SMA jump).
    maneuvers = detect_maneuver_epochs(sat_tles)
    maneuver_epochs = pd.DatetimeIndex(maneuvers["maneuver_epoch"])
    specs = _build_arc_specs(sat_tles, maneuver_epochs, ARC_OFFSETS_N)
    if not specs:
        print("no arcs survived the maneuver filter; aborting", file=sys.stderr)
        return 1
    print(
        f"built {len(specs)} arc(s) across N={ARC_OFFSETS_N}; "
        f"{len(maneuver_epochs)} maneuver epoch(s)",
        file=sys.stderr,
    )

    output_root = args.output_root.resolve()
    mission_path = args.mission.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    # Fresh manifest each run; this is a one-shot diagnostic, no resume path.
    manifest_path.unlink(missing_ok=True)

    run_specs = [_build_run_spec(arc, mission_path, output_root, sat_props) for arc in specs]
    with LocalJoblibPool(max_workers=args.workers) as pool:
        Sweep(
            runs=run_specs,
            backend=pool,
            manifest_path=manifest_path,
            output_dir=output_root,
            script_path=mission_path,
            parameter_spec={"_kind": "explicit", "columns": [], "rows": []},
            sweep_seed=None,
            progress=True,
        ).run()

    df = _aggregate(specs, manifest_path, output_root)
    parquet_path = args.output_root.parent / "_truth_floor.parquet"
    json_path = args.output_root.parent / "_truth_floor.json"
    png_path = args.output_root.parent / "_truth_floor.png"
    df.to_parquet(parquet_path, index=False)
    summary = _emit_summary(df, sat_props, json_path)
    _plot(df, summary, png_path)

    print(
        "summary:",
        json.dumps(
            {"by_n": summary["by_n"], "headline_median_dr_km": summary["headline_median_dr_km"]},
            indent=2,
        ),
        file=sys.stderr,
    )
    print(f"wrote {parquet_path}, {json_path}, {png_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
