"""Empirical truth-floor for next-TLE-as-truth, per (altitude shell x generation) cohort.

Estimates the order-of-magnitude size of `Δ_OD(epoch_j)` -- the operator's
own orbit-determination residual at the truth epoch -- by propagating
`TLE_{j-N}` forward to `t_j` with the same GMAT high-fidelity force model
the main sweep uses, and comparing to `SGP4(TLE_j, 0)`. Short arcs minimise
the integrated propagation contribution, so the residual is dominated by
the operator's `Δ_OD(epoch_j)` and a smaller `Δ_OD(epoch_{j-N})` carried
forward by Φ.

By default the diagnostic runs the densest-sampled corpus satellite in
each populated `(alt_shell, generation)` cell of Table~\\ref{tab:corpus}
-- 8 cells, ~3,000 arcs total -- so §2.1's cohort-dependent claim is
backed by per-cohort numbers rather than a single representative. For
each chosen sat we sweep `N ∈ {1, 2, 3, 6}` (arc lengths of order
5 / 10 / 15 / 30 hours). Maneuver-bearing arcs are filtered with the
same 100 m SMA-jump rule the corpus pipeline uses.

Outputs land under `outputs/_truth_floor/` (per-arc GMAT scratch) and
`outputs/_truth_floor.{parquet,json,png}` -- gitignored, just like the
peer `_diagnostic_sweep_scatter.py`. The `.json` carries the headline
medians per cohort so §2.1 prose can quote without re-running.

Usage (from repo root, GMAT-enabled env active):
    python src/scripts/_truth_floor.py \\
        --raw src/data/tles_raw.parquet \\
        --cache src/static/tles_cache.parquet \\
        --mission sweep/mission.script \\
        --output-root outputs/_truth_floor \\
        --workers 8

Pass `--norad-id <N>` to restrict the diagnostic to a single sat (useful
when iterating on the script).
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
    norad_id: int
    alt_shell: str
    generation: str
    n: int  # j - i index offset
    j: int  # TLE_j index in the per-sat history (the truth-side TLE)
    epoch_i: pd.Timestamp  # = epoch of TLE_{j-N}
    epoch_j: pd.Timestamp
    actual_dt_sec: float
    line1_i: str
    line2_i: str
    line1_j: str
    line2_j: str


def _pick_densest_per_cohort(raw: pd.DataFrame, cache: pd.DataFrame) -> list[dict]:
    """One densest sat per populated (alt_shell, generation) cohort.

    Picks within the corpus pair cache so spacecraft properties exactly
    match what the main sweep propagated. v2-mini at the 540 km shell
    is absent in the corpus (see Table~\\ref{tab:corpus}); the
    enumeration falls out of the per-(shell, gen) groupby and naturally
    skips empty cells.
    """
    counts = raw.groupby("norad_id").size().rename("n_tles_in_window")
    sat_meta = (
        cache[["norad_id", "generation", "alt_shell", "dry_mass_kg", "drag_area_m2", "srp_area_m2"]]
        .drop_duplicates("norad_id")
        .merge(counts, left_on="norad_id", right_index=True)
    )
    picks = (
        sat_meta.sort_values("n_tles_in_window", ascending=False)
        .groupby(["alt_shell", "generation"], sort=True)
        .first()
        .reset_index()
    )
    return [
        {
            "norad_id": int(row.norad_id),
            "generation": str(row.generation),
            "alt_shell": str(row.alt_shell),
            "dry_mass_kg": float(row.dry_mass_kg),
            "drag_area_m2": float(row.drag_area_m2),
            "srp_area_m2": float(row.srp_area_m2),
            "n_tles_in_window": int(row.n_tles_in_window),
        }
        for row in picks.itertuples(index=False)
    ]


def _build_arc_specs(
    raw: pd.DataFrame,
    sats: list[dict],
    offsets: tuple[int, ...],
) -> list[_ArcSpec]:
    """For each sat in `sats` and each `N ∈ offsets`, emit one arc per `j ≥ N`.

    Maneuvers are detected per-sat against the 100 m SMA-jump rule and
    any arc whose interval covers a detected maneuver is dropped, mirroring
    the corpus pair-construction filter.
    """
    specs: list[_ArcSpec] = []
    arc_id = 0
    for sat in sats:
        sat_tles = (
            raw[raw["norad_id"] == sat["norad_id"]].sort_values("epoch").reset_index(drop=True)
        )
        maneuvers = detect_maneuver_epochs(sat_tles)
        maneuver_epochs = pd.DatetimeIndex(maneuvers["maneuver_epoch"])
        n_arcs_this_sat = 0
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
                        norad_id=sat["norad_id"],
                        alt_shell=sat["alt_shell"],
                        generation=sat["generation"],
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
                n_arcs_this_sat += 1
        print(
            f"  sat {sat['norad_id']} ({sat['generation']}, {sat['alt_shell']} km, "
            f"{sat['n_tles_in_window']} TLEs): {n_arcs_this_sat} arc(s), "
            f"{len(maneuver_epochs)} maneuver epoch(s)",
            file=sys.stderr,
        )
    return specs


def _build_run_spec(
    arc: _ArcSpec,
    mission_path: Path,
    output_root: Path,
    props_by_norad: dict[int, dict],
) -> RunSpec:
    """Build the GMAT RunSpec for a single arc. Mirrors run_sweep._build_run_spec."""
    jd_i, fr_i = _jd_fr_from_epoch(arc.epoch_i)
    r_teme, v_teme = _sgp4_state_teme(arc.line1_i, arc.line2_i, jd_i, fr_i)
    r_mj, v_mj = _teme_to_mj2000(r_teme, v_teme, arc.epoch_i)
    props = props_by_norad[arc.norad_id]
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
                "norad_id": arc.norad_id,
                "alt_shell": arc.alt_shell,
                "generation": arc.generation,
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


def _emit_summary(df: pd.DataFrame, sats: list[dict], out_json: Path) -> dict:
    by_cohort: dict[str, dict] = {}
    for (shell, gen), sub in df.groupby(["alt_shell", "generation"]):
        key = f"{shell}_{gen}"
        per_n = {}
        for n, sub_n in sub.groupby("n"):
            per_n[int(n)] = {
                "n_arcs": int(len(sub_n)),
                "median_dt_h": float(sub_n["actual_dt_h"].median()),
                "median_dr_km": float(sub_n["dr_km"].median()),
                "q25_dr_km": float(sub_n["dr_km"].quantile(0.25)),
                "q75_dr_km": float(sub_n["dr_km"].quantile(0.75)),
            }
        by_cohort[key] = {
            "alt_shell": str(shell),
            "generation": str(gen),
            "norad_id": int(sub["norad_id"].iloc[0]),
            "by_n": per_n,
        }
    summary = {
        "satellites": sats,
        "cohorts": by_cohort,
        "headline_n": min(ARC_OFFSETS_N),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    return summary


def _plot(df: pd.DataFrame, sats: list[dict], out_path: Path) -> None:
    """Per-cohort boxplot grid. Rows = altitude shell, cols = generation.

    Cells absent from the corpus (e.g. 540 km x v2-mini) render as a
    blank panel with a placeholder note so the grid stays legible.
    """
    shells = sorted({s["alt_shell"] for s in sats})
    gens = ["v1.0", "v1.5", "v2-mini"]
    fig, axes = plt.subplots(
        len(shells), len(gens), figsize=(3.6 * len(gens), 2.9 * len(shells)), sharey=True
    )
    rng = np.random.default_rng(seed=0)
    for r, shell in enumerate(shells):
        for c, gen in enumerate(gens):
            ax = axes[r][c] if len(shells) > 1 else axes[c]
            sub = df[(df["alt_shell"] == shell) & (df["generation"] == gen)]
            if sub.empty:
                ax.text(
                    0.5, 0.5, "(no corpus sats)", ha="center", va="center", transform=ax.transAxes
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ns = sorted(sub["n"].unique())
                positions = list(range(len(ns)))
                box_data = [sub.loc[sub["n"] == n, "dr_km"] for n in ns]
                ax.boxplot(box_data, positions=positions, widths=0.6, showfliers=False)
                for i, _n in enumerate(ns):
                    ax.scatter(
                        np.full(len(box_data[i]), i) + rng.uniform(-0.12, 0.12, len(box_data[i])),
                        box_data[i],
                        s=6,
                        alpha=0.35,
                        color="#1f77b4",
                    )
                ax.set_xticks(positions)
                ax.set_xticklabels([f"N={n}" for n in ns])
                ax.set_yscale("log")
                ax.grid(True, which="both", alpha=0.25)
            if r == 0:
                ax.set_title(gen)
            if c == 0:
                ax.set_ylabel(f"{shell} km\n|Δr| (km, log)")
    fig.suptitle(
        "Truth-floor diagnostic — one densest sat per (shell × generation) cohort", fontsize=11
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
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
        help="Restrict to a single NORAD ID. Default is one densest sat per "
        "(alt_shell, generation) cohort.",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    raw = pd.read_parquet(args.raw)
    cache = pd.read_parquet(args.cache)
    if args.norad_id is not None:
        row = cache[cache["norad_id"] == args.norad_id]
        if row.empty:
            raise SystemExit(f"norad_id {args.norad_id} not in corpus cache")
        meta = row.iloc[0]
        sats = [
            {
                "norad_id": int(args.norad_id),
                "generation": str(meta["generation"]),
                "alt_shell": str(meta["alt_shell"]),
                "dry_mass_kg": float(meta["dry_mass_kg"]),
                "drag_area_m2": float(meta["drag_area_m2"]),
                "srp_area_m2": float(meta["srp_area_m2"]),
                "n_tles_in_window": int((raw["norad_id"] == args.norad_id).sum()),
            }
        ]
    else:
        sats = _pick_densest_per_cohort(raw, cache)
    print(f"selected {len(sats)} sat(s) across populated cohort cells", file=sys.stderr)
    for sat in sats:
        print(f"  {sat}", file=sys.stderr)

    specs = _build_arc_specs(raw, sats, ARC_OFFSETS_N)
    if not specs:
        print("no arcs survived the maneuver filter; aborting", file=sys.stderr)
        return 1
    print(
        f"total {len(specs)} arc(s) across {len(sats)} sat(s) and N={ARC_OFFSETS_N}",
        file=sys.stderr,
    )

    output_root = args.output_root.resolve()
    mission_path = args.mission.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    # Fresh manifest each run; this is a one-shot diagnostic, no resume path.
    manifest_path.unlink(missing_ok=True)

    props_by_norad = {s["norad_id"]: s for s in sats}
    run_specs = [_build_run_spec(arc, mission_path, output_root, props_by_norad) for arc in specs]
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

    df = _aggregate(specs, manifest_path)
    parquet_path = args.output_root.parent / "_truth_floor.parquet"
    json_path = args.output_root.parent / "_truth_floor.json"
    png_path = args.output_root.parent / "_truth_floor.png"
    df.to_parquet(parquet_path, index=False)
    summary = _emit_summary(df, sats, json_path)
    _plot(df, sats, png_path)

    print("summary (medians, km):", file=sys.stderr)
    for key, cohort in summary["cohorts"].items():
        per_n_str = "  ".join(
            f"N={n}: {row['median_dr_km']:.2f} (n={row['n_arcs']})"
            for n, row in cohort["by_n"].items()
        )
        print(f"  {key}: {per_n_str}", file=sys.stderr)
    print(f"wrote {parquet_path}, {json_path}, {png_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
