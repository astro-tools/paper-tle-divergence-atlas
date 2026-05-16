"""Figure F6: along-track / cross-track / radial error decomposition.

Three side-by-side panels (along / cross / radial). Within each panel,
per-bucket median + IQR ribbon vs. Δt, with SGP4 (solid) and high-fid
(dashed) on the same axes for each pooled generation. The expectation
(borne out by the F2 / F3 reveal) is that along-track dominates at long
Δt; cross-track and radial are an order of magnitude smaller.

Uses ``|component|`` (absolute value) rather than the signed component
so the log y-axis stays meaningful; the sign carries through to the
joint regression in ``_regression.py`` if needed.

Usage:
    python src/scripts/fig_error_decomposition.py \\
        --all-runs outputs/all_runs.parquet \\
        --out src/tex/figures/fig_error_decomposition.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from _style import (
    BUCKET_LABELS,
    BUCKET_SECONDS,
    POOLED_GENERATION_COLORS,
    apply_rc,
    pool_sparse_generations,
)

COMPONENTS = (
    ("along", "along-track"),
    ("cross", "cross-track"),
    ("radial", "radial"),
)
PROPAGATORS = (
    ("sgp4", "dr_sgp4_{c}_km", "SGP4", "-"),
    ("hifi", "dr_hifi_{c}_km", "high-fid", "--"),
)
MIN_PAIRS_PER_BUCKET = 3

# Circular-orbit speed at the 550 km central shell:
# v_c = sqrt(mu / (R_E + h)) with mu = 398600.4418 km^3/s^2 and
# R_E + h = 6928.137 km. Used to convert the along-track km axis to
# seconds of orbital phase on F6's secondary y-axis.
ALONG_TRACK_REF_SPEED_KMPS = 7.59


def _component_curve(
    df: pd.DataFrame,
    col: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Per-bucket (hours, median, q1, q3) of ``|df[col]|``.

    Buckets with fewer than `MIN_PAIRS_PER_BUCKET` pairs are skipped.
    """
    hours: list[float] = []
    meds: list[float] = []
    q1s: list[float] = []
    q3s: list[float] = []
    abs_col = df[col].abs()
    for bucket in BUCKET_SECONDS:
        mask = df["target_dt_sec"] == bucket
        cell = abs_col[mask]
        cell = cell[cell > 0]
        if len(cell) < MIN_PAIRS_PER_BUCKET:
            continue
        hours.append(bucket / 3600.0)
        meds.append(float(cell.median()))
        q1s.append(float(cell.quantile(0.25)))
        q3s.append(float(cell.quantile(0.75)))
    return hours, meds, q1s, q3s


def plot_error_decomposition(df: pd.DataFrame, out_path: Path) -> None:
    apply_rc()

    df, pool_note = pool_sparse_generations(df)
    pooled_gens = sorted(df["gen_pooled"].unique())
    gen_colors = {g: POOLED_GENERATION_COLORS.get(g, f"C{i}") for i, g in enumerate(pooled_gens)}

    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(4 * len(COMPONENTS), 4.2), sharey=True)
    bucket_hours = [b / 3600.0 for b in BUCKET_SECONDS]

    for ax, (component, title) in zip(axes, COMPONENTS, strict=True):
        for gen in pooled_gens:
            gen_df = df[df["gen_pooled"] == gen]
            if gen_df.empty:
                continue
            color = gen_colors[gen]
            for _, col_template, _, linestyle in PROPAGATORS:
                col = col_template.format(c=component)
                hours, meds, q1s, q3s = _component_curve(gen_df, col)
                if not hours:
                    continue
                if linestyle == "-":
                    ax.fill_between(hours, q1s, q3s, color=color, alpha=0.20, linewidth=0)
                ax.plot(
                    hours,
                    meds,
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    markersize=4,
                    linewidth=1.4,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(bucket_hours)
        ax.set_xticklabels([BUCKET_LABELS[b] for b in BUCKET_SECONDS])
        ax.set_xlabel("Δt since epoch")
        ax.set_title(title)

        if component == "along":
            sec_ax = ax.secondary_yaxis(
                "right",
                functions=(
                    lambda km: km / ALONG_TRACK_REF_SPEED_KMPS,
                    lambda s: s * ALONG_TRACK_REF_SPEED_KMPS,
                ),
            )
            sec_ax.set_ylabel(f"timing equivalent at {ALONG_TRACK_REF_SPEED_KMPS} km/s (s)")

    axes[0].set_ylabel("|component of Δr| (km)")

    handles: list[mlines.Line2D] = []
    for gen in pooled_gens:
        handles.append(
            mlines.Line2D([], [], color=gen_colors[gen], marker="o", linestyle="-", label=gen)
        )
    handles.append(mlines.Line2D([], [], color="0.4", linestyle="-", label="SGP4"))
    handles.append(mlines.Line2D([], [], color="0.4", linestyle="--", label="high-fid"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=1)

    fig.suptitle(
        "Error decomposition: along / cross / radial vs. Δt",
        fontsize=11,
    )

    if pool_note:
        fig.text(0.5, 0.005, pool_note, ha="center", fontsize=7, style="italic", color="0.35")
        rect = (0.0, 0.02, 1.0, 0.95)
    else:
        rect = (0.0, 0.0, 1.0, 0.95)
    fig.tight_layout(rect=rect)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-runs",
        type=Path,
        default=Path("outputs/all_runs.parquet"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/tex/figures/fig_error_decomposition.pdf"),
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.all_runs)
    plot_error_decomposition(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
