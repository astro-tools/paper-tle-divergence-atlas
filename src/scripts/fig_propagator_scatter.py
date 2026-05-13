"""Figure F4: SGP4 vs. high-fidelity error at fixed Δt buckets.

Small-multiple grid: rows = Δt bucket (6h / 1d / 3d / 7d), columns =
altitude shell (540 / 550 / 560 km). Within each panel: scatter of
`dr_sgp4_km` (x) vs `dr_hifi_km` (y) on log-log axes, colored by pooled
generation, with the y = x reference line drawn so points below the
line are pairs where the high-fidelity propagator beat SGP4.

Generations pooled via `_style.pool_sparse_generations` so the same
pooling convention applies across F2 / F3 / F4.

Usage:
    python src/scripts/fig_propagator_scatter.py \\
        --all-runs outputs/all_runs.parquet \\
        --out src/tex/figures/fig_propagator_scatter.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _style import (
    ALT_SHELL_ORDER,
    BUCKET_LABELS,
    BUCKET_SECONDS,
    POOLED_GENERATION_COLORS,
    apply_rc,
    pool_sparse_generations,
    shared_error_ylim,
)


def plot_propagator_scatter(df: pd.DataFrame, out_path: Path) -> None:
    apply_rc()

    df, pool_note = pool_sparse_generations(df)
    pooled_gens = sorted(df["gen_pooled"].unique())
    # Fall back to per-generation default colors when pooling did not fire.
    gen_colors = {g: POOLED_GENERATION_COLORS.get(g, f"C{i}") for i, g in enumerate(pooled_gens)}

    lim_lo, lim_hi = shared_error_ylim(df)

    n_rows, n_cols = len(BUCKET_SECONDS), len(ALT_SHELL_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        sharex=True,
        sharey=True,
    )

    diag = np.array([lim_lo, lim_hi])

    for i, bucket in enumerate(BUCKET_SECONDS):
        for j, shell in enumerate(ALT_SHELL_ORDER):
            ax = axes[i, j]
            cell = df[(df["target_dt_sec"] == bucket) & (df["alt_shell"] == shell)]
            for gen in pooled_gens:
                sub = cell[cell["gen_pooled"] == gen]
                if sub.empty:
                    continue
                ax.scatter(
                    sub["dr_sgp4_km"],
                    sub["dr_hifi_km"],
                    c=[gen_colors[gen]],
                    s=5,
                    alpha=0.5,
                    linewidths=0,
                )

            ax.plot(diag, diag, color="0.4", linestyle="--", linewidth=0.8, zorder=0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.set_aspect("equal")

            if i == 0:
                ax.set_title(f"alt shell {shell} km")
            if i == n_rows - 1:
                ax.set_xlabel("|Δr| SGP4 (km)")
            if j == 0:
                ax.set_ylabel(f"Δt = {BUCKET_LABELS[bucket]}\n|Δr| high-fid (km)")

    handles = [
        mlines.Line2D([], [], color=gen_colors[g], marker="o", linestyle="", markersize=6, label=g)
        for g in pooled_gens
    ]
    handles.append(mlines.Line2D([], [], color="0.4", linestyle="--", linewidth=1.0, label="y = x"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=1)

    fig.suptitle(
        "SGP4 vs. high-fidelity error at fixed Δt buckets",
        fontsize=11,
        y=0.995,
    )
    if pool_note:
        fig.text(0.5, 0.005, pool_note, ha="center", fontsize=7, style="italic", color="0.35")
        rect = (0.0, 0.02, 1.0, 0.97)
    else:
        rect = (0.0, 0.0, 1.0, 0.97)
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
        help="Path to the aggregated sweep parquet (built by sweep.aggregate).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/tex/figures/fig_propagator_scatter.pdf"),
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.all_runs)
    plot_propagator_scatter(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
