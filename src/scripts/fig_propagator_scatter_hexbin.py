"""Figure F4-appendix: hexbin density variant of the propagator scatter.

Companion to ``fig_propagator_scatter.py``: same 12-panel grid
(rows = Δt bucket, columns = altitude shell) but each panel is a
log-binned hexbin density of (|Δr| SGP4, |Δr| high-fid) pairs rather
than a per-pair scatter. The main-body F4 keeps the scatter — pairs
*above* vs. *below* the y=x diagonal is the structural claim — and
this appendix companion makes the joint-density structure
(horizontal-band saturation at low altitude / long Δt, v2-mini wedge
below the diagonal) easier to read where individual points overplot.

Pooling and axis limits match the main F4 via the same ``_style``
helpers so the two figures are visually superimposable.

Usage:
    python src/scripts/fig_propagator_scatter_hexbin.py \\
        --all-runs outputs/all_runs.parquet \\
        --out src/tex/figures/fig_propagator_scatter_hexbin.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _style import (
    ALT_SHELL_ORDER,
    BUCKET_LABELS,
    BUCKET_SECONDS,
    apply_rc,
    pool_sparse_generations,
    shared_error_ylim,
)


def plot_propagator_scatter_hexbin(df: pd.DataFrame, out_path: Path) -> None:
    apply_rc()

    df, pool_note = pool_sparse_generations(df)
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
    hex_artists = []
    log_extent = (np.log10(lim_lo), np.log10(lim_hi), np.log10(lim_lo), np.log10(lim_hi))

    for i, bucket in enumerate(BUCKET_SECONDS):
        for j, shell in enumerate(ALT_SHELL_ORDER):
            ax = axes[i, j]
            cell = df[(df["target_dt_sec"] == bucket) & (df["alt_shell"] == shell)]

            if not cell.empty:
                # Hexbin on log-space coordinates; positive Δr only.
                x = cell["dr_sgp4_km"]
                y = cell["dr_hifi_km"]
                mask = (x > 0) & (y > 0)
                if mask.any():
                    hb = ax.hexbin(
                        np.log10(x[mask]),
                        np.log10(y[mask]),
                        gridsize=30,
                        cmap="viridis",
                        mincnt=1,
                        extent=log_extent,
                        linewidths=0,
                    )
                    hex_artists.append(hb)

            # y=x in the same log space, drawn on top.
            ax.plot(
                np.log10(diag),
                np.log10(diag),
                color="0.9",
                linestyle="--",
                linewidth=0.8,
                zorder=10,
            )
            ax.set_xlim(np.log10(lim_lo), np.log10(lim_hi))
            ax.set_ylim(np.log10(lim_lo), np.log10(lim_hi))
            ax.set_aspect("equal")

            # Manual log-scale tick labels at the powers of 10 we care about.
            decades = np.arange(int(np.ceil(np.log10(lim_lo))), int(np.floor(np.log10(lim_hi))) + 1)
            ax.set_xticks(decades)
            ax.set_yticks(decades)
            ax.set_xticklabels([f"$10^{{{d}}}$" for d in decades])
            ax.set_yticklabels([f"$10^{{{d}}}$" for d in decades])

            if not cell.empty:
                wins = (cell["dr_hifi_km"] < cell["dr_sgp4_km"]).mean() * 100.0
                ax.text(
                    0.97,
                    0.04,
                    f"hi-fid: {wins:.0f}%",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color="0.92",
                    bbox={
                        "facecolor": "black",
                        "edgecolor": "none",
                        "alpha": 0.55,
                        "pad": 1.5,
                    },
                )

            if i == 0:
                ax.set_title(f"alt shell {shell} km")
            if i == n_rows - 1:
                ax.set_xlabel("|Δr| SGP4 (km)")
            if j == 0:
                ax.set_ylabel(f"Δt = {BUCKET_LABELS[bucket]}\n|Δr| high-fid (km)")

    # Shared colour bar to the right of the grid; use the last hexbin
    # for the normalisation reference (matplotlib hexbin auto-normalises
    # per-axis, so this bar is illustrative — quantitative comparison
    # across panels uses the win-fraction annotation, not the colour
    # depth).
    if hex_artists:
        cbar_ax = fig.add_axes((0.93, 0.12, 0.012, 0.76))
        fig.colorbar(hex_artists[-1], cax=cbar_ax, label="pairs per hex (last panel scale)")

    fig.suptitle(
        "SGP4 vs. high-fidelity error at fixed Δt buckets (hexbin density)",
        fontsize=11,
        y=0.995,
    )
    if pool_note:
        fig.text(0.5, 0.005, pool_note, ha="center", fontsize=7, style="italic", color="0.35")
        rect = (0.0, 0.02, 0.92, 0.97)
    else:
        rect = (0.0, 0.0, 0.92, 0.97)
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
        default=Path("src/tex/figures/fig_propagator_scatter_hexbin.pdf"),
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.all_runs)
    plot_propagator_scatter_hexbin(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
