"""Figure F2: SGP4 propagation error vs. Δt since epoch.

Small-multiple by pooled generation (sparse generations collapsed via
`_style.pool_sparse_generations`). Within each panel:

  - Light scatter of every (actual_dt, dr_sgp4_km) pair, colored by
    altitude shell — gives the appearance of a continuous curve while
    keeping the analysis bucket-based.
  - Per-bucket median + IQR ribbon by altitude shell, anchored at the
    bucket center.

Log-log axes. Y-limits are computed from the union of `dr_sgp4_km` and
`dr_hifi_km` (via `_style.shared_error_ylim`) so this figure shares an
identical scale with F3 — the F2-vs-F3 visual comparison is what
carries the H2 read.

Usage:
    python src/scripts/fig_sgp4_growth.py \\
        --all-runs outputs/all_runs.parquet \\
        --out src/tex/figures/fig_sgp4_growth.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from _style import (
    ALT_SHELL_COLORS,
    ALT_SHELL_ORDER,
    BUCKET_LABELS,
    BUCKET_SECONDS,
    apply_rc,
    pool_sparse_generations,
    shared_error_ylim,
)


def plot_growth(
    df: pd.DataFrame,
    error_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Render the small-multiple growth figure to `out_path`.

    Shared between F2 (SGP4) and F3 (high-fidelity) via the `error_col`
    argument; both call this with identical y-limits.
    """
    apply_rc()

    df, pool_note = pool_sparse_generations(df)
    pooled_gens = sorted(df["gen_pooled"].unique())
    ylim = shared_error_ylim(df)

    fig, axes = plt.subplots(1, len(pooled_gens), figsize=(4 * len(pooled_gens), 4.2), sharey=True)
    if len(pooled_gens) == 1:
        axes = [axes]

    bucket_hours = [b / 3600.0 for b in BUCKET_SECONDS]

    for ax, gen in zip(axes, pooled_gens, strict=True):
        gen_df = df[df["gen_pooled"] == gen]
        for shell in ALT_SHELL_ORDER:
            sub = gen_df[gen_df["alt_shell"] == shell]
            if sub.empty:
                continue
            color = ALT_SHELL_COLORS[shell]

            ax.scatter(
                sub["actual_dt_sec"] / 3600.0,
                sub[error_col],
                c=[color],
                s=3,
                alpha=0.08,
                linewidths=0,
            )

            hours, meds, q1s, q3s = [], [], [], []
            for bucket in BUCKET_SECONDS:
                cell = sub[sub["target_dt_sec"] == bucket]
                if len(cell) < 3:
                    continue
                hours.append(bucket / 3600.0)
                meds.append(cell[error_col].median())
                q1s.append(cell[error_col].quantile(0.25))
                q3s.append(cell[error_col].quantile(0.75))
            if hours:
                ax.fill_between(hours, q1s, q3s, color=color, alpha=0.25, linewidth=0)
                ax.plot(
                    hours,
                    meds,
                    color=color,
                    marker="o",
                    markersize=5,
                    linewidth=1.5,
                    label=f"alt shell {shell} km (n={len(sub)})",
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(bucket_hours)
        ax.set_xticklabels([BUCKET_LABELS[b] for b in BUCKET_SECONDS])
        ax.set_xlabel("Δt since epoch")
        ax.set_title(f"{gen}  (n={len(gen_df):,})")
        ax.set_ylim(ylim)
        ax.legend(loc="upper left")

    axes[0].set_ylabel(ylabel)

    fig.suptitle(title, fontsize=11)

    if pool_note:
        fig.text(0.5, 0.01, pool_note, ha="center", fontsize=7, style="italic", color="0.35")
        rect = (0.0, 0.04, 1.0, 0.95)
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
        help="Path to the aggregated sweep parquet (built by sweep.aggregate).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/tex/figures/fig_sgp4_growth.pdf"),
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.all_runs)
    plot_growth(
        df,
        error_col="dr_sgp4_km",
        ylabel="|Δr| SGP4 vs. next-TLE truth (km)",
        title="SGP4 propagation error vs. time since epoch",
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
