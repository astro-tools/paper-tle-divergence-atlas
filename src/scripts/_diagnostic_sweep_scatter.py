"""Diagnostic scatter of `actual_dt_sec` vs. `dr_sgp4_km` across the sweep.

Sanity-checks the full sweep output before any manuscript figure work
begins. Surfaces:

  - Monotonic error growth with Δt within each altitude shell.
  - Per-generation behaviour separating v1.0 / v1.5 / v2-mini.
  - Catastrophic outliers (any pair returning > 10,000 km or NaN).

Underscored filename so `showyourwork` does not match it as a
manuscript figure rule. Output PNG goes to `outputs/` which is
gitignored — the bundle reaches Git only via Zenodo at v0.1.0.

Usage:
    python src/scripts/_diagnostic_sweep_scatter.py \\
        --all-runs outputs/all_runs.parquet \\
        --out outputs/_diagnostic_sweep_scatter.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

GENERATION_COLORS = {
    "v1.0": "#1f77b4",
    "v1.5": "#ff7f0e",
    "v2-mini": "#2ca02c",
}


def plot_scatter(all_runs: pd.DataFrame, out_path: Path) -> None:
    shells = sorted(all_runs["alt_shell"].unique())
    fig, axes = plt.subplots(1, len(shells), figsize=(5 * len(shells), 4.5), sharey=True)
    if len(shells) == 1:
        axes = [axes]

    for ax, shell in zip(axes, shells, strict=True):
        shell_df = all_runs[all_runs["alt_shell"] == shell]
        for gen, color in GENERATION_COLORS.items():
            sub = shell_df[shell_df["generation"] == gen]
            if sub.empty:
                continue
            ax.scatter(
                sub["actual_dt_sec"] / 3600.0,
                sub["dr_sgp4_km"],
                s=6,
                alpha=0.4,
                color=color,
                label=f"{gen} (n={len(sub)})",
            )
        ax.set_xlabel("actual Δt (hours)")
        ax.set_yscale("log")
        ax.set_title(f"alt_shell = {shell} km")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("|Δr| SGP4 vs. next-TLE truth (km, log)")
    n_total = len(all_runs)
    n_outliers = int((all_runs["dr_sgp4_km"] > 10_000).sum())
    n_nan = int(all_runs["dr_sgp4_km"].isna().sum())
    fig.suptitle(
        f"Sweep diagnostic — {n_total} runs, {n_outliers} outliers (>10,000 km), {n_nan} NaN",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-runs",
        type=Path,
        default=Path("outputs/all_runs.parquet"),
        help="Path to the aggregated sweep parquet (built by sweep.aggregate)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/_diagnostic_sweep_scatter.png"),
        help="Where to write the diagnostic PNG.",
    )
    args = parser.parse_args()

    all_runs = pd.read_parquet(args.all_runs)
    print(f"loaded {len(all_runs)} runs from {args.all_runs}", file=sys.stderr)
    plot_scatter(all_runs, args.out)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
