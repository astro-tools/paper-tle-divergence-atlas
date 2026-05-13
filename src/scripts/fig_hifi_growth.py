"""Figure F3: High-fidelity propagation error vs. Δt since epoch.

Identical layout and axes to F2 (`fig_sgp4_growth.py`) — small-multiple
by pooled generation, per-bucket median + IQR ribbon by altitude shell,
log-log axes, shared y-limits computed from the union of
`dr_sgp4_km` and `dr_hifi_km` via `_style.shared_error_ylim`.

Side-by-side with F2 this is the H2 negative-result reveal: investing
in a high-fidelity force model does not necessarily beat SGP4 for
operator-updated truth at megaconstellation altitudes.

Usage:
    python src/scripts/fig_hifi_growth.py \\
        --all-runs outputs/all_runs.parquet \\
        --out src/tex/figures/fig_hifi_growth.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from fig_sgp4_growth import plot_growth


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
        default=Path("src/tex/figures/fig_hifi_growth.pdf"),
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.all_runs)
    plot_growth(
        df,
        error_col="dr_hifi_km",
        ylabel="|Δr| high-fid vs. next-TLE truth (km)",
        title="High-fidelity propagation error vs. time since epoch",
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
