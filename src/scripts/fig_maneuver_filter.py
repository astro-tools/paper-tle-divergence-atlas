"""Figure F8 (appendix): SMA-jump histogram with the maneuver-filter threshold.

The Starlink ion-drive maneuver signature is a discontinuity in mean motion
between consecutive TLEs. Plot the distribution of |Δa| across all pairs in
the raw cache; the threshold separating quiet pairs from maneuvering pairs
should sit cleanly between two modes of the histogram.

This figure is not wired into ms.tex until Day 6 (issue #5). Day 2 commits
the script so the calibration of the 100-m default threshold is reviewable
alongside the pipeline.

Usage (Day 2):
    python src/scripts/fig_maneuver_filter.py \\
        --raw src/data/tles_raw.parquet \\
        --out src/tex/figures/fig_maneuver_filter.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sweep.tle_pipeline import DEFAULT_MANEUVER_THRESHOLD_KM, build_pairs


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("src/data/tles_raw.parquet"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/tex/figures/fig_maneuver_filter.pdf"),
    )
    parser.add_argument(
        "--threshold-km",
        type=float,
        default=DEFAULT_MANEUVER_THRESHOLD_KM,
        help="Vertical threshold to annotate.",
    )
    args = parser.parse_args()

    raw = pd.read_parquet(args.raw)
    pairs = build_pairs(raw)

    # Symmetric log-binned histogram covers ~1 m to ~100 km of SMA jump.
    jumps = pairs["sma_jump_km"].to_numpy()
    jumps_nz = np.maximum(jumps, 1e-4)  # floor zeros so log binning works
    bins = np.logspace(-4, 2, 60)

    kept = int((jumps <= args.threshold_km).sum())
    dropped = int((jumps > args.threshold_km).sum())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(jumps_nz, bins=bins, color="C0", alpha=0.7, edgecolor="white", linewidth=0.4)
    ax.axvline(
        args.threshold_km,
        color="C3",
        linestyle="--",
        linewidth=1.5,
        label=f"{args.threshold_km * 1000:.0f} m threshold",
    )
    ax.set_xscale("log")
    ax.set_xlabel("|Δa| between consecutive TLEs (km)")
    ax.set_ylabel("pair count")
    ax.set_title(
        f"Maneuver-filter calibration — {kept:,} pairs kept, {dropped:,} dropped",
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
