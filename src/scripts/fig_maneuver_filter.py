"""Figure F8 (appendix): SMA-jump histogram with the maneuver-filter threshold.

The Starlink ion-drive maneuver signature is a discontinuity in mean motion
between consecutive TLEs. Plot the distribution of ``|Δa|`` across all pairs
of consecutive TLEs in the raw cache; the threshold separating quiet pairs
from maneuvering pairs should sit cleanly between two modes of the histogram.

Reads a committed `src/static/maneuver_jumps.parquet` (one column,
``abs_da_km``) so CI can build the figure without the gitignored
``src/data/tles_raw.parquet`` (~100 MB). Regenerate the static parquet
with ``python -m sweep.tle_pipeline maneuver-jumps`` whenever the raw
cache changes.

Usage:
    python src/scripts/fig_maneuver_filter.py \\
        --jumps src/static/maneuver_jumps.parquet \\
        --out src/tex/figures/fig_maneuver_filter.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mirrors `sweep.tle_pipeline.DEFAULT_MANEUVER_THRESHOLD_KM`. Inlined so
# the figure script doesn't need the `sweep` namespace package on its
# import path — showyourwork runs figure scripts as bare `python <script>`
# from the repo root, which doesn't add the repo root to sys.path.
DEFAULT_MANEUVER_THRESHOLD_KM = 0.1


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jumps",
        type=Path,
        default=Path("src/static/maneuver_jumps.parquet"),
        help="Static parquet of |Δa| values produced by `tle_pipeline maneuver-jumps`.",
    )
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

    jumps = pd.read_parquet(args.jumps)["abs_da_km"].to_numpy()

    # Floor zeros so log binning works; covers ~0.1 m to ~100 km.
    jumps_nz = np.maximum(jumps, 1e-4)
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
    ax.set_ylabel("count")
    ax.set_title(
        f"Maneuver-filter calibration — {kept:,} quiet, {dropped:,} maneuvering",
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
