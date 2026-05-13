"""Figure F1: Constellation map of the sampled Starlink corpus.

Corner-style scatter matrix over (semi-major axis, eccentricity,
inclination) for the ~500 sampled satellites. The lower triangle shows
pairwise scatters; the diagonal shows marginal histograms. Altitude
shell rides on color (sequential viridis), generation rides on marker
shape (circle / square / triangle).

One representative TLE per satellite — the first epoch present in the
cached corpus. Eccentricity and inclination are parsed from the TLE
`line2` directly via the `sgp4` library since `tles_cache.parquet`
currently only carries the SMA derivative.

Usage:
    python src/scripts/fig_constellation_map.py \\
        --corpus src/static/tles_cache.parquet \\
        --out src/tex/figures/fig_constellation_map.pdf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _style import (
    ALT_SHELL_COLORS,
    ALT_SHELL_ORDER,
    GENERATION_MARKERS,
    GENERATION_ORDER,
    apply_rc,
)
from sgp4.api import Satrec


def _per_sat_elements(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per norad_id with (sma_km, ecc, inc_deg, alt_shell, generation)."""
    first = corpus.sort_values(["norad_id", "epoch_i"]).drop_duplicates(
        subset="norad_id", keep="first"
    )

    eccs = np.empty(len(first), dtype=float)
    incs = np.empty(len(first), dtype=float)
    for idx, (l1, l2) in enumerate(zip(first["line1_i"], first["line2_i"], strict=True)):
        sat = Satrec.twoline2rv(l1, l2)
        eccs[idx] = sat.ecco
        incs[idx] = math.degrees(sat.inclo)

    return pd.DataFrame(
        {
            "norad_id": first["norad_id"].to_numpy(),
            "sma_km": first["sma_i_km"].to_numpy(),
            "ecc": eccs,
            "inc_deg": incs,
            "alt_shell": first["alt_shell"].to_numpy(),
            "generation": first["generation"].to_numpy(),
        }
    )


def _scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> None:
    for shell in ALT_SHELL_ORDER:
        for gen in GENERATION_ORDER:
            sub = df[(df["alt_shell"] == shell) & (df["generation"] == gen)]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col],
                sub[y_col],
                c=[ALT_SHELL_COLORS[shell]],
                marker=GENERATION_MARKERS[gen],
                s=14,
                alpha=0.7,
                edgecolor="white",
                linewidth=0.3,
            )


def _hist_panel(ax: plt.Axes, df: pd.DataFrame, col: str, n_bins: int = 30) -> None:
    for shell in ALT_SHELL_ORDER:
        sub = df[df["alt_shell"] == shell]
        if sub.empty:
            continue
        ax.hist(
            sub[col],
            bins=n_bins,
            color=ALT_SHELL_COLORS[shell],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.3,
        )


def plot_constellation_map(df: pd.DataFrame, out_path: Path) -> None:
    apply_rc()

    cols = [
        ("sma_km", "semi-major axis (km)"),
        ("ecc", "eccentricity"),
        ("inc_deg", "inclination (deg)"),
    ]
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(7.5, 7.5))

    for i, (yc, ylab) in enumerate(cols):
        for j, (xc, xlab) in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                _hist_panel(ax, df, xc)
                ax.set_yticks([])
            elif i > j:
                _scatter_panel(ax, df, xc, yc)
            else:
                ax.set_visible(False)
                continue

            if i == n - 1:
                ax.set_xlabel(xlab)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(ylab)
            elif j != 0:
                ax.set_yticklabels([])

    shell_handles = [
        mlines.Line2D(
            [],
            [],
            color=ALT_SHELL_COLORS[s],
            marker="s",
            linestyle="",
            markersize=7,
            label=f"alt shell {s} km",
        )
        for s in ALT_SHELL_ORDER
    ]
    gen_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.3",
            marker=GENERATION_MARKERS[g],
            linestyle="",
            markersize=7,
            label=g,
        )
        for g in GENERATION_ORDER
    ]
    fig.legend(
        handles=shell_handles + gen_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=2,
    )

    fig.suptitle(
        f"Sampled Starlink corpus — {len(df)} satellites",
        y=0.995,
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("src/static/tles_cache.parquet"),
        help="Path to the sampled corpus parquet (built by sweep.tle_pipeline build).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/tex/figures/fig_constellation_map.pdf"),
    )
    args = parser.parse_args()

    corpus = pd.read_parquet(args.corpus)
    elements = _per_sat_elements(corpus)
    plot_constellation_map(elements, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
