"""Appendix figure: selection-effect diagnostic for the ±2 h pair-matching tolerance.

Two panels:

(a) Histogram of inter-TLE intervals across all per-sat consecutive
    pairs in the 501-sat corpus. Log-x. Vertical lines mark the
    matching tolerance window edge (4 h, since ±2 h is 4 h wide) and
    a 12 h reference. The bulk of the distribution clusters near the
    median ~4.8 h, with a right tail extending past 24 h — the
    quantitative basis for the §3.3 claim that the per-Δt unmatched
    fraction is Poisson-window-dominated, not tracking-gap-dominated.

(b) Empirical CDF of the per-sat longest-gap-in-window distribution
    across the 501 corpus sats. Vertical lines mark the four Δt
    targets ({6, 24, 72, 168} h). The 95th-percentile worst per-sat
    gap sits well inside the longest Δt target, so no sat is
    systematically excluded by the matching tolerance — the
    quantitative basis for the §3.3 claim that the tracking-gap-bias
    is bounded.

Reads a committed `src/static/selection_stats.parquet` (long form:
columns `kind`, `value_h`; `kind ∈ {interval_h, longest_per_sat_h}`)
so CI can build the figure without the gitignored
`src/data/tles_raw.parquet`. Regenerate the static parquet with
`python -m sweep.tle_pipeline selection-stats` whenever the raw
cache changes.

Usage:
    python src/scripts/fig_selection_effect.py \\
        --stats src/static/selection_stats.parquet \\
        --out src/tex/figures/fig_selection_effect.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Δt target offsets in hours, mirrors `sweep.tle_pipeline.DEFAULT_TARGET_DTS_SEC`.
# Inlined so the figure script doesn't need the `sweep` package on import path.
TARGET_DTS_H: tuple[float, ...] = (6.0, 24.0, 72.0, 168.0)
TOLERANCE_FULL_WINDOW_H: float = 4.0  # ±2 h is a 4 h-wide acceptance window


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("src/static/selection_stats.parquet"),
        help="Static parquet produced by `tle_pipeline selection-stats`.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/tex/figures/fig_selection_effect.pdf"),
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.stats)
    intervals_h = df.loc[df["kind"] == "interval_h", "value_h"].to_numpy()
    longest_per_sat_h = df.loc[df["kind"] == "longest_per_sat_h", "value_h"].to_numpy()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ----- panel (a): inter-TLE interval histogram, log-x -----
    bins = np.logspace(np.log10(0.5), np.log10(max(intervals_h.max(), 200.0)), 50)
    ax_a.hist(
        intervals_h,
        bins=bins,
        color="C0",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_a.axvline(
        TOLERANCE_FULL_WINDOW_H,
        color="C3",
        linestyle="--",
        linewidth=1.4,
        label=f"±2 h tolerance window ({TOLERANCE_FULL_WINDOW_H:.0f} h)",
    )
    ax_a.axvline(
        12.0,
        color="C2",
        linestyle=":",
        linewidth=1.4,
        label="12 h reference",
    )
    median_h = float(np.median(intervals_h))
    p99_h = float(np.quantile(intervals_h, 0.99))
    ax_a.set_xscale("log")
    ax_a.set_xlabel("inter-TLE interval (h, log)")
    ax_a.set_ylabel("count")
    ax_a.set_title(
        f"(a) Inter-TLE intervals — n={len(intervals_h):,}, "
        f"median {median_h:.1f} h, p99 {p99_h:.1f} h",
        fontsize=10,
    )
    ax_a.legend(fontsize=8, loc="upper right")
    ax_a.grid(True, which="both", alpha=0.25)

    # ----- panel (b): per-sat longest-gap ECDF, log-x -----
    sorted_longest = np.sort(longest_per_sat_h)
    n_sats = len(sorted_longest)
    cdf = np.arange(1, n_sats + 1) / n_sats
    ax_b.step(sorted_longest, cdf, where="post", color="C0", linewidth=1.5)
    palette = ["C3", "C1", "C2", "C4"]
    for dt_h, color in zip(TARGET_DTS_H, palette, strict=True):
        ax_b.axvline(
            dt_h,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"Δt = {dt_h:.0f} h",
        )
    p95_longest = float(np.quantile(longest_per_sat_h, 0.95))
    median_longest = float(np.median(longest_per_sat_h))
    ax_b.set_xscale("log")
    ax_b.set_xlim(left=max(1.0, float(sorted_longest.min()) * 0.8))
    ax_b.set_ylim(0.0, 1.02)
    ax_b.set_xlabel("per-sat longest within-window gap (h, log)")
    ax_b.set_ylabel("ECDF across corpus sats")
    ax_b.set_title(
        f"(b) Worst per-sat tracking gap — n={n_sats} sats, "
        f"median {median_longest:.1f} h, p95 {p95_longest:.1f} h",
        fontsize=10,
    )
    ax_b.legend(fontsize=8, loc="lower right")
    ax_b.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
