"""Figure F7: solar-activity modulation of SGP4 staleness coefficient.

For each (altitude shell, pooled generation) cell, scatter the
per-satellite SGP4 staleness coefficient ``A_i`` (the same per-sat
power-law fit as before, recovered via ``_style.fit_powerlaw`` from
``dr_sgp4_km`` against the four staleness buckets) against the
satellite's mean daily-observed F10.7 over its window of starting
epochs. Each panel overlays an OLS line ``log10 A = alpha + beta *
F10.7`` with a 95% CI ribbon (1,000-resample satellite-level bootstrap
per ``_h3_regression.py``) and reports ``beta_hat`` with its bootstrap
CI and the t-stat p-value in the legend. Addresses Hypothesis H3
(error growth under drag-driven conditions scales with solar EUV proxy).

The figure is laid out as a 2 x 3 small-multiple: rows are pooled
generation (v1.x, v2-mini), columns are altitude shell (540, 550, 560
km). The 540 km x v2-mini cell is empty in the corpus -- Table 1 of
the manuscript shows zero v2-mini satellites at that altitude -- and
the panel is annotated rather than fit. Stratifying within shell by
generation is load-bearing: a shell-pooled fit at 550 or 560 km would
be confounded by the Simpson-paradox composition shift documented in
the ``_h3_regression`` module docstring and the matching subsection of
section 3.7 of the manuscript.

The per-sat fit requires at least ``MIN_BUCKETS_PER_SAT`` distinct
buckets with three or more pairs each -- sats that fall below that
floor are dropped, with the count reported in the figure footer.

Usage:
    python src/scripts/fig_solar_modulation.py \\
        --all-runs outputs/all_runs.parquet \\
        --sw-cache src/static/sw_cache.parquet \\
        --out src/tex/figures/fig_solar_modulation.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _h3_regression import (
    GEN_POOLED_ORDER,
    HEADLINE_PREDICTOR,
    MIN_BUCKETS_PER_SAT,
    _bootstrap_samples,
    _fit_ols,
    per_sat_predictors,
)
from _style import ALT_SHELL_ORDER, POOLED_GENERATION_COLORS, apply_rc

ROLLING_WINDOW: Final = 10  # sats per rolling-median window in each panel
RIBBON_GRID_POINTS: Final = 80
MIN_SATS_FOR_FIT: Final = 5


def _evaluate_ribbon(
    intercepts: np.ndarray,
    slopes: np.ndarray,
    x_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """95% percentile envelope of the resampled OLS lines on ``x_grid``."""
    lines = intercepts[:, None] + slopes[:, None] * x_grid[None, :]
    lo = np.percentile(lines, 2.5, axis=0)
    hi = np.percentile(lines, 97.5, axis=0)
    return lo, hi


def _plot_panel(
    ax: plt.Axes,
    cell: pd.DataFrame,
    color: str,
) -> tuple[float, float, float, float, int] | None:
    """Render one (shell, gen) panel; return ``(beta, p, ci_lo, ci_hi, n)``.

    Returns ``None`` when the cell has too few sats for a defensible
    fit; the panel still receives a scatter so the absence of v1.0 / 540
    / v2-mini doesn't leave a blank in the grid except where the corpus
    is genuinely empty.
    """
    cell_sorted = cell.sort_values(HEADLINE_PREDICTOR)
    ax.scatter(
        cell_sorted[HEADLINE_PREDICTOR],
        cell_sorted["A"],
        s=18,
        color=color,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.3,
    )
    if len(cell_sorted) >= ROLLING_WINDOW:
        rolling = cell_sorted["A"].rolling(ROLLING_WINDOW, center=True).median()
        ax.plot(
            cell_sorted[HEADLINE_PREDICTOR],
            rolling,
            color=color,
            linewidth=1.4,
            alpha=0.75,
        )
    if len(cell) < MIN_SATS_FOR_FIT:
        return None

    fit = _fit_ols(cell, HEADLINE_PREDICTOR)
    intercepts, slopes = _bootstrap_samples(cell, HEADLINE_PREDICTOR)
    x_grid = np.linspace(
        cell[HEADLINE_PREDICTOR].min(),
        cell[HEADLINE_PREDICTOR].max(),
        RIBBON_GRID_POINTS,
    )
    y_fit = 10 ** (fit["intercept"] + fit["slope"] * x_grid)
    ribbon_lo_log, ribbon_hi_log = _evaluate_ribbon(intercepts, slopes, x_grid)
    ax.fill_between(
        x_grid, 10**ribbon_lo_log, 10**ribbon_hi_log, color="0.4", alpha=0.18, linewidth=0
    )
    ax.plot(x_grid, y_fit, color="0.2", linewidth=1.6, linestyle="--")

    ci_lo = float(np.percentile(slopes, 2.5))
    ci_hi = float(np.percentile(slopes, 97.5))
    return fit["slope"], fit["slope_p"], ci_lo, ci_hi, fit["n_sats"]


def _format_p(p: float) -> str:
    """Compact p-value formatting for in-panel annotation."""
    if p < 1e-3:
        return f"p={p:.1e}"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def plot_solar_modulation(all_runs: pd.DataFrame, sw_cache: pd.DataFrame, out_path: Path) -> None:
    apply_rc()
    per_sat = per_sat_predictors(all_runs, sw_cache)

    # Symmetric log y-limits across all panels so altitude attenuation
    # is visually comparable between cells.
    a_pos = per_sat["A"][per_sat["A"] > 0]
    y_lo = 10 ** (np.log10(a_pos.min()) - 0.15)
    y_hi = 10 ** (np.log10(a_pos.max()) + 0.15)

    n_rows = len(GEN_POOLED_ORDER)
    n_cols = len(ALT_SHELL_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11, 6.5),
        sharex=False,
        sharey=True,
        squeeze=False,
    )

    for r, gen in enumerate(GEN_POOLED_ORDER):
        for c, shell in enumerate(ALT_SHELL_ORDER):
            ax = axes[r, c]
            cell = per_sat[(per_sat["alt_shell"] == shell) & (per_sat["gen_pooled"] == gen)]
            color = POOLED_GENERATION_COLORS.get(gen, "0.4")
            if cell.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no satellites in corpus",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.45",
                    style="italic",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"{shell} km")
                if c == 0:
                    ax.set_ylabel(f"{gen}\nA  (km at Δt = 1 h)")
                continue
            ax.set_yscale("log")
            ax.set_ylim(y_lo, y_hi)
            fit_summary = _plot_panel(ax, cell, color)
            if fit_summary is not None:
                beta, p, ci_lo, ci_hi, n_fit = fit_summary
                ax.text(
                    0.02,
                    0.98,
                    (
                        f"n = {n_fit}\n"
                        f"β̂ = {beta:+.4f}\n"
                        f"95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]\n"
                        f"{_format_p(p)}"
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="0.15",
                    family="monospace",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.7",
                        "boxstyle": "round,pad=0.25",
                        "alpha": 0.85,
                    },
                )
            if r == 0:
                ax.set_title(f"{shell} km")
            if c == 0:
                ax.set_ylabel(f"{gen}\nA  (km at Δt = 1 h)")
            if r == n_rows - 1:
                ax.set_xlabel("mean F10.7 (sfu)")

    handles = [
        mlines.Line2D(
            [],
            [],
            color="0.2",
            linestyle="--",
            linewidth=1.6,
            label="OLS fit (full-sample)",
        ),
        mlines.Line2D(
            [],
            [],
            color="0.4",
            linestyle="-",
            linewidth=8,
            alpha=0.18,
            label="95% CI (1,000-resample sat-level bootstrap)",
        ),
        mlines.Line2D(
            [],
            [],
            color="0.4",
            linestyle="-",
            linewidth=1.4,
            label=f"rolling median (window = {ROLLING_WINDOW})",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.045),
        frameon=False,
    )

    n_per_sat_dropped = per_sat_predictors_dropped_count(all_runs, sw_cache, per_sat)
    footer_bits: list[str] = [
        "predictor: per-sat mean of daily-observed F10.7 over the sat's starting-epoch window",
        f"per-sat fit floor: ≥ {MIN_BUCKETS_PER_SAT} usable buckets",
    ]
    if n_per_sat_dropped:
        footer_bits.append(f"{n_per_sat_dropped} sat(s) dropped at per-sat fit stage")
    fig.text(
        0.5,
        0.005,
        "; ".join(footer_bits),
        ha="center",
        fontsize=7,
        style="italic",
        color="0.35",
    )

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def per_sat_predictors_dropped_count(
    all_runs: pd.DataFrame,
    sw_cache: pd.DataFrame,
    fitted: pd.DataFrame,
) -> int:
    """How many (norad_id, alt_shell) sats survived the bucket-count floor."""
    candidates = all_runs[["alt_shell", "norad_id"]].drop_duplicates()
    survived = fitted[["alt_shell", "norad_id"]].drop_duplicates()
    return int(len(candidates) - len(survived))


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-runs",
        type=Path,
        default=Path("outputs/all_runs.parquet"),
    )
    parser.add_argument(
        "--sw-cache",
        type=Path,
        default=Path("src/static/sw_cache.parquet"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/tex/figures/fig_solar_modulation.pdf"),
    )
    args = parser.parse_args()

    all_runs = pd.read_parquet(args.all_runs)
    sw_cache = pd.read_parquet(args.sw_cache)
    plot_solar_modulation(all_runs, sw_cache, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
