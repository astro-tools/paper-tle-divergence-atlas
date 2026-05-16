"""Figure F7: solar-activity modulation of SGP4 staleness coefficient.

For each altitude shell, scatter the per-satellite SGP4 staleness
coefficient ``A_i`` (recovered via ``_style.fit_powerlaw`` from the
satellite's per-bucket ``dr_sgp4_km``) against the satellite's mean
daily-observed F10.7 over its window of starting epochs. Each panel
overlays the ANCOVA fit ``log10 A_i = alpha_gen(i) + beta * F10.7_i``
specified in section 3.8.3 of the manuscript -- a single F10.7 slope
``beta`` per shell, separate intercepts per pooled generation, with
``_h3_regression.py`` providing the point estimates, the bootstrap CI
on ``beta``, and the F-test against the interaction model.

Layout is a one-row, three-column small-multiple ordered by altitude
shell (540 / 550 / 560 km). Within each panel: scatter points coloured
by pooled generation, a per-gen rolling median for visual context, and
one OLS line per gen present (parallel lines sharing the ANCOVA slope
beta but offset by the per-gen intercept). The 95% CI ribbon for each
gen line propagates the bootstrap distribution of (slope, intercept,
gen-offset) onto a predictor grid covering that gen's F10.7 span.

The 540 km shell carries no v2-mini satellites (Table 1 of the
manuscript), so the 540 panel renders a single line. The in-panel
annotation reports the shared beta_hat with its 95% bootstrap CI, the
t-stat p-value for H0: beta=0, the model R^2, the F-test p-value for
the additive-vs-interaction comparison (where applicable), and the
v2-mini intercept offset versus v1.x.

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
    REFERENCE_GEN,
    _ancova_design,
    _bootstrap_ancova,
    _fit_additive,
    _fit_interaction,
    _fit_single_gen,
    _summary_from_fit,
    per_sat_predictors,
)
from _style import ALT_SHELL_ORDER, POOLED_GENERATION_COLORS, apply_rc

ROLLING_WINDOW: Final = 10  # sats per rolling-median window per gen
RIBBON_GRID_POINTS: Final = 80
MIN_SATS_PER_SHELL_FIT: Final = 10
MIN_SATS_PER_GEN_LINE: Final = 5


def _format_p(p: float) -> str:
    """Compact p-value formatting for in-panel annotation."""
    if p < 1e-3:
        return f"p={p:.1e}"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def _ribbon_for_gen(
    samples: dict[str, np.ndarray],
    gen: str,
    x_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap CI ribbon for the OLS line at a given gen.

    Line for `gen` is ``intercept_ref + offset_<gen> + slope * x``, with
    ``offset_<REFERENCE_GEN> = 0`` by construction. Returns the 2.5 /
    97.5 percentile envelopes evaluated on ``x_grid`` in log10 A.
    """
    offsets = (
        samples.get(f"offset_{gen}") if gen != REFERENCE_GEN else np.zeros(len(samples["slope"]))
    )
    if offsets is None:
        # Gen absent from the resample series -- shouldn't happen with
        # the >=10-sat shell floor, but bail gracefully.
        return np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)
    n = min(len(samples["slope"]), len(samples["intercept_ref"]), len(offsets))
    if n == 0:
        return np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)
    lines = (
        samples["intercept_ref"][:n, None]
        + offsets[:n, None]
        + samples["slope"][:n, None] * x_grid[None, :]
    )
    return np.percentile(lines, 2.5, axis=0), np.percentile(lines, 97.5, axis=0)


def _plot_shell_panel(
    ax: plt.Axes,
    shell_df: pd.DataFrame,
) -> dict | None:
    """Render one shell panel; return the ANCOVA summary used in the legend.

    Returns ``None`` if the shell has too few sats to fit; the caller
    annotates the panel accordingly.
    """
    if len(shell_df) < MIN_SATS_PER_SHELL_FIT:
        return None
    gens_present = [g for g in GEN_POOLED_ORDER if (shell_df["gen_pooled"] == g).any()]
    design = _ancova_design(shell_df, HEADLINE_PREDICTOR)
    if len(gens_present) <= 1:
        additive_fit = _fit_single_gen(design)
        interaction_p = None
    else:
        additive_fit = _fit_additive(design)
        f_stat, interaction_p_val, _df = _fit_interaction(design).compare_f_test(additive_fit)
        interaction_p = float(interaction_p_val)
    summary = _summary_from_fit(additive_fit, gens_present)
    samples = _bootstrap_ancova(shell_df, HEADLINE_PREDICTOR, gens_present)

    slope = summary["slope"]
    intercept_ref = summary["intercept_ref"]

    for gen in gens_present:
        gen_df = shell_df[shell_df["gen_pooled"] == gen].sort_values(HEADLINE_PREDICTOR)
        color = POOLED_GENERATION_COLORS.get(gen, "0.4")
        ax.scatter(
            gen_df[HEADLINE_PREDICTOR],
            gen_df["A"],
            s=18,
            color=color,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.3,
            label=f"{gen} (n={len(gen_df)})",
        )
        if len(gen_df) >= ROLLING_WINDOW:
            rolling = gen_df["A"].rolling(ROLLING_WINDOW, center=True).median()
            ax.plot(
                gen_df[HEADLINE_PREDICTOR],
                rolling,
                color=color,
                linewidth=1.3,
                alpha=0.75,
            )
        if len(gen_df) < MIN_SATS_PER_GEN_LINE:
            continue
        x_grid = np.linspace(
            gen_df[HEADLINE_PREDICTOR].min(),
            gen_df[HEADLINE_PREDICTOR].max(),
            RIBBON_GRID_POINTS,
        )
        gen_offset = summary["intercept_offsets"].get(gen, {"estimate": 0.0})["estimate"]
        y_fit = 10 ** (intercept_ref + gen_offset + slope * x_grid)
        lo_log, hi_log = _ribbon_for_gen(samples, gen, x_grid)
        ax.fill_between(x_grid, 10**lo_log, 10**hi_log, color=color, alpha=0.18, linewidth=0)
        ax.plot(x_grid, y_fit, color=color, linewidth=1.8, linestyle="--")

    return {
        "summary": summary,
        "samples": samples,
        "interaction_p": interaction_p,
        "gens_present": gens_present,
    }


def _annotate_panel(ax: plt.Axes, fit_result: dict) -> None:
    summary = fit_result["summary"]
    samples = fit_result["samples"]
    interaction_p = fit_result["interaction_p"]
    slope_samples = samples["slope"]
    ci_lo = float(np.percentile(slope_samples, 2.5))
    ci_hi = float(np.percentile(slope_samples, 97.5))
    lines = [
        f"n = {summary['n_sats']}",
        f"β̂ = {summary['slope']:+.4f}",
        f"95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]",
        f"{_format_p(summary['slope_p'])}   R²={summary['r_squared']:.2f}",
    ]
    if interaction_p is not None:
        lines.append(f"F10.7×gen: {_format_p(interaction_p)}")
    for gen, payload in summary["intercept_offsets"].items():
        lines.append(f"Δα({gen}−{REFERENCE_GEN}) = {payload['estimate']:+.3f}")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
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
            "alpha": 0.88,
        },
    )


def plot_solar_modulation(all_runs: pd.DataFrame, sw_cache: pd.DataFrame, out_path: Path) -> None:
    apply_rc()
    per_sat = per_sat_predictors(all_runs, sw_cache)

    # Symmetric log y-limits across all panels so altitude attenuation
    # is visually comparable between cells.
    a_pos = per_sat["A"][per_sat["A"] > 0]
    y_lo = 10 ** (np.log10(a_pos.min()) - 0.15)
    y_hi = 10 ** (np.log10(a_pos.max()) + 0.15)

    n_shells = len(ALT_SHELL_ORDER)
    fig, axes = plt.subplots(1, n_shells, figsize=(13, 4.6), sharey=True, squeeze=False)

    for c, shell in enumerate(ALT_SHELL_ORDER):
        ax = axes[0, c]
        shell_df = per_sat[per_sat["alt_shell"] == shell]
        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{shell} km")
        ax.set_xlabel("per-sat mean daily F10.7 (sfu)")
        if c == 0:
            ax.set_ylabel("staleness coefficient A  (km at Δt = 1 h)")
        fit_result = _plot_shell_panel(ax, shell_df)
        if fit_result is not None:
            _annotate_panel(ax, fit_result)
        else:
            ax.text(
                0.5,
                0.5,
                f"n_sats={len(shell_df)} below shell fit floor",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="0.45",
                style="italic",
            )

    gen_legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=POOLED_GENERATION_COLORS.get(gen, "0.4"),
            marker="o",
            linestyle="none",
            markersize=6,
            label=gen,
        )
        for gen in GEN_POOLED_ORDER
    ]
    method_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.2",
            linestyle="--",
            linewidth=1.8,
            label="ANCOVA fit per gen (shared slope β)",
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
            linewidth=1.3,
            label=f"per-gen rolling median (window = {ROLLING_WINDOW})",
        ),
    ]
    fig.legend(
        handles=gen_legend_handles + method_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.045),
        frameon=False,
        fontsize=8,
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

    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
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
