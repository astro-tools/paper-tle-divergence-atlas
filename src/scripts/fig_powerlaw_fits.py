"""Figure F5: power-law fits to the per-bucket staleness curves.

For each (altitude shell × pooled generation) cell, fit
``‖Δr(Δt)‖ ≈ A · Δt_hours^k`` to the per-bucket medians of `dr_sgp4_km`
and `dr_hifi_km`. The fit is a weighted log-log linear regression
(closed-form, robust to the 4-bucket Δt sampling) with weights
proportional to inverse log-IQR — buckets with tight spread carry more
weight than diffuse ones.

Bootstrap 95% CIs (1000 resamples) are computed via
`_style.bootstrap_by_sat`, resampling at the satellite level to respect
within-sat correlation across buckets.

Outputs:
    src/tex/figures/fig_powerlaw_fits.pdf
    src/tex/tables/tab_powerlaw.tex

Usage:
    python src/scripts/fig_powerlaw_fits.py \\
        --all-runs outputs/all_runs.parquet \\
        --out src/tex/figures/fig_powerlaw_fits.pdf \\
        --table-out src/tex/tables/tab_powerlaw.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _style import (
    ALT_SHELL_ORDER,
    apply_rc,
    bootstrap_by_sat,
    fit_powerlaw,
    pool_sparse_generations,
)

PROPAGATORS = (("sgp4", "dr_sgp4_km", "SGP4"), ("hifi", "dr_hifi_km", "high-fid"))
PROPAGATOR_COLORS = {"sgp4": "#1f77b4", "hifi": "#d62728"}


def _estimate_all_cells(df_pooled: pd.DataFrame, gens: list[str]) -> dict[str, float]:
    """Per (shell, gen, propagator) → A and k, keyed for the bootstrap.

    Returns a flat dict with keys ``f"{shell}|{gen}|{prop}|{A|k}"`` so
    `bootstrap_by_sat` can build percentile CIs by key independently.
    """
    out: dict[str, float] = {}
    for shell in ALT_SHELL_ORDER:
        for gen in gens:
            cell = df_pooled[(df_pooled["alt_shell"] == shell) & (df_pooled["gen_pooled"] == gen)]
            if cell.empty:
                continue
            for prop, col, _ in PROPAGATORS:
                try:
                    A, k = fit_powerlaw(cell, col)  # noqa: N806 — math notation
                except ValueError:
                    continue
                out[f"{shell}|{gen}|{prop}|A"] = A
                out[f"{shell}|{gen}|{prop}|k"] = k
    return out


def _bar_panel(
    ax: plt.Axes,
    param: str,
    gen: str,
    shells: list[str],
    point: dict[str, float],
    cis: dict[str, tuple[float, float]],
) -> None:
    """Grouped bar chart for one (param × generation) panel.

    x positions = shells; two bars per shell (SGP4 / hifi), color-coded
    by propagator; error bars from `cis`.
    """
    n_props = len(PROPAGATORS)
    bar_width = 0.8 / n_props
    x_pos = np.arange(len(shells))

    for i, (prop, _, label) in enumerate(PROPAGATORS):
        heights: list[float] = []
        err_lo: list[float] = []
        err_hi: list[float] = []
        present_x: list[float] = []
        for j, shell in enumerate(shells):
            key = f"{shell}|{gen}|{prop}|{param}"
            if key not in point:
                continue
            heights.append(point[key])
            lo, hi = cis.get(key, (float("nan"), float("nan")))
            err_lo.append(point[key] - lo if np.isfinite(lo) else 0.0)
            err_hi.append(hi - point[key] if np.isfinite(hi) else 0.0)
            present_x.append(x_pos[j] + (i - 0.5) * bar_width)

        ax.bar(
            present_x,
            heights,
            width=bar_width,
            yerr=[err_lo, err_hi],
            color=PROPAGATOR_COLORS[prop],
            label=label,
            edgecolor="white",
            linewidth=0.4,
            error_kw={"linewidth": 0.8, "ecolor": "0.3"},
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{s} km" for s in shells])


def plot_powerlaw_fits(
    df: pd.DataFrame,
    out_path: Path,
    n_resamples: int = 1000,
) -> tuple[dict[str, float], dict[str, tuple[float, float]], list[str], str]:
    """Render the F5 small-multiple and return (point, cis, gens, pool_note).

    The point/CI dicts and the pooled-generation list are returned so
    the caller can also dump the LaTeX table without re-running the
    bootstrap.
    """
    apply_rc()
    df_pooled, pool_note = pool_sparse_generations(df)
    gens = sorted(df_pooled["gen_pooled"].unique())

    point, cis = bootstrap_by_sat(
        df_pooled,
        lambda d: _estimate_all_cells(d, gens),
        n_resamples=n_resamples,
    )

    n_rows = len(gens)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(8, 3.0 * n_rows),
        squeeze=False,
    )

    for r, gen in enumerate(gens):
        _bar_panel(axes[r, 0], "A", gen, ALT_SHELL_ORDER, point, cis)
        axes[r, 0].set_ylabel(f"{gen}\nA  (km at Δt = 1 h)")
        axes[r, 0].set_yscale("log")

        _bar_panel(axes[r, 1], "k", gen, ALT_SHELL_ORDER, point, cis)
        axes[r, 1].set_ylabel("k  (power-law exponent)")
        axes[r, 1].axhline(1.0, color="0.6", linestyle=":", linewidth=0.8, zorder=0)

        if r == 0:
            axes[r, 0].set_title("coefficient A")
            axes[r, 1].set_title("exponent k")
            axes[r, 1].legend(loc="upper right")

    fig.suptitle(
        "Power-law fits: ‖Δr‖ ≈ A · Δt$^k$ per altitude shell × generation",
        fontsize=11,
    )

    if pool_note:
        fig.text(0.5, 0.005, pool_note, ha="center", fontsize=7, style="italic", color="0.35")
        rect = (0.0, 0.02, 1.0, 0.96)
    else:
        rect = (0.0, 0.0, 1.0, 0.96)
    fig.tight_layout(rect=rect)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return point, cis, gens, pool_note


def _format_ci(point_val: float, ci: tuple[float, float], precision: int = 2) -> str:
    """Render ``value [lo, hi]`` for the LaTeX table. ``--`` if missing."""
    if not np.isfinite(point_val):
        return "--"
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{point_val:.{precision}g}"
    return f"{point_val:.{precision}g} [{lo:.{precision}g}, {hi:.{precision}g}]"


def write_table(
    point: dict[str, float],
    cis: dict[str, tuple[float, float]],
    gens: list[str],
    out_path: Path,
) -> None:
    """Booktabs table of (shell × gen) → A and k for SGP4 and hifi.

    Written with ``\\input{...}``-able fragment (no \\begin{table} or
    caption); ms.tex wraps it.
    """
    lines = [
        "% Auto-generated by src/scripts/fig_powerlaw_fits.py — do not edit.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{SGP4} & \\multicolumn{2}{c}{high-fid} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "shell & generation & $A$ [95\\% CI] & $k$ [95\\% CI] & $A$ [95\\% CI] & $k$ [95\\% CI] \\\\",
        "\\midrule",
    ]
    for shell in ALT_SHELL_ORDER:
        for gen in gens:
            cells: list[str] = [f"{shell} km", gen]
            seen = False
            for prop, _, _ in PROPAGATORS:
                for param in ("A", "k"):
                    key = f"{shell}|{gen}|{prop}|{param}"
                    pv = point.get(key, float("nan"))
                    cells.append(_format_ci(pv, cis.get(key, (float("nan"), float("nan")))))
                    if np.isfinite(pv):
                        seen = True
            if seen:
                lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-runs",
        type=Path,
        default=Path("outputs/all_runs.parquet"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/tex/figures/fig_powerlaw_fits.pdf"),
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=Path("src/tex/tables/tab_powerlaw.tex"),
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=1000,
        help="Bootstrap resamples (default 1000).",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.all_runs)
    point, cis, gens, _ = plot_powerlaw_fits(df, args.out, n_resamples=args.n_resamples)
    write_table(point, cis, gens, args.table_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
