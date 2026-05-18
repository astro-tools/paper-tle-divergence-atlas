"""Figure F5: per-pair power-law fits to the staleness data.

For each (altitude shell × pooled generation) cell, fit
``‖Δr(Δt)‖ ≈ A · Δt_hours^k`` directly to the per-pair points (using
each pair's measured `actual_dt_sec`, not the nominal bucket centre)
via the §3.7.1 unweighted log-log OLS estimator
`_style.fit_powerlaw_perpair`. Bootstrap 95% CIs (1000 resamples) are
computed via `_style.bootstrap_by_sat`, resampling at the satellite
level so within-sat correlation across the four buckets is preserved.

Alongside the bar chart this script emits the booktabs companion
``tab_powerlaw.tex`` carrying the same point + CI estimates plus the
per-cell coefficient of determination ``R²`` and likelihood-ratio
p-values for the two physically interpretable nulls ``k = 1`` (constant
mean-motion bias, linear along-track growth) and ``k = 2`` (constant
along-track acceleration, quadratic growth).

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
    fit_powerlaw_perpair,
    pool_sparse_generations,
)

PROPAGATORS = (("sgp4", "dr_sgp4_km", "SGP4"), ("hifi", "dr_hifi_km", "high-fid"))
PROPAGATOR_COLORS = {"sgp4": "#1f77b4", "hifi": "#d62728"}
# Statistics returned by the per-pair fit. A and k drive the bar chart;
# all four are written into the booktabs table.
FIT_KEYS = ("A", "k", "r_squared", "p_lrt_k1", "p_lrt_k2")


def _estimate_all_cells(df_pooled: pd.DataFrame, gens: list[str]) -> dict[str, float]:
    """Per (shell, gen, propagator) → A, k, R², LRT p-values.

    Returns a flat dict with keys ``f"{shell}|{gen}|{prop}|{stat}"`` so
    `bootstrap_by_sat` can build percentile CIs by key independently.
    Only A and k drive the figure bars; R² and the LRT p-values feed
    `tab_powerlaw.tex`.
    """
    out: dict[str, float] = {}
    for shell in ALT_SHELL_ORDER:
        for gen in gens:
            cell = df_pooled[(df_pooled["alt_shell"] == shell) & (df_pooled["gen_pooled"] == gen)]
            if cell.empty:
                continue
            for prop, col, _ in PROPAGATORS:
                try:
                    fit = fit_powerlaw_perpair(cell, col)
                except ValueError:
                    continue
                for stat in FIT_KEYS:
                    out[f"{shell}|{gen}|{prop}|{stat}"] = fit[stat]
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


def _format_ci(
    point_val: float,
    ci: tuple[float, float],
    precision: int = 2,
    *,
    fmt: str = "g",
) -> str:
    """Render ``value [lo, hi]`` for the LaTeX table. ``--`` if missing.

    ``fmt="g"`` uses Python's significant-figures formatting (compact, but
    strips trailing zeros — so an exact-CI bound like 1.30 renders as
    ``1.3`` and a tight interval can collapse to ``1.3 [1.3, 1.3]``).
    ``fmt="f"`` uses fixed decimals (``1.30 [1.26, 1.30]``), which the
    ``k`` column needs to preserve CI-bound separation at the second
    decimal.
    """
    if not np.isfinite(point_val):
        return "--"
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{point_val:.{precision}{fmt}}"
    return (
        f"{point_val:.{precision}{fmt}} "
        f"[{lo:.{precision}{fmt}}, {hi:.{precision}{fmt}}]"
    )


def _format_r_squared(point_val: float) -> str:
    """Render the cell's R² with two decimals, ``--`` if missing."""
    if not np.isfinite(point_val):
        return "--"
    return f"{point_val:.2f}"


def _format_p(p_val: float) -> str:
    """Render an LRT p-value compactly — ``<10^{-3}`` past three decimals."""
    if not np.isfinite(p_val):
        return "--"
    if p_val < 1e-3:
        return "$<\\!10^{-3}$"
    return f"{p_val:.3f}"


def write_table(
    point: dict[str, float],
    cis: dict[str, tuple[float, float]],
    gens: list[str],
    out_path: Path,
) -> None:
    """Booktabs table of (shell × gen) → A, k, R², LRT for SGP4 and hifi.

    Written as an ``\\input{...}``-able fragment (no \\begin{table} or
    caption); ms.tex wraps it. Each propagator block carries:
    ``A`` and ``k`` point estimates with 95% sat-level bootstrap CIs,
    the per-cell ``R²`` (per the unconstrained per-pair fit, not
    bootstrapped), and likelihood-ratio p-values for the two nulls
    ``k = 1`` and ``k = 2``.
    """
    lines = [
        "% Auto-generated by src/scripts/fig_powerlaw_fits.py — do not edit.",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        " & & \\multicolumn{4}{c}{SGP4} & \\multicolumn{4}{c}{high-fid} \\\\",
        "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}",
        "shell & generation & $A$ [95\\% CI] & $k$ [95\\% CI] & $R^{2}$ & $p_{k=1}/p_{k=2}$"
        " & $A$ [95\\% CI] & $k$ [95\\% CI] & $R^{2}$ & $p_{k=1}/p_{k=2}$ \\\\",
        "\\midrule",
    ]
    for shell in ALT_SHELL_ORDER:
        for gen in gens:
            cells: list[str] = [f"{shell} km", gen]
            seen = False
            # Bold the A entry of the propagator with the smaller fitted
            # A in this (shell, gen) cell. NaN-skipping keeps the column
            # honest if a propagator's fit didn't converge for the cell.
            a_by_prop = {
                prop: point.get(f"{shell}|{gen}|{prop}|A", float("nan"))
                for prop, _, _ in PROPAGATORS
            }
            finite_props = [p for p, v in a_by_prop.items() if np.isfinite(v)]
            lower_a_prop = min(finite_props, key=a_by_prop.get) if len(finite_props) >= 2 else None
            for prop, _, _ in PROPAGATORS:
                A_key = f"{shell}|{gen}|{prop}|A"  # noqa: N806 — math notation
                k_key = f"{shell}|{gen}|{prop}|k"
                r2_key = f"{shell}|{gen}|{prop}|r_squared"
                p1_key = f"{shell}|{gen}|{prop}|p_lrt_k1"
                p2_key = f"{shell}|{gen}|{prop}|p_lrt_k2"
                A_val = point.get(A_key, float("nan"))  # noqa: N806
                k_val = point.get(k_key, float("nan"))
                nan_pair = (float("nan"), float("nan"))
                a_cell = _format_ci(A_val, cis.get(A_key, nan_pair))
                if prop == lower_a_prop:
                    a_cell = f"\\textbf{{{a_cell}}}"
                cells.append(a_cell)
                cells.append(
                    _format_ci(k_val, cis.get(k_key, nan_pair), fmt="f")
                )
                cells.append(_format_r_squared(point.get(r2_key, float("nan"))))
                cells.append(
                    f"{_format_p(point.get(p1_key, float('nan')))}"
                    f" / {_format_p(point.get(p2_key, float('nan')))}"
                )
                if np.isfinite(A_val):
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
