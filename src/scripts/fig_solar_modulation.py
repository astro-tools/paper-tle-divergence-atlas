"""Figure F7: solar-activity modulation of SGP4 error coefficient.

For the sub-550 km altitude shell only — where atmospheric drag
dominates and so the staleness coefficient should track F10.7 most
strongly — fit a per-satellite power-law ``A · Δt^k`` against
``dr_sgp4_km`` and scatter the fitted ``A`` against the satellite's
mean F10.7 over its corpus time window.

A rolling-median overlay per pooled generation gives a non-parametric
read of the modulation. Addresses H3 in the project plan: error growth
under drag-driven conditions scales with solar EUV proxy.

The per-sat fit requires at least `MIN_BUCKETS_FOR_FIT` distinct buckets
with `MIN_PAIRS_PER_BUCKET` pairs each — sats that fall below this floor
are dropped and reported in the figure's pool-note line.

Usage:
    python src/scripts/fig_solar_modulation.py \\
        --all-runs outputs/all_runs.parquet \\
        --out src/tex/figures/fig_solar_modulation.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from _style import (
    POOLED_GENERATION_COLORS,
    apply_rc,
    fit_powerlaw,
    pool_sparse_generations,
)

DRAG_DOMINATED_SHELL = "540"
MIN_BUCKETS_PER_SAT = 3
ROLLING_WINDOW = 10  # sats per rolling-median window in the F10.7 overlay


def _per_sat_fits(df: pd.DataFrame) -> pd.DataFrame:
    """For each sat in `df`, fit A & k against dr_sgp4_km and average F10.7.

    Returns a frame with columns ``norad_id, gen_pooled, A, k,
    f107_mean, n_pairs``. Sats whose per-sat fit fails (not enough
    buckets) are omitted.
    """
    rows: list[dict] = []
    for (sat, gen), sat_df in df.groupby(["norad_id", "gen_pooled"], observed=True):
        # Need at least MIN_BUCKETS_PER_SAT distinct buckets with enough pairs.
        usable_buckets = (
            sat_df.groupby("target_dt_sec", observed=True)["dr_sgp4_km"]
            .apply(lambda s: (s > 0).sum())
            .ge(3)
            .sum()
        )
        if usable_buckets < MIN_BUCKETS_PER_SAT:
            continue
        try:
            A, k = fit_powerlaw(sat_df, "dr_sgp4_km")  # noqa: N806 — math notation
        except ValueError:
            continue
        rows.append(
            {
                "norad_id": sat,
                "gen_pooled": gen,
                "A": A,
                "k": k,
                "f107_mean": float(sat_df["f107"].mean()),
                "n_pairs": int(len(sat_df)),
            }
        )
    return pd.DataFrame(rows)


def plot_solar_modulation(df: pd.DataFrame, out_path: Path) -> None:
    apply_rc()

    df_shell = df[df["alt_shell"] == DRAG_DOMINATED_SHELL]
    if df_shell.empty:
        raise ValueError(
            f"no rows with alt_shell == '{DRAG_DOMINATED_SHELL}' — F7 needs the drag-dominated bin"
        )

    df_shell, pool_note = pool_sparse_generations(df_shell)
    fits = _per_sat_fits(df_shell)
    if fits.empty:
        raise ValueError(
            f"no sats survived the per-sat fit floor "
            f"(need {MIN_BUCKETS_PER_SAT} buckets each); F7 cannot be drawn"
        )

    pooled_gens = sorted(fits["gen_pooled"].unique())
    gen_colors = {g: POOLED_GENERATION_COLORS.get(g, f"C{i}") for i, g in enumerate(pooled_gens)}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for gen in pooled_gens:
        sub = fits[fits["gen_pooled"] == gen].sort_values("f107_mean")
        if sub.empty:
            continue
        color = gen_colors[gen]
        ax.scatter(
            sub["f107_mean"],
            sub["A"],
            s=18,
            color=color,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.3,
            label=f"{gen} (n={len(sub)} sats)",
        )
        if len(sub) >= ROLLING_WINDOW:
            rolling = sub["A"].rolling(ROLLING_WINDOW, center=True).median()
            ax.plot(sub["f107_mean"], rolling, color=color, linewidth=1.6, alpha=0.9)

    ax.set_xlabel("mean F10.7 over satellite's corpus window (sfu)")
    ax.set_ylabel("fitted coefficient A  (km at Δt = 1 h)")
    ax.set_yscale("log")
    ax.set_title(
        f"Per-sat SGP4 staleness coefficient vs. solar activity "
        f"(alt shell {DRAG_DOMINATED_SHELL} km)"
    )
    handles = [
        mlines.Line2D(
            [],
            [],
            color=gen_colors[g],
            marker="o",
            linestyle="-",
            markersize=6,
            label=f"{g} (n={int((fits['gen_pooled'] == g).sum())})",
        )
        for g in pooled_gens
    ]
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="0.4",
            linestyle="-",
            linewidth=1.6,
            label=f"rolling median (window={ROLLING_WINDOW})",
        )
    )
    ax.legend(handles=handles, loc="upper left")

    n_dropped_sats = int(df_shell["norad_id"].nunique() - len(fits))
    footer_parts: list[str] = []
    if pool_note:
        footer_parts.append(pool_note)
    if n_dropped_sats:
        footer_parts.append(
            f"{n_dropped_sats} sat(s) dropped (< {MIN_BUCKETS_PER_SAT} usable buckets)"
        )
    if footer_parts:
        fig.text(
            0.5,
            0.005,
            "; ".join(footer_parts),
            ha="center",
            fontsize=7,
            style="italic",
            color="0.35",
        )
        rect = (0.0, 0.03, 1.0, 1.0)
    else:
        rect = (0.0, 0.0, 1.0, 1.0)

    fig.tight_layout(rect=rect)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


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
        default=Path("src/tex/figures/fig_solar_modulation.pdf"),
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.all_runs)
    plot_solar_modulation(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
