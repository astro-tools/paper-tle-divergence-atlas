"""Shared matplotlib styling and palettes for `fig_*.py`.

Imported by every manuscript figure script. Centralizing rcParams, colors,
and the generation-pooling helper avoids drift across figures and keeps
the F2 / F3 visual comparison honest.

Underscored filename so `showyourwork` does not match it as a manuscript
figure rule.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Per-cell sample-size floor below which a generation is pooled with the
# closest non-empty generation for that (shell, Δt) cell. Set per the issue-4
# methodology update: "if any (shell × generation) cell has fewer than ~30
# pairs at a given Δt bucket, pool with the adjacent generation".
POOL_MIN_SAMPLES = 30

ALT_SHELL_ORDER = ["540", "550", "560"]
GENERATION_ORDER = ["v1.0", "v1.5", "v2-mini"]
BUCKET_SECONDS = [21600, 86400, 259200, 604800]
BUCKET_LABELS = {21600: "6h", 86400: "1d", 259200: "3d", 604800: "7d"}

# Altitude-shell color: viridis ramp by altitude. Used as the primary
# encoding on F1, and as the within-panel hue on F2 / F3.
_shell_cmap = mpl.colormaps["viridis"]
ALT_SHELL_COLORS = {
    "540": _shell_cmap(0.15),
    "550": _shell_cmap(0.50),
    "560": _shell_cmap(0.85),
}

# Generation palette matches `_diagnostic_sweep_scatter.py` so the dev
# diagnostic and the manuscript figures read the same way.
GENERATION_COLORS = {
    "v1.0": "#1f77b4",
    "v1.5": "#ff7f0e",
    "v2-mini": "#2ca02c",
}

# When v1.0 gets pooled into v1.5 (the corpus has <30 v1.0 pairs per cell
# across the board), call the union "v1.x" and recolor distinctly so a
# reader who only glances at F2 / F3 sees the pooling without checking the
# caption.
POOLED_GENERATION_COLORS = {
    "v1.x": "#7570b3",
    "v2-mini": "#2ca02c",
}

GENERATION_MARKERS = {
    "v1.0": "o",
    "v1.5": "s",
    "v2-mini": "^",
}


def apply_rc() -> None:
    """Set rcParams. Call once at the top of each figure script's _cli()."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.markersize": 4,
        }
    )


def pool_sparse_generations(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Add a `gen_pooled` column collapsing sparse generations.

    Checks every (alt_shell, generation, target_dt_sec) cell. If any cell
    for a given generation falls under POOL_MIN_SAMPLES, that generation
    is merged with its adjacent neighbour for the whole dataframe (a
    blanket pool — simpler and more honest visually than per-cell
    pooling, since the sparse generation would otherwise appear in some
    panels and vanish in others).

    Adjacency: v1.0 ↔ v1.5 (→ "v1.x"), v2-mini stays distinct unless v1.5
    is also thin (not observed in the Day-4 corpus).

    Returns the augmented dataframe and a one-line note suitable for a
    figure-corner annotation. The note is empty when no pooling fired.
    """
    counts = df.groupby(["alt_shell", "generation", "target_dt_sec"], observed=False).size()
    out = df.copy()
    out["gen_pooled"] = out["generation"]
    notes: list[str] = []

    if "v1.0" in counts.index.get_level_values("generation"):
        v10_thin = (counts.xs("v1.0", level="generation") < POOL_MIN_SAMPLES).any()
        if v10_thin:
            out.loc[out["generation"].isin({"v1.0", "v1.5"}), "gen_pooled"] = "v1.x"
            notes.append(f"v1.0 pooled into v1.5 → v1.x (n<{POOL_MIN_SAMPLES} in some cells)")

    return out, "; ".join(notes)


def shared_error_ylim(df: pd.DataFrame, pad_decades: float = 0.15) -> tuple[float, float]:
    """Symmetric log y-limits covering both `dr_sgp4_km` and `dr_hifi_km`.

    Used by F2 and F3 so they share identical axes — the F2-vs-F3 visual
    comparison is what carries the H2 read.
    """
    both = pd.concat([df["dr_sgp4_km"], df["dr_hifi_km"]])
    both = both[both > 0]
    lo = both.min()
    hi = both.max()
    import math

    log_lo = math.log10(lo) - pad_decades
    log_hi = math.log10(hi) + pad_decades
    return 10**log_lo, 10**log_hi
