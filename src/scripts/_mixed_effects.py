"""Linear mixed-effects supplement to the §3.7.1 main power-law fit.

Underscored filename so showyourwork does not match it as a manuscript
figure rule. Output is `outputs/mixed_effects_results.csv`, referenced
from the Appendix-B "Mixed-effects sensitivity" subsection but not
rendered through a `\\script{}` block.

For each (altitude shell × pooled generation × propagator) cell, fit
``log10(|Δr|) ~ log10(actual_dt_sec/3600) + (1 | norad_id)`` with a
satellite-level random intercept. The fixed-effect slope is the
mixed-effects analogue of the per-pair OLS exponent `k` reported in
Table~1; the random-intercept SD captures the per-satellite scatter
in the coefficient ``log10(A)``. This is the v0.1.0 concession to R1
#10 (correlated observations within sat) — the bootstrap-by-sat
percentile CIs in the main analysis preserve the same correlation
structure non-parametrically, and this CSV provides a parametric
cross-check that a reader can compare to Table~1's ``k`` column.

A reader who finds the mixed-effects slope outside the bootstrap CI
should treat the main estimator as the authoritative one (the
percentile CI bounds the sat-level resampling distribution directly;
the mixed-effects fixed-effect SE assumes a normal random-intercept
distribution that thin or unbalanced cells violate).

Usage:
    python src/scripts/_mixed_effects.py \\
        --all-runs outputs/all_runs.parquet \\
        --out outputs/mixed_effects_results.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from _style import ALT_SHELL_ORDER, pool_sparse_generations

PROPAGATORS = (("sgp4", "dr_sgp4_km"), ("hifi", "dr_hifi_km"))
MIN_PAIRS_PER_CELL = 30
MIN_SATS_PER_CELL = 5


def _fit_cell(cell: pd.DataFrame, error_col: str) -> dict[str, float] | None:
    """Fit ``log_dr ~ log_dt + (1 | norad_id)`` for one cell.

    Returns None when the cell has too few pairs/sats or the optimiser
    fails to converge — both are recorded as ``status`` in the parent
    sweep so a reader can audit which cells were dropped.
    """
    sub = cell[(cell[error_col] > 0) & np.isfinite(cell["actual_dt_sec"])]
    if len(sub) < MIN_PAIRS_PER_CELL:
        return None
    if sub["norad_id"].nunique() < MIN_SATS_PER_CELL:
        return None

    design = pd.DataFrame(
        {
            "log_dr": np.log10(sub[error_col].to_numpy()),
            "log_dt": np.log10(sub["actual_dt_sec"].to_numpy() / 3600.0),
            "norad_id": sub["norad_id"].to_numpy(),
        }
    )

    # statsmodels emits ConvergenceWarning freely on thin cells; we
    # encode convergence in the returned status field rather than
    # surfacing the warning chatter. We leave the optimiser at
    # statsmodels' default (BFGS) — L-BFGS pulls the random-intercept
    # variance to the zero boundary on the well-populated cells in
    # this corpus, which then aliases the random-intercept scale into
    # the fixed-effect intercept and produces a degenerate fit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = smf.mixedlm(
                "log_dr ~ log_dt",
                data=design,
                groups=design["norad_id"],
                re_formula="~1",
            ).fit(reml=True)
        except (np.linalg.LinAlgError, ValueError):
            return None

    if not result.converged:
        return None

    intercept_fe = float(result.fe_params["Intercept"])
    slope_fe = float(result.fe_params["log_dt"])
    slope_se = float(result.bse_fe["log_dt"])
    re_var = float(result.cov_re.iloc[0, 0]) if not result.cov_re.empty else float("nan")
    re_sd = float(np.sqrt(re_var)) if np.isfinite(re_var) and re_var >= 0 else float("nan")

    return {
        "intercept_fe": intercept_fe,
        "A_implied": float(10**intercept_fe),
        "k_fe": slope_fe,
        "k_fe_se": slope_se,
        "k_fe_ci_lo": float(slope_fe - 1.96 * slope_se),
        "k_fe_ci_hi": float(slope_fe + 1.96 * slope_se),
        "re_sd_log10A": re_sd,
        "n_pairs": int(len(sub)),
        "n_sats": int(sub["norad_id"].nunique()),
    }


def run_all_cells(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Fit every (shell × gen × propagator) cell.

    Emits one row per cell, with ``status`` in {``ok``, ``skipped``} so a
    downstream tabulator can be unambiguous about missing entries.
    """
    df, _ = pool_sparse_generations(all_runs)
    gens = sorted(df["gen_pooled"].unique())
    rows: list[dict] = []
    for shell in ALT_SHELL_ORDER:
        for gen in gens:
            cell = df[(df["alt_shell"] == shell) & (df["gen_pooled"] == gen)]
            if cell.empty:
                continue
            for prop, error_col in PROPAGATORS:
                base = {
                    "alt_shell": shell,
                    "gen_pooled": gen,
                    "propagator": prop,
                }
                fit = _fit_cell(cell, error_col)
                if fit is None:
                    rows.append({**base, "status": "skipped"})
                    continue
                rows.append({**base, "status": "ok", **fit})
    return pd.DataFrame(rows)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-runs", type=Path, default=Path("outputs/all_runs.parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/mixed_effects_results.csv"))
    args = parser.parse_args()

    all_runs = pd.read_parquet(args.all_runs)
    results = run_all_cells(all_runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
