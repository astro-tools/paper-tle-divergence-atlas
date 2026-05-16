"""H3 regression: per-satellite SGP4 staleness coefficient vs. F10.7.

For each (altitude shell, pooled generation) cell, fit
``log10 A_i = alpha + beta * F10.7_i + epsilon_i`` across the satellites
sampled in that cell. ``A_i`` is the per-sat SGP4 staleness coefficient
recovered by ``_style.fit_powerlaw`` from the satellite's per-bucket
``dr_sgp4_km`` (the same estimator F7 has used since #25). Stratifying
within shell by generation is load-bearing: pooled-by-shell fits at 550
and 560 km are confounded by a Simpson-paradox composition shift -- the
v1.x and v2-mini cohorts have systematically different per-sat ``A``
*and* slightly different F10.7 windows, which mixes into a spurious
cross-generation slope that masks the within-cohort F10.7 sensitivity.
The (shell x generation) granularity keeps each fit generation-clean.

Two predictors are reported side by side:

* ``f107_daily_mean`` -- per-sat mean of daily-observed F10.7 over the
  satellite's window of starting epochs, already carried on every row of
  ``all_runs.parquet`` as ``f107``. Headline number per the v3 plan.
* ``f107_avg81_mean`` -- per-sat mean of the CelesTrak 81-day-centred
  F10.7 average over the same epochs, joined post-hoc from
  ``src/static/sw_cache.parquet``. NRLMSISE-00's long-term thermospheric
  driver; reported as a robustness check.

Fitting machinery per (shell, gen, predictor):

* ``alpha_hat`` / ``beta_hat`` / ``slope_se`` / ``slope_p`` / ``r_squared``
  / ``n_sats`` -- full-sample OLS via statsmodels with the asymptotic
  t-stat p-value for ``H0: beta = 0``.
* ``slope_ci_95`` / ``intercept_ci_95`` -- percentile CIs from a
  1,000-resample satellite-level bootstrap. Each resample draws sats
  with replacement from the cell and refits the OLS line; CI bounds
  are the 2.5th / 97.5th percentiles of the resampled coefficient
  distribution. Same satellite-level resampling pattern as
  ``_style.bootstrap_by_sat`` and the per-cell power-law CIs in
  Table 3 (the ``fit_powerlaw_perpair`` / ``bootstrap_by_sat`` chain).

The raw resampled (alpha, beta) arrays are returned alongside the
summary stats so ``fig_solar_modulation.py`` can compute the OLS line's
95% CI ribbon at any predictor grid by evaluating the resampled lines
and percentile-collapsing the resulting band.

Output of ``_cli()`` is ``outputs/h3_regression.json`` -- summary stats
only, one block per (shell, gen, predictor); the bootstrap raw samples
are in-memory only.

Usage:
    python src/scripts/_h3_regression.py \\
        --all-runs outputs/all_runs.parquet \\
        --sw-cache src/static/sw_cache.parquet \\
        --out outputs/h3_regression.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import statsmodels.api as sm
from _style import ALT_SHELL_ORDER, fit_powerlaw, pool_sparse_generations

MIN_BUCKETS_PER_SAT: Final = 3
MIN_SATS_PER_CELL: Final = 5
N_BOOTSTRAP: Final = 1000
BOOTSTRAP_SEED: Final = 42

# Pooled-generation labels we expect to see after `pool_sparse_generations`.
# v1.0 always pools into v1.5 -> "v1.x" on the corpus the paper depends on;
# v2-mini stays distinct. Cells that are empty in the corpus (notably
# 540 x v2-mini) are surfaced in the JSON with `n_sats: 0, fit: None`.
GEN_POOLED_ORDER: Final = ("v1.x", "v2-mini")

# Per-sat F10.7 predictors, in the order they appear in the JSON.
PREDICTORS: Final = ("f107_daily_mean", "f107_avg81_mean")
HEADLINE_PREDICTOR: Final = "f107_daily_mean"


def per_sat_predictors(all_runs: pd.DataFrame, sw_cache: pd.DataFrame) -> pd.DataFrame:
    """One row per (shell, sat) with the H3 inputs.

    Columns: ``alt_shell, norad_id, gen_pooled, A, k, n_pairs,
    f107_daily_mean, f107_avg81_mean``. Satellites whose per-sat
    power-law fit cannot be recovered (fewer than
    ``MIN_BUCKETS_PER_SAT`` usable buckets) are omitted.

    ``f107_daily_mean`` is the mean of the ``f107`` column already on
    ``all_runs`` (CelesTrak daily observed); ``f107_avg81_mean`` is the
    per-sat mean of the SW cache's ``f107_avg81`` column joined by
    ``date(t_i)``. The 81-day-centred average is the NRLMSISE-00 long-term
    driver and a near-constant over the 30-day corpus window, which is
    one of the H3 results worth reporting honestly rather than burying.
    """
    df = all_runs.copy()
    df["date_i"] = pd.to_datetime(df["t_i"]).dt.date
    sw = sw_cache[["date", "f107_avg81"]].copy()
    sw["date"] = pd.to_datetime(sw["date"]).dt.date
    df = df.merge(sw, left_on="date_i", right_on="date", how="left", validate="many_to_one")
    df, _ = pool_sparse_generations(df)

    rows: list[dict] = []
    for (shell, sat, gen), sat_df in df.groupby(
        ["alt_shell", "norad_id", "gen_pooled"], observed=True
    ):
        # Power-law fit needs MIN_BUCKETS_PER_SAT distinct buckets with
        # >=3 positive errors each -- otherwise F7's per-sat A is not
        # recoverable for this satellite.
        usable_buckets = (
            sat_df.groupby("target_dt_sec", observed=True)["dr_sgp4_km"]
            .apply(lambda s: (s > 0).sum())
            .ge(3)
            .sum()
        )
        if usable_buckets < MIN_BUCKETS_PER_SAT:
            continue
        try:
            A, k = fit_powerlaw(sat_df, "dr_sgp4_km")  # noqa: N806 -- math notation
        except ValueError:
            continue
        rows.append(
            {
                "alt_shell": shell,
                "norad_id": sat,
                "gen_pooled": gen,
                "A": A,
                "k": k,
                "n_pairs": int(len(sat_df)),
                "f107_daily_mean": float(sat_df["f107"].mean()),
                "f107_avg81_mean": float(sat_df["f107_avg81"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _fit_ols(per_sat: pd.DataFrame, predictor: str) -> dict[str, float]:
    """Full-sample OLS of ``log10 A`` on ``predictor`` with t-stat p-value.

    Returns intercept, slope, slope SE, slope two-sided p-value, R^2,
    and n_sats. The ``slope`` field is the regression coefficient
    ``beta_hat`` of the model ``log10 A_i = alpha + beta * predictor_i``.
    """
    x = per_sat[predictor].to_numpy(dtype=float)
    y = np.log10(per_sat["A"].to_numpy(dtype=float))
    X = sm.add_constant(x)  # noqa: N806 -- statsmodels convention
    res = sm.OLS(y, X).fit()
    return {
        "intercept": float(res.params[0]),
        "slope": float(res.params[1]),
        "slope_se": float(res.bse[1]),
        "slope_p": float(res.pvalues[1]),
        "r_squared": float(res.rsquared),
        "n_sats": int(res.nobs),
    }


def _ols_pointwise(per_sat: pd.DataFrame, predictor: str) -> tuple[float, float]:
    """Closed-form OLS slope and intercept -- bootstrap inner loop.

    The full-sample fit uses statsmodels for the SE/p-value machinery;
    the bootstrap only needs point estimates per resample, so a numpy
    ``polyfit`` is cheaper. Returns ``(intercept, slope)``.
    """
    x = per_sat[predictor].to_numpy(dtype=float)
    y = np.log10(per_sat["A"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(intercept), float(slope)


def _bootstrap_samples(
    per_sat: pd.DataFrame,
    predictor: str,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Satellite-level bootstrap of the H3 OLS fit.

    Each resample picks ``n_sats`` satellites with replacement and
    refits ``log10 A ~ predictor`` on the resampled per-sat frame.
    Resamples whose design matrix becomes degenerate (all predictor
    values identical) are dropped. Returns two arrays of equal length
    holding the resampled (intercept, slope) pairs; the caller takes
    percentiles for CIs and evaluates the resampled lines at any
    predictor grid for the F7 CI ribbon.
    """
    rng = np.random.default_rng(seed)
    n = len(per_sat)
    intercepts: list[float] = []
    slopes: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resamp = per_sat.iloc[idx]
        try:
            intercept, slope = _ols_pointwise(resamp, predictor)
        except (ValueError, np.linalg.LinAlgError):
            continue
        intercepts.append(intercept)
        slopes.append(slope)
    return np.asarray(intercepts), np.asarray(slopes)


def fit_h3_per_cell(
    per_sat_all: pd.DataFrame,
    shell: str,
    gen: str,
    predictor: str,
) -> dict:
    """Per-(shell, gen) H3 fit + bootstrap; ready for the JSON output.

    The returned dict carries the full-sample point estimate and t-stat
    p-value, the 95% percentile CI from the satellite-level bootstrap on
    both ``intercept`` and ``slope``, and the raw bootstrap arrays in a
    nested ``_samples`` dict the figure script consumes for the CI
    ribbon. Empty cells (notably 540 x v2-mini, which the corpus does
    not populate -- see Table 1 of the manuscript) are surfaced with
    ``n_sats: 0, fit: None`` rather than dropped from the JSON so a
    downstream consumer sees the complete (shell x gen) grid.
    """
    cell = per_sat_all[(per_sat_all["alt_shell"] == shell) & (per_sat_all["gen_pooled"] == gen)]
    if len(cell) < MIN_SATS_PER_CELL:
        return {
            "shell": shell,
            "gen_pooled": gen,
            "predictor": predictor,
            "n_sats": int(len(cell)),
            "fit": None,
            "note": f"n_sats={len(cell)} below the {MIN_SATS_PER_CELL}-sat fitting floor",
        }
    fit = _fit_ols(cell, predictor)
    intercepts, slopes = _bootstrap_samples(cell, predictor)
    if len(slopes) < N_BOOTSTRAP // 2:
        slope_ci = (float("nan"), float("nan"))
        intercept_ci = (float("nan"), float("nan"))
    else:
        slope_ci = (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)))
        intercept_ci = (
            float(np.percentile(intercepts, 2.5)),
            float(np.percentile(intercepts, 97.5)),
        )
    return {
        "shell": shell,
        "gen_pooled": gen,
        "predictor": predictor,
        "n_sats": fit["n_sats"],
        "n_resamples_kept": int(len(slopes)),
        "fit": fit,
        "slope_ci_95": list(slope_ci),
        "intercept_ci_95": list(intercept_ci),
        "predictor_min": float(cell[predictor].min()),
        "predictor_max": float(cell[predictor].max()),
        "_samples": {
            "intercepts": intercepts.tolist(),
            "slopes": slopes.tolist(),
        },
    }


def run_all(all_runs: pd.DataFrame, sw_cache: pd.DataFrame) -> dict:
    """Top-level driver: per-sat predictors then (shell x gen x predictor) fits."""
    per_sat = per_sat_predictors(all_runs, sw_cache)
    blocks: list[dict] = []
    for shell in ALT_SHELL_ORDER:
        for gen in GEN_POOLED_ORDER:
            for predictor in PREDICTORS:
                blocks.append(fit_h3_per_cell(per_sat, shell, gen, predictor))
    return {
        "headline_predictor": HEADLINE_PREDICTOR,
        "n_resamples": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "min_buckets_per_sat": MIN_BUCKETS_PER_SAT,
        "min_sats_per_cell": MIN_SATS_PER_CELL,
        "per_sat_count_total": int(len(per_sat)),
        "blocks": blocks,
    }


def _strip_raw_samples(payload: dict) -> dict:
    """Drop the bootstrap ``_samples`` arrays for the on-disk JSON."""
    light = dict(payload)
    light["blocks"] = [{k: v for k, v in b.items() if k != "_samples"} for b in payload["blocks"]]
    return light


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-runs", type=Path, default=Path("outputs/all_runs.parquet"))
    parser.add_argument("--sw-cache", type=Path, default=Path("src/static/sw_cache.parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/h3_regression.json"))
    args = parser.parse_args()

    all_runs = pd.read_parquet(args.all_runs)
    sw_cache = pd.read_parquet(args.sw_cache)
    payload = run_all(all_runs, sw_cache)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_strip_raw_samples(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
