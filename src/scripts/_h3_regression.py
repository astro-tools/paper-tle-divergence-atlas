"""H3 regression: per-shell ANCOVA of SGP4 staleness coefficient vs. F10.7.

For each altitude shell, fit the additive model
``log10 A_i = alpha_gen(i) + beta * F10.7_i + epsilon_i`` across the
satellites sampled in the shell, with the per-satellite SGP4 staleness
coefficient ``A_i`` recovered by ``_style.fit_powerlaw`` from the
satellite's per-bucket ``dr_sgp4_km`` (the same per-sat estimator F7 has
used since #25). The single F10.7 slope ``beta`` is the H3 quantity of
interest; the per-cohort intercepts ``alpha_gen`` absorb whichever
baseline differences in ``(C_D * A / m)`` and operator-OD tuning the
v1.x and v2-mini cohorts carry into their fitted ``B*``.

The ANCOVA form follows from the drag-mismodelling decomposition. Drag
acceleration is ``a_drag = (1/2) * rho(h, F10.7) * v**2 * (C_D * A / m)``;
SGP4 absorbs ``(1/2) * (C_D * A / m) * rho_ref`` into a single scalar
``B_star`` fitted at epoch; the forward-propagation along-track error
from a ``B_star`` mis-fit is proportional to
``Delta rho(F10.7) / rho_ref``. Taking logs and differentiating in
F10.7,

    log A = log(C_D * A / m) + log rho_ref(h, F10.7_window) + const
    d(log A) / d(F10.7) = d(log rho) / d(F10.7) | h.

The slope is therefore a function of altitude and ambient F10.7 only;
spacecraft properties enter the intercept of ``log A`` versus F10.7,
not the slope. A naive per-shell pooled fit at 550 or 560 km can
nevertheless produce a Simpson-paradox spurious slope: v1.x and v2-mini
intercepts differ by roughly a factor of four in ``A`` and their epoch
distributions are slightly offset in F10.7, so the cross-cohort
intercept gap aliases into a wrong-signed pooled ``beta``. The
generation covariate ``C(gen_pooled)`` removes that confound while
preserving the physics-aligned "one beta per altitude" prediction.

The interaction model
``log10 A_i = alpha_gen(i) + (beta + delta_beta_gen(i)) * F10.7_i +
epsilon_i`` is fit as a diagnostic per shell; an F-test of
``H0: delta_beta_gen == 0`` for every populated non-reference generation
reports whether the slope itself differs by cohort. The expectation
under the physics derivation is failure-to-reject (slope is
gen-independent); if the test fires, the prose reports it as a
modelling departure worth following up rather than silently overriding
the additive headline.

Two predictors are reported per shell:

* ``f107_daily_mean`` -- per-sat mean of daily-observed F10.7 over the
  satellite's window of starting epochs, already carried on every row
  of ``all_runs.parquet`` as ``f107``. Headline number per the v3 plan.
* ``f107_avg81_mean`` -- per-sat mean of the CelesTrak 81-day-centred
  F10.7 average over the same epochs, joined post-hoc from
  ``src/static/sw_cache.parquet``. NRLMSISE-00's long-term thermospheric
  driver; reported as a robustness check.

Fitting machinery per (shell, predictor):

* Full-sample OLS via statsmodels with the asymptotic t-stat
  ``slope_p`` for ``H0: beta = 0``, the per-gen intercept offsets, and
  the model R^2.
* ``slope_ci_95`` and per-gen ``intercept_ci_95`` from a
  1,000-resample satellite-level bootstrap (each resample draws sats
  with replacement from the shell and refits the ANCOVA), 2.5 / 97.5
  percentile bounds. Identical resampling pattern to
  ``_style.bootstrap_by_sat`` and Table 3's per-cell power-law CIs.
* ``interaction_p`` from a partial-F (statsmodels ``compare_f_test``)
  contrasting the additive model against the interaction model on the
  same data; reported only when the shell has more than one populated
  cohort.

Output of ``_cli()`` is ``outputs/h3_regression.json`` -- summary stats
only, one block per (shell, predictor); the bootstrap raw samples are
in-memory only.

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
import statsmodels.formula.api as smf
from _style import ALT_SHELL_ORDER, fit_powerlaw, pool_sparse_generations
from statsmodels.regression.linear_model import RegressionResultsWrapper

MIN_BUCKETS_PER_SAT: Final = 3
MIN_SATS_PER_SHELL: Final = 10
N_BOOTSTRAP: Final = 1000
BOOTSTRAP_SEED: Final = 42

# Pooled-generation labels we expect to see after `pool_sparse_generations`.
# v1.0 always pools into v1.5 -> "v1.x" on the corpus the paper depends on;
# v2-mini stays distinct. Reference category for the ANCOVA intercept
# below is v1.x (alphabetically first; statsmodels' default).
GEN_POOLED_ORDER: Final = ("v1.x", "v2-mini")
REFERENCE_GEN: Final = "v1.x"

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
    ``date(t_i)``. The 81-day-centred average is the NRLMSISE-00
    long-term driver and a near-constant over the 30-day corpus window,
    which limits its statistical leverage as a predictor.
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


def _ancova_design(shell_df: pd.DataFrame, predictor: str) -> pd.DataFrame:
    """statsmodels-friendly frame for the ANCOVA fit.

    ``log_A`` is ``log10 A``; ``gen_pooled`` is a categorical with the
    reference category fixed to ``REFERENCE_GEN`` so the dummy-coded
    coefficients carry interpretable "delta-vs-v1.x" semantics.
    """
    design = pd.DataFrame(
        {
            "log_A": np.log10(shell_df["A"].to_numpy(dtype=float)),
            "F107": shell_df[predictor].to_numpy(dtype=float),
            "gen_pooled": pd.Categorical(
                shell_df["gen_pooled"], categories=GEN_POOLED_ORDER, ordered=False
            ),
        }
    )
    return design


def _fit_additive(design: pd.DataFrame) -> RegressionResultsWrapper:
    """Fit ``log_A ~ F107 + C(gen_pooled)`` with v1.x as reference."""
    return smf.ols(
        f"log_A ~ F107 + C(gen_pooled, Treatment(reference='{REFERENCE_GEN}'))", data=design
    ).fit()


def _fit_interaction(design: pd.DataFrame) -> RegressionResultsWrapper:
    """Fit the interaction-model counterpart of `_fit_additive`."""
    return smf.ols(
        f"log_A ~ F107 * C(gen_pooled, Treatment(reference='{REFERENCE_GEN}'))",
        data=design,
    ).fit()


def _fit_single_gen(design: pd.DataFrame) -> RegressionResultsWrapper:
    """Fallback for shells with only one populated gen (540 km)."""
    return smf.ols("log_A ~ F107", data=design).fit()


def _summary_from_fit(
    fit: RegressionResultsWrapper,
    gens_present: list[str],
) -> dict:
    """Extract the per-shell ANCOVA summary from a fitted model.

    Returns ``slope`` / ``slope_se`` / ``slope_p`` / ``r_squared`` /
    ``intercept_v1x`` plus a dict of ``intercept_offsets`` keyed by the
    non-reference gens actually present in the shell.
    """
    params = fit.params
    bse = fit.bse
    pvalues = fit.pvalues
    intercept = float(params["Intercept"])
    slope = float(params["F107"])
    slope_se = float(bse["F107"])
    slope_p = float(pvalues["F107"])

    # The treatment-coded categorical dummy names follow
    # `C(gen_pooled, Treatment(reference='v1.x'))[T.v2-mini]`.
    offsets: dict[str, dict[str, float]] = {}
    for gen in gens_present:
        if gen == REFERENCE_GEN:
            continue
        key = f"C(gen_pooled, Treatment(reference='{REFERENCE_GEN}'))[T.{gen}]"
        if key in params.index:
            offsets[gen] = {
                "estimate": float(params[key]),
                "se": float(bse[key]),
                "p": float(pvalues[key]),
            }
    return {
        "intercept_ref": intercept,
        "reference_gen": REFERENCE_GEN,
        "slope": slope,
        "slope_se": slope_se,
        "slope_p": slope_p,
        "intercept_offsets": offsets,
        "r_squared": float(fit.rsquared),
        "n_sats": int(fit.nobs),
    }


def _bootstrap_ancova(
    shell_df: pd.DataFrame,
    predictor: str,
    gens_present: list[str],
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Satellite-level bootstrap of the ANCOVA fit.

    Returns arrays for ``slope``, ``intercept_ref``, and one
    ``offset_<gen>`` per non-reference generation actually present in
    the shell. Resamples whose fit fails (e.g. a draw that contains a
    single gen) are dropped; the caller percentile-collapses the kept
    arrays for CIs.
    """
    rng = np.random.default_rng(seed)
    n = len(shell_df)
    samples: dict[str, list[float]] = {
        "slope": [],
        "intercept_ref": [],
    }
    for gen in gens_present:
        if gen != REFERENCE_GEN:
            samples[f"offset_{gen}"] = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resamp = shell_df.iloc[idx]
        gens_in_resamp = [g for g in gens_present if (resamp["gen_pooled"] == g).any()]
        try:
            design = _ancova_design(resamp, predictor)
            fit = _fit_single_gen(design) if len(gens_in_resamp) <= 1 else _fit_additive(design)
        except (ValueError, np.linalg.LinAlgError):
            continue
        summary = _summary_from_fit(fit, gens_in_resamp)
        samples["slope"].append(summary["slope"])
        samples["intercept_ref"].append(summary["intercept_ref"])
        for gen, payload in summary["intercept_offsets"].items():
            samples[f"offset_{gen}"].append(payload["estimate"])
    return {key: np.asarray(vals) for key, vals in samples.items()}


def _percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    if len(samples) < N_BOOTSTRAP // 2:
        return float("nan"), float("nan")
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def fit_h3_per_shell(
    per_sat: pd.DataFrame,
    shell: str,
    predictor: str,
) -> dict:
    """Per-shell ANCOVA fit + bootstrap + interaction diagnostic.

    Returns the JSON-ready block (with the raw bootstrap arrays in a
    nested ``_samples`` dict that the figure script consumes for the CI
    ribbon and the on-disk JSON strips).
    """
    shell_df = per_sat[per_sat["alt_shell"] == shell]
    gens_present = [g for g in GEN_POOLED_ORDER if (shell_df["gen_pooled"] == g).any()]
    if len(shell_df) < MIN_SATS_PER_SHELL:
        return {
            "shell": shell,
            "predictor": predictor,
            "n_sats": int(len(shell_df)),
            "gens_present": gens_present,
            "fit": None,
            "note": f"n_sats={len(shell_df)} below the {MIN_SATS_PER_SHELL}-sat shell floor",
        }
    design = _ancova_design(shell_df, predictor)
    if len(gens_present) <= 1:
        additive_fit = _fit_single_gen(design)
        interaction_p = None
    else:
        additive_fit = _fit_additive(design)
        interaction_fit = _fit_interaction(design)
        # `<full>.compare_f_test(<restricted>)` is statsmodels' nesting
        # convention -- the interaction model is the full one, the
        # additive model is its restriction at `delta_beta_gen = 0`.
        # Returns (F, p, df_diff).
        f_stat, interaction_p_val, df_diff = interaction_fit.compare_f_test(additive_fit)
        interaction_p = {
            "f_stat": float(f_stat),
            "df_diff": float(df_diff),
            "p_value": float(interaction_p_val),
        }
    summary = _summary_from_fit(additive_fit, gens_present)
    samples = _bootstrap_ancova(shell_df, predictor, gens_present)
    intercept_offsets_with_ci: dict[str, dict[str, float | list[float]]] = {}
    for gen, payload in summary["intercept_offsets"].items():
        ci_lo, ci_hi = _percentile_ci(samples.get(f"offset_{gen}", np.asarray([])))
        intercept_offsets_with_ci[gen] = {
            **payload,
            "ci_95": [ci_lo, ci_hi],
        }
    slope_ci_lo, slope_ci_hi = _percentile_ci(samples["slope"])
    intercept_ci_lo, intercept_ci_hi = _percentile_ci(samples["intercept_ref"])
    return {
        "shell": shell,
        "predictor": predictor,
        "n_sats": summary["n_sats"],
        "gens_present": gens_present,
        "n_resamples_kept": int(len(samples["slope"])),
        "fit": {
            "intercept_ref": summary["intercept_ref"],
            "reference_gen": summary["reference_gen"],
            "slope": summary["slope"],
            "slope_se": summary["slope_se"],
            "slope_p": summary["slope_p"],
            "r_squared": summary["r_squared"],
            "intercept_offsets": intercept_offsets_with_ci,
        },
        "slope_ci_95": [slope_ci_lo, slope_ci_hi],
        "intercept_ref_ci_95": [intercept_ci_lo, intercept_ci_hi],
        "predictor_min": float(shell_df[predictor].min()),
        "predictor_max": float(shell_df[predictor].max()),
        "interaction_test": interaction_p,
        "_samples": {key: arr.tolist() for key, arr in samples.items()},
    }


def run_all(all_runs: pd.DataFrame, sw_cache: pd.DataFrame) -> dict:
    """Top-level driver: per-sat predictors then (shell x predictor) fits."""
    per_sat = per_sat_predictors(all_runs, sw_cache)
    blocks: list[dict] = []
    for shell in ALT_SHELL_ORDER:
        for predictor in PREDICTORS:
            blocks.append(fit_h3_per_shell(per_sat, shell, predictor))
    return {
        "headline_predictor": HEADLINE_PREDICTOR,
        "model": "log10 A ~ F107 + C(gen_pooled)  [ANCOVA, v1.x reference]",
        "interaction_diagnostic": "log10 A ~ F107 * C(gen_pooled), compared by partial-F",
        "n_resamples": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "min_buckets_per_sat": MIN_BUCKETS_PER_SAT,
        "min_sats_per_shell": MIN_SATS_PER_SHELL,
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
