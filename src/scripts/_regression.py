"""Joint regression of propagation error on orbital, temporal, and solar predictors.

Underscored filename so showyourwork does not match it as a manuscript
figure rule. Output is `outputs/regression_results.csv`, referenced from
the Results section's methodological context but not from a `\\script{}`
block.

Two model variants per propagator (SGP4, high-fid):

    bucket    — log10(‖Δr‖) ~ sma + ecc + inc + C(target_dt_sec) + f107
                   + ap + C(generation) + f107:C(generation)
    continuous — log10(‖Δr‖) ~ sma + ecc + inc + log10(actual_dt_sec) + f107
                   + ap + C(generation) + f107:C(generation)

The bucket variant treats Δt as a 4-level factor (per the methodology
update folded from PR #9); the continuous variant keeps Δt as a real
covariate so the within-bucket spread carries information. F10.7 is
crossed with generation so the H3 "drag-dominant generations respond
more to solar activity" question gets a per-generation slope estimate.

Per-sat (sma, ecc, inc) are not on `all_runs.parquet` — they're parsed
from the first cached TLE for each sat via `sgp4.api.Satrec`, mirroring
the trick in `fig_constellation_map.py`.

Usage:
    python src/scripts/_regression.py \\
        --all-runs outputs/all_runs.parquet \\
        --corpus src/static/tles_cache.parquet \\
        --out outputs/regression_results.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from _style import pool_sparse_generations
from sgp4.api import Satrec
from statsmodels.regression.linear_model import RegressionResultsWrapper

BUCKET_FORMULA = (
    "log_dr ~ sma + ecc + inc + C(target_dt_sec) + f107 + ap + C(gen_pooled) + f107:C(gen_pooled)"
)
CONTINUOUS_FORMULA = (
    "log_dr ~ sma + ecc + inc + log_dt + f107 + ap + C(gen_pooled) + f107:C(gen_pooled)"
)
PROPAGATORS = (("sgp4", "dr_sgp4_km"), ("hifi", "dr_hifi_km"))


def per_sat_elements(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per norad_id with ``sma_km, ecc, inc_deg``.

    Reuses the F1 approach (`fig_constellation_map._per_sat_elements`):
    pick the earliest cached TLE per sat, parse line2 via sgp4 to
    recover eccentricity and inclination, take semi-major axis from the
    `sma_i_km` column the corpus already carries.
    """
    first = corpus.sort_values(["norad_id", "epoch_i"]).drop_duplicates(
        subset="norad_id", keep="first"
    )
    eccs = np.empty(len(first), dtype=float)
    incs = np.empty(len(first), dtype=float)
    for idx, (l1, l2) in enumerate(zip(first["line1_i"], first["line2_i"], strict=True)):
        sat = Satrec.twoline2rv(l1, l2)
        eccs[idx] = sat.ecco
        incs[idx] = math.degrees(sat.inclo)
    return pd.DataFrame(
        {
            "norad_id": first["norad_id"].to_numpy(),
            "sma": first["sma_i_km"].to_numpy(),
            "ecc": eccs,
            "inc": incs,
        }
    )


def build_design(all_runs: pd.DataFrame, corpus: pd.DataFrame, error_col: str) -> pd.DataFrame:
    """Join elements onto the run frame and add `log_dr`, `log_dt`.

    Drops rows where the error column is non-positive (log undefined).
    Applies the generation pooling so the regression's categorical
    `gen_pooled` matches the figures' visual convention.
    """
    elements = per_sat_elements(corpus)
    df = all_runs.merge(elements, on="norad_id", how="left", validate="many_to_one")
    df = df[df[error_col] > 0].copy()
    df, _ = pool_sparse_generations(df)
    df["log_dr"] = np.log10(df[error_col])
    df["log_dt"] = np.log10(df["actual_dt_sec"])
    return df


def fit_to_rows(
    result: RegressionResultsWrapper,
    model_name: str,
    propagator: str,
) -> list[dict]:
    """Flatten a statsmodels result into one row per coefficient."""
    ci = result.conf_int()
    rows = []
    for name in result.params.index:
        rows.append(
            {
                "model": model_name,
                "propagator": propagator,
                "predictor": name,
                "coefficient": float(result.params[name]),
                "std_err": float(result.bse[name]),
                "p_value": float(result.pvalues[name]),
                "ci_lo": float(ci.loc[name, 0]),
                "ci_hi": float(ci.loc[name, 1]),
                "r_squared": float(result.rsquared),
                "n_obs": int(result.nobs),
            }
        )
    return rows


def run_all_regressions(all_runs: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Fit (bucket, continuous) × (SGP4, hifi) and return a tidy frame."""
    rows: list[dict] = []
    for prop, error_col in PROPAGATORS:
        design = build_design(all_runs, corpus, error_col)
        bucket_fit = smf.ols(BUCKET_FORMULA, data=design).fit()
        rows.extend(fit_to_rows(bucket_fit, "bucket", prop))
        continuous_fit = smf.ols(CONTINUOUS_FORMULA, data=design).fit()
        rows.extend(fit_to_rows(continuous_fit, "continuous", prop))
    return pd.DataFrame(rows)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-runs", type=Path, default=Path("outputs/all_runs.parquet"))
    parser.add_argument("--corpus", type=Path, default=Path("src/static/tles_cache.parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/regression_results.csv"))
    args = parser.parse_args()

    all_runs = pd.read_parquet(args.all_runs)
    corpus = pd.read_parquet(args.corpus)
    results = run_all_regressions(all_runs, corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
