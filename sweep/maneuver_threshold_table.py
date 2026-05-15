"""Tabulate the maneuver-threshold sensitivity result.

Consumes the new augment frame produced by
``sweep.maneuver_threshold_sensitivity``
(``outputs/all_runs_maneuver_augment.parquet``), the 50 m / 100 m / 200 m
candidate corpora, the main-sweep aggregate
(``outputs/all_runs.parquet``), and the committed baseline
rejection-counts JSON (``src/static/maneuver_rejection_counts.json``).

Emits three artifacts:

- ``src/tex/tables/tab_maneuver_threshold.tex`` --- booktabs Appendix B
  table of per-``(alt_shell × Δt)`` baseline median ``|Δr|_hifi``, its
  95% sat-level bootstrap CI, and the 50 m / 200 m relative shifts. The
  PR description cites the max-abs-shift and the cell-fraction inside
  the baseline CI as headline numbers.
- ``src/tex/tables/tab_maneuver_rejections.tex`` --- booktabs Appendix A
  "fleet quietness" table sourced from the rejection-counts JSON at the
  baseline 100 m threshold. Reproducible from the committed JSON; the
  raw TLE cache (gitignored, ~100 MB) is not required to rebuild it.
- ``outputs/maneuver_threshold_summary.json`` --- the same numbers in
  machine form: pooled medians at each threshold, per-cell relative
  shifts, the fraction of cells whose 50 m and 200 m medians sit
  inside the baseline 95% CI, and a max-abs-shift summary.

The 50 m population is the filter of ``all_runs.parquet`` to pairs that
survive at the 50 m threshold (matched against the 50 m candidate
corpus by the same ``(norad_id, epoch_i, epoch_j)`` key the augment
driver uses). The 200 m population is ``all_runs.parquet`` ∪ augment.
The 100 m population is the unfiltered baseline. Inclusion
``50 m ⊂ 100 m ⊂ 200 m`` is verified empirically at the corpus-build
step and re-checked here for safety.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

BUCKET_LABELS: Final = {
    21_600: "6 h",
    86_400: "1 d",
    259_200: "3 d",
    604_800: "7 d",
}
BUCKET_ORDER: Final = (21_600, 86_400, 259_200, 604_800)
# The corpus parquets emitted by `sweep.tle_pipeline` use the column names
# below; the per-run frames produced by `sweep.run_sweep._postprocess_run`
# rename the same epochs to ``t_i`` / ``t_j``. `_key_set` accepts either
# convention transparently and emits the same triple.
PAIR_KEY: Final = ("norad_id", "epoch_i", "epoch_j")
_EPOCH_ALIASES: Final = {"epoch_i": "t_i", "epoch_j": "t_j"}

BOOTSTRAP_DRAWS: Final = 1_000
BOOTSTRAP_SEED: Final = 20260514
BOOTSTRAP_CI_LEVEL: Final = 0.95


# --- Population assembly ---------------------------------------------------


def _key_set(df: pd.DataFrame) -> set[tuple]:
    """Tuple-set on ``(norad_id, epoch_i, epoch_j)`` from a corpus or runs frame.

    Corpus parquets carry the epoch columns under the names ``epoch_i`` and
    ``epoch_j``; per-run frames rename them to ``t_i`` / ``t_j``. Resolve
    either layout transparently — the canonical key triple is unchanged.
    """
    cols = []
    for canonical in PAIR_KEY:
        if canonical in df.columns:
            cols.append(canonical)
        elif canonical in _EPOCH_ALIASES and _EPOCH_ALIASES[canonical] in df.columns:
            cols.append(_EPOCH_ALIASES[canonical])
        else:
            raise KeyError(f"frame is missing pair-key column {canonical}")
    return set(map(tuple, df[cols].itertuples(index=False, name=None)))


def filter_to_corpus(all_runs: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Filter `all_runs` to rows whose pair key is in `corpus`.

    The match is exact on ``(norad_id, epoch_i, epoch_j)``. Used to
    derive the 50 m population from the unfiltered ``all_runs.parquet``
    (the 50 m corpus is a strict subset of the 100 m baseline corpus,
    verified at build time). Both inputs may use the per-run column
    names ``t_i`` / ``t_j`` or the corpus names ``epoch_i`` / ``epoch_j``
    — `_key_set` normalises either.
    """
    keep = _key_set(corpus)
    # Resolve the epoch column names for the runs frame to whatever it carries.
    run_cols = []
    for canonical in PAIR_KEY:
        if canonical in all_runs.columns:
            run_cols.append(canonical)
        else:
            run_cols.append(_EPOCH_ALIASES.get(canonical, canonical))
    keys = all_runs[run_cols].apply(tuple, axis=1)
    return all_runs[keys.isin(keep)].copy()


def assemble_populations(
    all_runs: pd.DataFrame,
    augment: pd.DataFrame,
    corpus_50m: pd.DataFrame,
    corpus_200m: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return per-threshold populations: 50 m (filter), 100 m, 200 m.

    Verifies the corpus inclusion chain: every pair in
    ``all_runs.parquet`` must also be in the 200 m corpus, and every
    pair in the 50 m corpus must also be in ``all_runs``.
    """
    baseline_keys = _key_set(all_runs)
    if not _key_set(corpus_50m).issubset(baseline_keys):
        raise SystemExit(
            "50 m corpus contains pairs missing from all_runs.parquet; "
            "the 50 m corpus must be a subset of the 100 m baseline corpus.",
        )
    if not baseline_keys.issubset(_key_set(corpus_200m)):
        raise SystemExit(
            "100 m baseline corpus contains pairs missing from the 200 m corpus; "
            "the 100 m corpus must be a subset of the 200 m corpus.",
        )

    pop_50m = filter_to_corpus(all_runs, corpus_50m)
    pop_100m = all_runs
    # Augment carries the same schema as all_runs (joined corpus columns).
    pop_200m = pd.concat([all_runs, augment], ignore_index=True)
    return {"50m": pop_50m, "100m": pop_100m, "200m": pop_200m}


# --- Sat-level median bootstrap --------------------------------------------


def sat_level_bootstrap_median_ci(
    pairs: pd.DataFrame,
    metric: str = "dr_hifi_km",
    *,
    n_draws: int = BOOTSTRAP_DRAWS,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """``(median, ci_lo, ci_hi)`` from a satellite-level resample.

    Per-call: resample ``len(unique sats)`` sats with replacement, take
    every pair belonging to each resampled sat (with multiplicity if the
    same sat is drawn twice), compute the median of `metric`. Repeat
    `n_draws` times. The interval is a two-sided percentile range over
    the bootstrap distribution at confidence `ci_level`.

    Satellite-level (rather than pair-level) resampling preserves the
    within-sat correlation structure across Δt buckets --- a sat with
    an OD residual at epoch contributes a correlated bias to all of its
    pairs, and treating them as independent would shrink the CI
    artificially. This is the contract issue #34 (Theme G) wants for
    every per-cell bootstrap going forward; #31 lifts it forward for
    the threshold-sensitivity check only.
    """
    if pairs.empty:
        return float("nan"), float("nan"), float("nan")
    point = float(pairs[metric].median())

    by_sat = {nid: g[metric].to_numpy() for nid, g in pairs.groupby("norad_id", sort=False)}
    sat_ids = np.asarray(list(by_sat))
    if sat_ids.size == 0:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=float)
    for k in range(n_draws):
        resample = rng.choice(sat_ids, size=sat_ids.size, replace=True)
        values: list[np.ndarray] = []
        for nid in resample:
            values.append(by_sat[nid])
        draws[k] = float(np.median(np.concatenate(values)))

    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.percentile(draws, 100.0 * alpha))
    hi = float(np.percentile(draws, 100.0 * (1.0 - alpha)))
    return point, lo, hi


# --- Per-cell aggregation --------------------------------------------------


def per_cell_table(
    populations: dict[str, pd.DataFrame],
    *,
    metric: str = "dr_hifi_km",
    n_bootstrap: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Long table indexed by ``(alt_shell, target_dt_sec)``.

    Columns:
    ``n_100m, median_50m, median_100m, ci_lo_100m, ci_hi_100m, median_200m,
    shift_50m, shift_200m, in_ci_50m, in_ci_200m``.

    The bootstrap is computed on the 100 m baseline only; the 50 m and
    200 m point estimates are checked against that interval.
    """
    pop_baseline = populations["100m"]
    pop_50m = populations["50m"]
    pop_200m = populations["200m"]

    cells: list[dict] = []
    cell_keys = (
        pop_baseline.groupby(["alt_shell", "target_dt_sec"]).size().reset_index().drop(columns=0)
    )
    for _, row in cell_keys.iterrows():
        shell = row["alt_shell"]
        bucket = int(row["target_dt_sec"])

        base = pop_baseline[
            (pop_baseline["alt_shell"] == shell) & (pop_baseline["target_dt_sec"] == bucket)
        ]
        if base.empty:
            continue
        med_100, lo, hi = sat_level_bootstrap_median_ci(
            base,
            metric=metric,
            n_draws=n_bootstrap,
            seed=seed,
        )
        m50 = pop_50m[(pop_50m["alt_shell"] == shell) & (pop_50m["target_dt_sec"] == bucket)][
            metric
        ].median()
        m200 = pop_200m[(pop_200m["alt_shell"] == shell) & (pop_200m["target_dt_sec"] == bucket)][
            metric
        ].median()
        med_50 = float(m50) if not pd.isna(m50) else float("nan")
        med_200 = float(m200) if not pd.isna(m200) else float("nan")

        def _rel(x: float, baseline: float) -> float:
            if not np.isfinite(x) or baseline == 0.0:
                return float("nan")
            return (x - baseline) / baseline

        cells.append(
            {
                "alt_shell": shell,
                "target_dt_sec": bucket,
                "n_100m": int(len(base)),
                "median_50m_km": med_50,
                "median_100m_km": med_100,
                "ci_lo_100m_km": lo,
                "ci_hi_100m_km": hi,
                "median_200m_km": med_200,
                "shift_50m": _rel(med_50, med_100),
                "shift_200m": _rel(med_200, med_100),
                "in_ci_50m": (bool(lo <= med_50 <= hi) if np.isfinite(med_50) else False),
                "in_ci_200m": (bool(lo <= med_200 <= hi) if np.isfinite(med_200) else False),
            },
        )
    return pd.DataFrame(cells).sort_values(["alt_shell", "target_dt_sec"]).reset_index(drop=True)


# --- LaTeX rendering -------------------------------------------------------


def _fmt_shift(rel: float) -> str:
    if not np.isfinite(rel):
        return "---"
    return f"{rel * 100.0:+.1f}\\%"


def render_threshold_table(per_cell: pd.DataFrame) -> str:
    """booktabs LaTeX for Appendix B "Maneuver-threshold sensitivity".

    Five-column layout matching the CdA-sensitivity table's shape:
    Shell × Δt × baseline median (with bracket-bound 95% CI) × 50 m
    relative shift × 200 m relative shift. Per-cell rows are shell-
    blocked with ``\\midrule`` separators; a footnote-tied dagger marks
    cells where the alternative-threshold median sits *outside* the
    baseline 95% CI.
    """
    lines = [
        "% Generated by sweep.maneuver_threshold_table — do not edit by hand.",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Shell & $\\Delta t$ & "
        "$\\mathrm{med}\\,|\\Delta\\mathbf{r}|_{\\mathrm{hifi}}$ "
        "(100 m, 95\\% CI) & "
        "50 m shift & "
        "200 m shift \\\\",
        "(km) & & (km) & (vs.\\ baseline) & (vs.\\ baseline) \\\\",
        "\\midrule",
    ]
    shells = sorted(per_cell["alt_shell"].unique())
    for i, shell in enumerate(shells):
        block = per_cell[per_cell["alt_shell"] == shell]
        for _, row in block.iterrows():
            bucket = BUCKET_LABELS.get(int(row["target_dt_sec"]), f"{int(row['target_dt_sec'])} s")
            med = float(row["median_100m_km"])
            lo = float(row["ci_lo_100m_km"])
            hi = float(row["ci_hi_100m_km"])
            shift_50 = _fmt_shift(float(row["shift_50m"]))
            shift_200 = _fmt_shift(float(row["shift_200m"]))
            mark_50 = "" if bool(row["in_ci_50m"]) else "\\textsuperscript{$\\dagger$}"
            mark_200 = "" if bool(row["in_ci_200m"]) else "\\textsuperscript{$\\dagger$}"
            baseline_cell = (
                f"{med:.2f} [{lo:.2f}, {hi:.2f}]"
                if np.isfinite(med) and np.isfinite(lo) and np.isfinite(hi)
                else "---"
            )
            lines.append(
                f"{shell} & {bucket} & {baseline_cell} & "
                f"{shift_50}{mark_50} & {shift_200}{mark_200} \\\\",
            )
        if i < len(shells) - 1:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_rejections_table(rejection_counts: dict) -> str:
    """booktabs LaTeX for Appendix A rejection-count table at the baseline.

    Columns: shell × Δt × n_candidates × n_survivors × n_rejected ×
    rejection % (n_rejected / n_candidates). Shell-blocked rows with
    ``\\midrule`` separators between shells. A pooled row at the bottom
    totals candidates/survivors/rejected across all cells.
    """
    cells = sorted(
        rejection_counts["cells"],
        key=lambda c: (str(c["alt_shell"]), int(c["target_dt_sec"])),
    )
    totals = rejection_counts.get("totals", {})

    lines = [
        "% Generated by sweep.maneuver_threshold_table — do not edit by hand.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Shell & $\\Delta t$ & Candidates & Survivors & Rejected & Rejected (\\%) \\\\",
        "(km) & & & & & \\\\",
        "\\midrule",
    ]
    shells = sorted({c["alt_shell"] for c in cells})
    for i, shell in enumerate(shells):
        block = [c for c in cells if c["alt_shell"] == shell]
        for c in block:
            bucket = BUCKET_LABELS.get(int(c["target_dt_sec"]), f"{int(c['target_dt_sec'])} s")
            n_c = int(c["n_candidates"])
            n_s = int(c["n_survivors"])
            n_r = int(c["n_rejected"])
            pct = (n_r / n_c * 100.0) if n_c else float("nan")
            pct_cell = f"{pct:.1f}\\%" if np.isfinite(pct) else "---"
            lines.append(
                f"{shell} & {bucket} & {n_c:,} & {n_s:,} & {n_r:,} & {pct_cell} \\\\".replace(
                    ",",
                    "{,}",
                ),
            )
        if i < len(shells) - 1:
            lines.append("\\midrule")

    if totals:
        n_c = int(totals.get("n_candidates", 0))
        n_s = int(totals.get("n_survivors", 0))
        n_r = int(totals.get("n_rejected", 0))
        pct = (n_r / n_c * 100.0) if n_c else float("nan")
        pct_cell = f"{pct:.1f}\\%" if np.isfinite(pct) else "---"
        lines.append("\\midrule")
        lines.append(
            f"Total & & {n_c:,} & {n_s:,} & {n_r:,} & {pct_cell} \\\\".replace(",", "{,}"),
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


# --- Headline summary ------------------------------------------------------


def compute_summary(
    per_cell: pd.DataFrame,
    populations: dict[str, pd.DataFrame],
    rejection_counts: dict,
    *,
    metric: str = "dr_hifi_km",
) -> dict:
    """Headline numbers consumed by the PR body and the §B prose.

    Returns:
    - Pooled per-threshold median across cells (the metric reported in
      the caption opening).
    - Pooled relative shifts at 50 m and 200 m (the "well-inside" claim).
    - Max absolute relative shift and the cell it occurs in.
    - Fraction of cells whose 50 m / 200 m medians sit inside the
      baseline 95% sat-level bootstrap CI.
    - Surviving-pair counts per threshold population (n_50, n_100,
      n_200, n_augment).
    """
    pooled_50 = float(populations["50m"][metric].median())
    pooled_100 = float(populations["100m"][metric].median())
    pooled_200 = float(populations["200m"][metric].median())

    def pooled_shift(x: float) -> float:
        return (x - pooled_100) / pooled_100 if pooled_100 else float("nan")

    n_cells = len(per_cell)
    in_ci_50 = int(per_cell["in_ci_50m"].sum())
    in_ci_200 = int(per_cell["in_ci_200m"].sum())

    # Max |shift| and the offending cell.
    def _argmax_abs(col: str) -> dict:
        if per_cell.empty:
            return {"alt_shell": None, "target_dt_sec": None, "shift": float("nan")}
        idx = per_cell[col].abs().idxmax()
        row = per_cell.loc[idx]
        return {
            "alt_shell": str(row["alt_shell"]),
            "target_dt_sec": int(row["target_dt_sec"]),
            "shift": float(row[col]),
        }

    return {
        "metric": metric,
        "n_pairs": {
            "50m": int(len(populations["50m"])),
            "100m": int(len(populations["100m"])),
            "200m": int(len(populations["200m"])),
            "augment": int(len(populations["200m"]) - len(populations["100m"])),
        },
        "pooled_median_km": {
            "50m": pooled_50,
            "100m": pooled_100,
            "200m": pooled_200,
        },
        "pooled_relative_shift": {
            "50m": pooled_shift(pooled_50),
            "200m": pooled_shift(pooled_200),
        },
        "n_cells": n_cells,
        "cells_in_ci": {
            "50m": in_ci_50,
            "200m": in_ci_200,
        },
        "max_abs_relative_shift": {
            "50m": _argmax_abs("shift_50m"),
            "200m": _argmax_abs("shift_200m"),
        },
        "rejection_total": {
            "n_candidates": int(rejection_counts.get("totals", {}).get("n_candidates", 0)),
            "n_survivors": int(rejection_counts.get("totals", {}).get("n_survivors", 0)),
            "n_rejected": int(rejection_counts.get("totals", {}).get("n_rejected", 0)),
        },
        "by_cell": per_cell.to_dict(orient="records"),
    }


# --- End-to-end ------------------------------------------------------------


def build(
    all_runs_path: Path,
    augment_path: Path,
    corpus_50m_path: Path,
    corpus_200m_path: Path,
    rejection_counts_path: Path,
    *,
    metric: str = "dr_hifi_km",
    n_bootstrap: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[str, str, dict]:
    """Load every input, return ``(rejections_tex, threshold_tex, summary)``."""
    all_runs = pd.read_parquet(all_runs_path)
    augment = pd.read_parquet(augment_path)
    corpus_50m = pd.read_parquet(corpus_50m_path)
    corpus_200m = pd.read_parquet(corpus_200m_path)
    rejection_counts = json.loads(rejection_counts_path.read_text())

    populations = assemble_populations(all_runs, augment, corpus_50m, corpus_200m)
    per_cell = per_cell_table(populations, metric=metric, n_bootstrap=n_bootstrap, seed=seed)
    threshold_tex = render_threshold_table(per_cell)
    rejections_tex = render_rejections_table(rejection_counts)
    summary = compute_summary(per_cell, populations, rejection_counts, metric=metric)
    return rejections_tex, threshold_tex, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-runs",
        type=Path,
        default=Path("outputs/all_runs.parquet"),
    )
    parser.add_argument(
        "--augment",
        type=Path,
        default=Path("outputs/all_runs_maneuver_augment.parquet"),
    )
    parser.add_argument(
        "--corpus-50m",
        type=Path,
        default=Path("outputs/tles_cache_50m.parquet"),
    )
    parser.add_argument(
        "--corpus-200m",
        type=Path,
        default=Path("outputs/tles_cache_200m.parquet"),
    )
    parser.add_argument(
        "--rejection-counts",
        type=Path,
        default=Path("src/static/maneuver_rejection_counts.json"),
    )
    parser.add_argument(
        "--threshold-table-out",
        type=Path,
        default=Path("src/tex/tables/tab_maneuver_threshold.tex"),
    )
    parser.add_argument(
        "--rejections-table-out",
        type=Path,
        default=Path("src/tex/tables/tab_maneuver_rejections.tex"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/maneuver_threshold_summary.json"),
    )
    parser.add_argument("--metric", default="dr_hifi_km")
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rejections_tex, threshold_tex, summary = build(
        args.all_runs,
        args.augment,
        args.corpus_50m,
        args.corpus_200m,
        args.rejection_counts,
        metric=args.metric,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    for path in (args.threshold_table_out, args.rejections_table_out, args.summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.threshold_table_out.write_text(threshold_tex, encoding="utf-8")
    args.rejections_table_out.write_text(rejections_tex, encoding="utf-8")
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"wrote {args.threshold_table_out}, "
        f"{args.rejections_table_out}, and {args.summary_out}; "
        f"pooled |Δr|_hifi shift 50 m={summary['pooled_relative_shift']['50m'] * 100:+.1f}%, "
        f"200 m={summary['pooled_relative_shift']['200m'] * 100:+.1f}%; "
        f"cells inside CI: 50 m {summary['cells_in_ci']['50m']}/{summary['n_cells']}, "
        f"200 m {summary['cells_in_ci']['200m']}/{summary['n_cells']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
