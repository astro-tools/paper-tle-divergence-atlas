"""Per-cell hi-fid-vs-SGP4 win fractions with sat-level bootstrap CIs.

For every pair (TLE_i, TLE_j) in ``outputs/all_runs.parquet`` we compare
the high-fidelity propagator's end-state error against SGP4's, defining
``high-fid beats SGP4`` on two metrics:

* 3D L2 -- ``dr_hifi_km < dr_sgp4_km``;
* along-track -- ``|dr_hifi_along_km| < |dr_sgp4_along_km|``.

The first metric is the per-pair distance below the y = x diagonal of
Figure 4 (the H2 propagator-skill differential). The second probes
whether the L2 readout is being dominated by the along-track component
that Figure 6 shows is two decades above the transverse components at
every Δt bucket.

For each cell -- ``(altitude shell, target Δt bucket)`` -- we report the
point estimate of each fraction together with a 95% percentile bootstrap
confidence interval over 1,000 satellite-level resamples (seed 42), using
the same ``bootstrap_by_sat`` infrastructure as the §3.7.1 power-law CIs
in Table 3. Pooled-per-Δt rows aggregate across all shells and
generations and recover the population-level win fractions quoted in §5.

Output JSON schema (one block per cell / pooled row):

    {
        "pooled_by_dt": [
            {
                "target_dt_sec": 21600,
                "n_pairs": 5421,
                "hifi_wins_l2":    {"point": ..., "ci_95": [...]},
                "hifi_wins_along": {"point": ..., "ci_95": [...]}
            },
            ...
        ],
        "by_cell": [
            {
                "alt_shell": "540",
                "target_dt_sec": 21600,
                "n_pairs": 1957,
                "gens_present": ["v1.x"],
                "hifi_wins_l2":    {"point": ..., "ci_95": [...]},
                "hifi_wins_along": {"point": ..., "ci_95": [...]}
            },
            ...
        ]
    }

The booktabs ``.tex`` fragment is the §4.2 main-body table: one row per
(shell, Δt) cell, with the 3D L2 win fraction and the along-track win
fraction side by side and their 95% CIs underneath. Cells thinner than
``MIN_CELL_PAIRS`` are flagged in the JSON note but omitted from the
table -- the prose handles those by pooling generations or by referring
to the parent shell only.

Usage:
    python src/scripts/_propagator_wins.py \\
        --all-runs outputs/all_runs.parquet \\
        --json-out outputs/propagator_wins.json \\
        --table-out src/tex/tables/tab_propagator_wins.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from _style import (
    ALT_SHELL_ORDER,
    BUCKET_LABELS,
    BUCKET_SECONDS,
    bootstrap_by_sat,
    pool_sparse_generations,
)

N_BOOTSTRAP: Final = 1000
BOOTSTRAP_SEED: Final = 42
# Below this pair count we suppress the cell from the table -- the
# bootstrap CI becomes uninformative ([0, 1] in the limit) and the row
# would just be visual noise. Cells flagged here are still in the JSON
# so a downstream consumer can inspect them.
MIN_CELL_PAIRS: Final = 30


def _wins(cell: pd.DataFrame) -> dict[str, float]:
    """Bare estimator -- fractions of pairs where hi-fid beats SGP4.

    Both fractions are well-defined on any non-empty cell; the
    along-track variant uses absolute values because ``dr_*_along_km``
    is the signed RSW component and sign cancellation across pairs
    would otherwise misrepresent the propagator-skill question.
    """
    return {
        "hifi_wins_l2": float((cell["dr_hifi_km"] < cell["dr_sgp4_km"]).mean()),
        "hifi_wins_along": float(
            (cell["dr_hifi_along_km"].abs() < cell["dr_sgp4_along_km"].abs()).mean()
        ),
    }


def _block(cell: pd.DataFrame, *, label: dict[str, object]) -> dict:
    """Bootstrap one cell or pooled bucket into a JSON-ready block.

    ``label`` carries the cell-identifying keys (``alt_shell`` /
    ``target_dt_sec`` for ``by_cell`` rows, ``target_dt_sec`` alone for
    ``pooled_by_dt`` rows). ``gens_present`` is added on the per-cell
    path because the §4.2 prose splits its reading by cohort.
    """
    if cell.empty:
        return {**label, "n_pairs": 0, "hifi_wins_l2": None, "hifi_wins_along": None}
    point, cis = bootstrap_by_sat(cell, _wins, n_resamples=N_BOOTSTRAP, rng_seed=BOOTSTRAP_SEED)
    out: dict[str, object] = {
        **label,
        "n_pairs": int(len(cell)),
        "hifi_wins_l2": {
            "point": point["hifi_wins_l2"],
            "ci_95": [cis["hifi_wins_l2"][0], cis["hifi_wins_l2"][1]],
        },
        "hifi_wins_along": {
            "point": point["hifi_wins_along"],
            "ci_95": [cis["hifi_wins_along"][0], cis["hifi_wins_along"][1]],
        },
    }
    if "alt_shell" in label:
        gens = sorted(cell["generation"].unique().tolist())
        out["gens_present"] = gens
    return out


def compute(all_runs: pd.DataFrame) -> dict:
    """Top-level driver: pooled-per-Δt rows + per-cell + per-cohort rows.

    Pools v1.0 into v1.x via ``pool_sparse_generations`` so the by-cohort
    breakdown matches the figure convention (every v1.0 cell sits below
    POOL_MIN_SAMPLES in the corpus). The shell-level ``by_cell`` rows are
    preserved as the main-body §4.2 input; ``by_cell_gen`` carries the
    cohort decomposition that §5.1.1 discusses.
    """
    pooled_runs, _pool_note = pool_sparse_generations(all_runs)

    pooled: list[dict] = []
    for bucket in BUCKET_SECONDS:
        cell = pooled_runs[pooled_runs["target_dt_sec"] == bucket]
        pooled.append(_block(cell, label={"target_dt_sec": bucket}))

    by_cell: list[dict] = []
    by_cell_gen: list[dict] = []
    for shell in ALT_SHELL_ORDER:
        for bucket in BUCKET_SECONDS:
            cell = pooled_runs[
                (pooled_runs["alt_shell"] == shell)
                & (pooled_runs["target_dt_sec"] == bucket)
            ]
            by_cell.append(_block(cell, label={"alt_shell": shell, "target_dt_sec": bucket}))

            # Cohort breakdown within the same (shell, Δt) cell. Iterate
            # generations in the order the cell carries them (v1.x first,
            # v2-mini second), so the JSON / table read top-to-bottom by
            # generation maturity.
            for gen in sorted(cell["gen_pooled"].unique().tolist()):
                gen_cell = cell[cell["gen_pooled"] == gen]
                by_cell_gen.append(
                    _block(
                        gen_cell,
                        label={
                            "alt_shell": shell,
                            "target_dt_sec": bucket,
                            "gen_pooled": gen,
                        },
                    )
                )

    return {
        "n_resamples": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "min_cell_pairs_for_table": MIN_CELL_PAIRS,
        "metric_definitions": {
            "hifi_wins_l2": "dr_hifi_km < dr_sgp4_km",
            "hifi_wins_along": "|dr_hifi_along_km| < |dr_sgp4_along_km|",
        },
        "pooled_by_dt": pooled,
        "by_cell": by_cell,
        "by_cell_gen": by_cell_gen,
    }


def _format_fraction(payload: dict | None) -> str:
    """Render ``{point, ci_95}`` as ``28.6\\% [24.1, 33.2]``."""
    if payload is None:
        return "---"
    point = payload["point"] * 100.0
    lo = payload["ci_95"][0] * 100.0
    hi = payload["ci_95"][1] * 100.0
    if not np.isfinite(lo) or not np.isfinite(hi):
        return f"{point:.1f}\\%"
    return f"{point:.1f}\\% [{lo:.1f}, {hi:.1f}]"


def render_by_gen_table(payload: dict) -> str:
    """Booktabs fragment of the per-(shell × Δt × gen) win fractions.

    Lives in §5.1.1 alongside the cohort-mechanism discussion. The
    structure interleaves a (shell, Δt) header row over its
    cohort sub-rows so a reader can see the 60.2% shell-level number
    and the per-cohort decomposition that produces it on one
    visual block.
    """
    rows: list[str] = [
        "% Generated by src/scripts/_propagator_wins.py -- do not edit by hand.",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Shell & $\Delta t$ & Cohort & $n_{\mathrm{pairs}}$ & "
        r"\multicolumn{1}{c}{hi-fid wins, 3D L$_{2}$ [95\% CI]} & "
        r"\multicolumn{1}{c}{hi-fid wins, along-track [95\% CI]} \\",
        r"(km) & & & & & \\",
        r"\midrule",
    ]
    last_shell: str | None = None
    last_bucket: int | None = None
    for block in payload["by_cell_gen"]:
        if block["n_pairs"] < payload["min_cell_pairs_for_table"]:
            continue
        if last_shell is not None and block["alt_shell"] != last_shell:
            rows.append(r"\midrule")
        shell_cell = block["alt_shell"] if block["alt_shell"] != last_shell else ""
        bucket_cell = (
            BUCKET_LABELS[block["target_dt_sec"]]
            if (
                block["alt_shell"] != last_shell
                or block["target_dt_sec"] != last_bucket
            )
            else ""
        )
        last_shell = block["alt_shell"]
        last_bucket = block["target_dt_sec"]
        rows.append(
            f"{shell_cell} & {bucket_cell} & {block['gen_pooled']} "
            f"& {block['n_pairs']:,} & "
            f"{_format_fraction(block['hifi_wins_l2'])} & "
            f"{_format_fraction(block['hifi_wins_along'])} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    return "\n".join(rows) + "\n"


def render_table(payload: dict) -> str:
    """Booktabs fragment for ``\\input{tables/tab_propagator_wins.tex}``."""
    rows: list[str] = [
        "% Generated by src/scripts/_propagator_wins.py -- do not edit by hand.",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Shell & $\Delta t$ & $n_{\mathrm{pairs}}$ & "
        r"\multicolumn{1}{c}{hi-fid wins, 3D L$_{2}$ [95\% CI]} & "
        r"\multicolumn{1}{c}{hi-fid wins, along-track [95\% CI]} \\",
        r"(km) & & & & \\",
        r"\midrule",
    ]
    last_shell: str | None = None
    for block in payload["by_cell"]:
        if block["n_pairs"] < payload["min_cell_pairs_for_table"]:
            continue
        if last_shell is not None and block["alt_shell"] != last_shell:
            rows.append(r"\midrule")
        last_shell = block["alt_shell"]
        rows.append(
            f"{block['alt_shell']} & {BUCKET_LABELS[block['target_dt_sec']]} "
            f"& {block['n_pairs']:,} & {_format_fraction(block['hifi_wins_l2'])} "
            f"& {_format_fraction(block['hifi_wins_along'])} \\\\"
        )
    rows.append(r"\midrule")
    for pooled_row in payload["pooled_by_dt"]:
        rows.append(
            f"\\textit{{pooled}} & {BUCKET_LABELS[pooled_row['target_dt_sec']]} "
            f"& {pooled_row['n_pairs']:,} & "
            f"{_format_fraction(pooled_row['hifi_wins_l2'])} & "
            f"{_format_fraction(pooled_row['hifi_wins_along'])} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    return "\n".join(rows) + "\n"


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-runs", type=Path, default=Path("outputs/all_runs.parquet"))
    parser.add_argument("--json-out", type=Path, default=Path("outputs/propagator_wins.json"))
    parser.add_argument(
        "--table-out",
        type=Path,
        default=Path("src/tex/tables/tab_propagator_wins.tex"),
    )
    parser.add_argument(
        "--by-gen-table-out",
        type=Path,
        default=Path("src/tex/tables/tab_propagator_wins_by_gen.tex"),
    )
    args = parser.parse_args()

    all_runs = pd.read_parquet(args.all_runs)
    payload = compute(all_runs)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))

    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.write_text(render_table(payload))

    args.by_gen_table_out.parent.mkdir(parents=True, exist_ok=True)
    args.by_gen_table_out.write_text(render_by_gen_table(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
