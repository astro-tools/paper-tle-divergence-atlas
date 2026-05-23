"""Aggregate per-run sweep parquets into a single analysis-ready frame.

After `make sweep` finishes, every successful run has a one-row
`comparison.parquet` recorded as its manifest entry's `extra_outputs`
under the key `comparison` (written by
`sweep.run_sweep._postprocess_run`, the gmat-sweep postprocess hook).
This module loads them all via `gmat_sweep.lazy_extra_outputs`,
filters to entries the sweep marked `ok`, and joins the corpus columns
the figure scripts need (notably `generation`).

Pipeline:

    per_run = lazy_extra_outputs(manifest_path, "comparison")
    per_run = per_run[per_run["__status"] == "ok"].drop(columns="__status")
    corpus = pd.read_parquet(tles)[["run_id", *CARRIED]]
    all_runs = per_run.merge(corpus, on="run_id", how="left", validate="1:1")

`lazy_extra_outputs` is manifest-driven (it walks the manifest's
`extra_outputs` map, not the filesystem), so a stray
`comparison.parquet` left in an old per-run directory by an earlier
batch is never picked up — only the entries the current manifest
references land in the aggregate.

The join key is `run_id` — the row index of `src/static/tles_cache.parquet`
after `reset_index(drop=True)` in `sweep.run_sweep.main`. Every run has
a unique run_id and every corpus row maps to exactly one run, so the
1:1 validate catches both "duplicate per-run frames" (which would mean
two `comparison.parquet` files for the same run_id — a bug) and
"corpus has duplicate run_ids" (which would mean
`reset_index(drop=True)` is being skipped upstream).

Failed runs do not appear in `outputs/`, so they're absent from
`all_runs.parquet`. The manifest is the source of truth for failure
accounting; see `sweep.sweep_stats`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import pandas as pd
from gmat_sweep import lazy_extra_outputs

CARRIED_COLUMNS: Final = ("generation", "dry_mass_kg", "drag_area_m2", "srp_area_m2")


def _load_per_run(manifest_path: Path) -> pd.DataFrame:
    """Stream every ok manifest entry's `comparison` extra-output into one frame.

    `lazy_extra_outputs` returns a `run_id`-indexed DataFrame with one
    row per ok run and a NaN-filled marker row (with `__status` set) for
    every failed/skipped entry. We drop the markers here, then move
    `run_id` from the index to a column so the downstream merge can use
    it as a join key.
    """
    frame = lazy_extra_outputs(manifest_path, "comparison")
    if frame.empty:
        raise FileNotFoundError(
            f"manifest {manifest_path} produced an empty per-run frame; "
            f"did the sweep produce any successful runs with a comparison output?"
        )
    ok_mask = frame["__status"] == "ok"
    if not ok_mask.any():
        raise FileNotFoundError(
            f"manifest {manifest_path} has no ok entries with a comparison output"
        )
    n_failed = int((~ok_mask).sum())
    if n_failed:
        print(
            f"aggregate: skipping {n_failed} non-ok manifest entr(ies)",
            file=sys.stderr,
        )
    per_run = frame.loc[ok_mask].drop(columns="__status")
    return per_run.reset_index()


def _load_corpus_subset(tles_path: Path) -> pd.DataFrame:
    corpus = pd.read_parquet(tles_path).reset_index(drop=True)
    subset = corpus[list(CARRIED_COLUMNS)].copy()
    subset.insert(0, "run_id", subset.index.astype(int))
    return subset


def aggregate(tles_path: Path, manifest_path: Path) -> pd.DataFrame:
    """Concat per-run comparison parquets and join the carried corpus columns."""
    per_run = _load_per_run(manifest_path)
    corpus_subset = _load_corpus_subset(tles_path)

    # validate="1:1" catches: (a) duplicate run_id in the per-run frame —
    # two comparison.parquet files for the same run_id, a bug; (b)
    # duplicate run_ids in the corpus subset — tle_pipeline produced a
    # non-unique corpus and the figure scripts can't safely groupby on
    # the carried columns.
    all_runs = per_run.merge(
        corpus_subset,
        on="run_id",
        how="left",
        validate="1:1",
    )

    # A per-run row with no corpus match means --tles came from a
    # different sweep batch than the manifest. Fail loudly rather than
    # ship a frame with NaN `generation` to the figure scripts.
    unmatched = all_runs["generation"].isna()
    if unmatched.any():
        n = int(unmatched.sum())
        raise RuntimeError(
            f"{n} per-run row(s) had no corpus match on run_id; "
            f"--tles must come from the same sweep batch as the manifest"
        )
    return all_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tles",
        type=Path,
        required=True,
        help="Path to the cached TLE-pair corpus (src/static/tles_cache.parquet)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to sweep/manifest.jsonl — drives the iteration",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output Parquet path (conventionally outputs/all_runs.parquet)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_runs = aggregate(args.tles, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_runs.to_parquet(args.out, index=False)
    print(
        f"wrote {args.out}: {len(all_runs)} rows, {len(all_runs.columns)} cols",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
