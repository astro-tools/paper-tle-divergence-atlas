"""Aggregate per-run sweep parquets into a single analysis-ready frame.

After `make sweep` finishes, every successful run leaves an
`outputs/run_<id>.parquet` with the 17-column per-run schema. This
module concatenates them into `outputs/all_runs.parquet` and joins the
corpus columns the figure scripts need (notably `generation`).

Pipeline:

    paths = sorted(output_dir.glob("run_*.parquet"))
    per_run = pd.concat(read_parquet(p) for p in paths)
    corpus = pd.read_parquet(tles)[KEY + CARRIED]
    all_runs = per_run.merge(corpus, on=KEY, how="left", validate="1:1")

`gmat_sweep.lazy_multiindex` works for streaming workflows but the
per-run parquets here carry one row each, so a straight concat is the
simpler path.

The join key is `(norad_id, target_dt_sec, epoch_i)`. Within a
`(norad_id, target_dt_sec)` group the corpus has ~20 epochs per sat per
Δt bucket, so `epoch_i` is required for uniqueness. The per-run frame
calls it `t_i` (renamed by `run_sweep._postprocess_run`); we rename it
back on the corpus side before merging.

`actual_dt_sec` and `alt_shell` are already on the per-run frame and
deliberately not carried again — bringing them across would force
`_x`/`_y` suffixes after the merge for no gain.

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

JOIN_KEYS: Final = ("norad_id", "target_dt_sec", "t_i")
CARRIED_COLUMNS: Final = ("generation", "dry_mass_kg", "drag_area_m2", "srp_area_m2")


def _load_per_run(output_dir: Path) -> pd.DataFrame:
    paths = sorted(output_dir.glob("run_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no run_*.parquet files under {output_dir}; "
            f"did the sweep finish, or is --output-dir wrong?"
        )
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _load_corpus_subset(tles_path: Path) -> pd.DataFrame:
    corpus = pd.read_parquet(tles_path)
    subset = corpus[["norad_id", "target_dt_sec", "epoch_i", *CARRIED_COLUMNS]].copy()
    return subset.rename(columns={"epoch_i": "t_i"})


def aggregate(output_dir: Path, tles_path: Path) -> pd.DataFrame:
    """Concat per-run parquets and join the carried corpus columns."""
    per_run = _load_per_run(output_dir)
    corpus_subset = _load_corpus_subset(tles_path)

    # validate="1:1" catches: (a) duplicate (norad_id, target_dt_sec, t_i)
    # in the per-run frame — which would mean two run_<id>.parquet files
    # for the same pair, i.e. a bug; (b) duplicate keys in the corpus
    # subset — which means tle_pipeline produced a non-unique corpus and
    # the figure scripts can't safely groupby on the carried columns.
    all_runs = per_run.merge(
        corpus_subset,
        on=list(JOIN_KEYS),
        how="left",
        validate="1:1",
    )

    # A per-run row with no corpus match means --output-dir and --tles
    # came from different sweep batches. Fail loudly rather than ship a
    # frame with NaN `generation` to the figure scripts.
    unmatched = all_runs["generation"].isna()
    if unmatched.any():
        n = int(unmatched.sum())
        raise RuntimeError(
            f"{n} per-run row(s) had no corpus match on {list(JOIN_KEYS)}; "
            f"--output-dir and --tles must come from the same sweep batch"
        )
    return all_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing run_<id>.parquet files (the sweep's --output-dir)",
    )
    parser.add_argument(
        "--tles",
        type=Path,
        required=True,
        help="Path to the cached TLE-pair corpus (src/data/tles_cache.parquet)",
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
    all_runs = aggregate(args.output_dir, args.tles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_runs.to_parquet(args.out, index=False)
    print(
        f"wrote {args.out}: {len(all_runs)} rows, {len(all_runs.columns)} cols",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
