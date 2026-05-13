"""Aggregate per-run sweep parquets into a single analysis-ready frame.

After `make sweep` finishes, every successful run leaves an
`outputs/run_<id>.parquet` with the 17-column per-run schema. This
module concatenates them into `outputs/all_runs.parquet` and joins the
corpus columns the figure scripts need (notably `generation`).

Pipeline:

    manifest = Manifest.load(manifest_path)
    ok_ids = [e.run_id for e in manifest.entries if e.status == "ok"]
    per_run = pd.concat(read_parquet(output_dir / f"run_{rid}.parquet") for rid in ok_ids)
    corpus = pd.read_parquet(tles)[KEY + CARRIED]
    all_runs = per_run.merge(corpus, on=KEY, how="left", validate="1:1")

The per-run comparison parquets are written by
`run_sweep._postprocess_run` outside the gmat-sweep manifest's
`output_paths`, so `gmat_sweep.lazy_multiindex` does not see them.
What we do instead: drive the iteration from `Manifest.entries`,
keeping only `status == "ok"` runs. This rules out stale
`run_<id>.parquet` files from previous batches contaminating the
aggregation on a resumed sweep — they exist on disk but are not
referenced by the current manifest.

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
from gmat_sweep import Manifest

JOIN_KEYS: Final = ("norad_id", "target_dt_sec", "t_i")
CARRIED_COLUMNS: Final = ("generation", "dry_mass_kg", "drag_area_m2", "srp_area_m2")


def _load_per_run(output_dir: Path, manifest_path: Path) -> pd.DataFrame:
    """Concat run_<id>.parquet for every manifest entry with `status == "ok"`.

    Manifest-driven rather than glob-driven so a stale `run_<id>.parquet`
    from an earlier batch on the same `output_dir` cannot land in the
    aggregation. Ok entries whose comparison parquet is missing from
    disk (post-process gap) are surfaced as a warning and skipped.
    """
    manifest = Manifest.load(manifest_path)
    ok_run_ids = [e.run_id for e in manifest.entries if e.status == "ok"]
    if not ok_run_ids:
        raise FileNotFoundError(
            f"manifest {manifest_path} has no ok entries; "
            f"did the sweep produce any successful runs?"
        )

    frames: list[pd.DataFrame] = []
    missing: list[int] = []
    for run_id in ok_run_ids:
        path = output_dir / f"run_{run_id}.parquet"
        if not path.exists():
            missing.append(run_id)
            continue
        frames.append(pd.read_parquet(path))

    if missing:
        preview = ", ".join(str(rid) for rid in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        print(
            f"warning: {len(missing)} ok manifest entr(ies) had no "
            f"run_<id>.parquet under {output_dir} (postprocess gap?): "
            f"{preview}{suffix}",
            file=sys.stderr,
        )
    if not frames:
        raise FileNotFoundError(
            f"none of the {len(ok_run_ids)} ok manifest entries had a "
            f"run_<id>.parquet under {output_dir}"
        )
    return pd.concat(frames, ignore_index=True)


def _load_corpus_subset(tles_path: Path) -> pd.DataFrame:
    corpus = pd.read_parquet(tles_path)
    subset = corpus[["norad_id", "target_dt_sec", "epoch_i", *CARRIED_COLUMNS]].copy()
    return subset.rename(columns={"epoch_i": "t_i"})


def aggregate(output_dir: Path, tles_path: Path, manifest_path: Path) -> pd.DataFrame:
    """Concat per-run parquets and join the carried corpus columns."""
    per_run = _load_per_run(output_dir, manifest_path)
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
    all_runs = aggregate(args.output_dir, args.tles, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_runs.to_parquet(args.out, index=False)
    print(
        f"wrote {args.out}: {len(all_runs)} rows, {len(all_runs.columns)} cols",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
