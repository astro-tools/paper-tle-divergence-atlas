"""Stratified 1,000-pair sensitivity subset of the locked corpus.

Selects a deterministic, stratified-random subset of pair `run_id`s from
`src/static/tles_cache.parquet` for downstream sensitivity studies (#28
CdA ±20% on v2-mini, #31 maneuver-threshold {50 m, 200 m}). Both studies
need a much smaller workload than the full 24,641-pair sweep but want
the same `(alt_shell × generation × target_dt_sec)` cell coverage as the
main results, so they share one canonical subset rather than each
rolling its own ad-hoc sample.

Pipeline:

    corpus  = pd.read_parquet(tles).reset_index(drop=True)   # row index = run_id
    counts  = corpus.groupby(strata).size()
    quotas  = proportional_allocation(counts, target_size)
    chosen  = []
    for cell, q in quotas.items():
        chosen += rng.choice(corpus[cell].index, size=q, replace=False)
    chosen.sort()
    write_lines(out, chosen)

The `run_id`s are the row indices of `tles_cache.parquet` after
`reset_index(drop=True)` — exactly the same convention as
`sweep.run_sweep.main`, so a downstream caller can do
`pairs.loc[run_ids]` against the corpus to materialise the subset
without re-deriving any sampling logic.

Determinism. The default seed `20260513` and the default target size of
1,000 are baked into the issue scope (#30); they should not be changed
without coordinating with #28 and #31, which assume a fixed subset.

Outputs land at `outputs/sensitivity_subset_pair_ids.txt` — one
`run_id` per line, ASCII, sorted ascending. Plain text rather than
Parquet so the file diffs cleanly in PRs and reviewers can eyeball it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

DEFAULT_SEED: Final = 20260513
DEFAULT_TARGET_SIZE: Final = 1_000
STRATA_COLUMNS: Final = ("alt_shell", "generation", "target_dt_sec")


def _proportional_allocation(
    cell_counts: pd.Series,
    target_size: int,
) -> pd.Series:
    """Allocate `target_size` draws across cells proportional to cell counts.

    Floors per-cell quotas at the proportional share, then distributes
    the rounding remainder to the cells with the largest fractional
    parts (largest-remainder method). Cells with zero population get
    zero draws; cells smaller than their quota are clipped to their
    population, and the surplus is redistributed proportionally to the
    remaining cells. The clip-and-redistribute loop terminates because
    each iteration either fills the target or strictly shrinks the set
    of cells with headroom.
    """
    if (cell_counts < 0).any():
        raise ValueError("cell counts must be non-negative")
    total = int(cell_counts.sum())
    if total == 0:
        raise ValueError("corpus has no pairs to sample from")
    target_size = min(target_size, total)

    quotas = pd.Series(0, index=cell_counts.index, dtype=int)
    headroom = cell_counts.copy()
    remaining = target_size

    while remaining > 0:
        active = headroom > 0
        if not active.any():
            raise RuntimeError(
                "could not allocate the full target_size; this should be "
                "unreachable because target_size is clipped to total."
            )
        share = headroom[active].astype(float) / float(headroom[active].sum()) * remaining
        floor = np.floor(share).astype(int)
        # Largest-remainder distribution of the leftover units.
        leftover = remaining - int(floor.sum())
        remainder_order = (share - floor).sort_values(ascending=False).index
        bump = pd.Series(0, index=floor.index, dtype=int)
        for cell in remainder_order[:leftover]:
            bump.loc[cell] = 1
        proposed = floor + bump

        # Clip to per-cell headroom; redistribute any surplus in the
        # next loop iteration.
        capped = proposed.clip(upper=headroom[active])
        quotas.loc[capped.index] = quotas.loc[capped.index] + capped
        headroom.loc[capped.index] = headroom.loc[capped.index] - capped
        consumed = int(capped.sum())
        if consumed == 0:
            # All active cells were already at their cap; should not
            # happen because target_size ≤ total, but guard anyway.
            break
        remaining -= consumed

    if int(quotas.sum()) != target_size:
        raise RuntimeError(
            f"allocator produced {int(quotas.sum())} draws, expected {target_size}; "
            f"cell_counts={cell_counts.to_dict()}"
        )
    return quotas


def select_subset(
    corpus: pd.DataFrame,
    target_size: int = DEFAULT_TARGET_SIZE,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Return the `run_id`s of a stratified-random subset of `corpus`.

    `corpus` is expected to have a default integer index — the row
    index is treated as the `run_id` (matching the convention in
    `sweep.run_sweep`). Sampling is without replacement, deterministic
    under `seed`, and stratified by `STRATA_COLUMNS`. Returns a sorted
    1-D numpy array of length ≤ ``target_size``.
    """
    missing = [c for c in STRATA_COLUMNS if c not in corpus.columns]
    if missing:
        raise KeyError(f"corpus is missing strata columns: {missing}")
    if not isinstance(corpus.index, pd.RangeIndex) or corpus.index.start != 0:
        raise ValueError(
            "corpus must have a default 0-based RangeIndex so the row "
            "index can be interpreted as run_id; call "
            "`.reset_index(drop=True)` before passing it in."
        )

    cell_counts = corpus.groupby(list(STRATA_COLUMNS), sort=True).size()
    quotas = _proportional_allocation(cell_counts, target_size)

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for cell, q in quotas.items():
        if q == 0:
            continue
        cell_index = corpus.index[
            (corpus["alt_shell"] == cell[0])
            & (corpus["generation"] == cell[1])
            & (corpus["target_dt_sec"] == cell[2])
        ]
        chosen.extend(int(x) for x in rng.choice(cell_index, size=q, replace=False))
    return np.sort(np.asarray(chosen, dtype=np.int64))


def write_pair_ids(run_ids: np.ndarray, out_path: Path) -> None:
    """Write `run_ids` to `out_path`, one per line, ASCII, sorted ascending."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(int(x)) for x in run_ids) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tles",
        type=Path,
        required=True,
        help="Path to the cached TLE-pair corpus (src/static/tles_cache.parquet)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output text path (conventionally outputs/sensitivity_subset_pair_ids.txt)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f"Total number of pairs to draw (default {DEFAULT_TARGET_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"PRNG seed (default {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus = pd.read_parquet(args.tles).reset_index(drop=True)
    run_ids = select_subset(corpus, target_size=args.target_size, seed=args.seed)
    write_pair_ids(run_ids, args.out)
    print(
        f"wrote {args.out}: {len(run_ids)} run_ids (target {args.target_size}, seed {args.seed})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
