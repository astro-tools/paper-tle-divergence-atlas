"""Quantify the ±2 h matching-tolerance selection effect for §3.3.

The corpus pipeline searches the same sat's per-window history for a
TLE whose epoch lies within ±2 h of `t_i + Δt`. A reviewer concern is
that this may bias the corpus *against* periods of poor coverage
(tracking gaps), since a sat-day with a multi-hour gap straddling
`t_i + Δt` would fail to produce a pair. This diagnostic produces two
related quantities:

1. **Per-Δt match success** -- the fraction of (starting-TLE × Δt)
   attempts that yield a matched pair. This number is naturally low
   even with dense tracking, because the ±2 h window is 4 h wide
   against an inter-TLE interval whose typical spacing is also
   $\\sim$~6 h: it reflects Poisson-window statistics, not tracking
   quality.
2. **Per-sat inter-TLE interval distribution** -- the actually
   load-bearing diagnostic for tracking-gap bias. If the right tail
   of the inter-TLE interval is bounded (no multi-hour gaps), then
   the ±2 h tolerance does not asymmetrically discard "quiet
   periods" -- every sat-day has roughly the same Poisson-window
   chance of matching, regardless of operator tracking quality.

Maneuver filtering is intentionally **not** applied -- we want the
pure matching-tolerance selection effect, separate from the maneuver
filter's effect on corpus size. The corpus sat population is taken
from `tles_cache.parquet` (the same 501 sats the main sweep uses).

Outputs `outputs/_selection_effect.json` (gitignored). Cheap (~seconds
on a laptop, no GMAT required).

Usage (from repo root):
    python src/scripts/_selection_effect.py \\
        --raw src/data/tles_raw.parquet \\
        --cache src/static/tles_cache.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sweep.tle_pipeline import (  # noqa: E402
    DEFAULT_TARGET_DTS_SEC,
    DEFAULT_TOLERANCE_SEC,
    build_pairs,
    subsample_starting_tles,
)


def _interval_stats(raw: pd.DataFrame) -> dict:
    """Inter-TLE interval distribution across per-sat consecutive pairs."""
    sorted_raw = raw.sort_values(["norad_id", "epoch"]).reset_index(drop=True)
    diffs = sorted_raw.groupby("norad_id")["epoch"].diff().dropna()
    diff_h = diffs.dt.total_seconds() / 3600.0
    longest_gap_per_sat = (
        sorted_raw.groupby("norad_id")["epoch"].apply(
            lambda s: s.diff().max().total_seconds() / 3600.0 if len(s) > 1 else float("nan"),
        )
    ).dropna()
    return {
        "n_intervals": int(len(diff_h)),
        "median_h": float(diff_h.median()),
        "p90_h": float(diff_h.quantile(0.90)),
        "p99_h": float(diff_h.quantile(0.99)),
        "max_h": float(diff_h.max()),
        "n_intervals_over_12h": int((diff_h > 12.0).sum()),
        "frac_intervals_over_12h": float((diff_h > 12.0).mean()),
        "median_longest_gap_per_sat_h": float(np.median(longest_gap_per_sat)),
        "p95_longest_gap_per_sat_h": float(np.quantile(longest_gap_per_sat, 0.95)),
        "max_longest_gap_per_sat_h": float(np.max(longest_gap_per_sat)),
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("src/data/tles_raw.parquet"))
    parser.add_argument("--cache", type=Path, default=Path("src/static/tles_cache.parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/_selection_effect.json"))
    args = parser.parse_args()

    raw = pd.read_parquet(args.raw)
    cache = pd.read_parquet(args.cache)
    corpus_ids = set(cache["norad_id"].unique())
    corpus_raw = raw[raw["norad_id"].isin(corpus_ids)].copy()
    print(
        f"corpus: {len(corpus_ids)} sat(s), {len(corpus_raw)} raw TLEs in window",
        file=sys.stderr,
    )

    starting = subsample_starting_tles(corpus_raw, per_day=1)
    n_starting = len(starting)
    print(f"one-TLE-per-sat-day sample: {n_starting} starting TLEs", file=sys.stderr)

    pairs = build_pairs(
        starting,
        corpus_raw,
        target_dts_sec=DEFAULT_TARGET_DTS_SEC,
        tolerance_sec=DEFAULT_TOLERANCE_SEC,
    )

    per_dt: dict[str, dict[str, float]] = {}
    overall_attempts = 0
    overall_unmatched = 0
    for target_dt in DEFAULT_TARGET_DTS_SEC:
        matched = int((pairs["target_dt_sec"] == target_dt).sum())
        unmatched = n_starting - matched
        per_dt[f"{target_dt}s"] = {
            "target_dt_h": target_dt / 3600.0,
            "n_attempts": n_starting,
            "n_matched": matched,
            "n_unmatched": unmatched,
            "unmatched_fraction": unmatched / n_starting if n_starting else 0.0,
        }
        overall_attempts += n_starting
        overall_unmatched += unmatched

    interval_stats = _interval_stats(corpus_raw)
    print(f"interval stats: {interval_stats}", file=sys.stderr)

    summary = {
        "corpus_sats": len(corpus_ids),
        "starting_tles_per_day": 1,
        "n_starting_tles": n_starting,
        "tolerance_sec": DEFAULT_TOLERANCE_SEC,
        "per_target_dt": per_dt,
        "overall": {
            "n_attempts": overall_attempts,
            "n_unmatched": overall_unmatched,
            "unmatched_fraction": overall_unmatched / overall_attempts if overall_attempts else 0.0,
        },
        "inter_tle_intervals_h": interval_stats,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
