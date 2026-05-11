"""Run the full TLE-divergence-atlas sweep.

Drives gmat-sweep over the corpus of consecutive Starlink TLE pairs. Each run
propagates a satellite forward from t_i to t_j using both SGP4 (from TLE_i)
and GMAT high-fidelity force models, then compares both predictions against
the operator's next-TLE truth.

Outputs land as one Parquet per run in --output-dir. A reproducibility
manifest is written to --manifest.

Day-3 implementation. The skeleton here documents the intended CLI; the
guts come once the TLE-pair pipeline (Day 2) is in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        type=Path,
        required=True,
        help="Path to the GMAT mission .script",
    )
    parser.add_argument(
        "--tles",
        type=Path,
        required=True,
        help="Path to the cached TLE-pair Parquet (produced by tle_pipeline.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-run Parquet outputs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to write the reproducibility manifest (JSONL)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first N=8 pairs (for pipeline validation)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (joblib backend)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = args
    raise NotImplementedError(
        "Day-3 work item: implement sweep over TLE pairs. "
        "Pipeline: load TLE pairs from {tles}; for each pair build a RunSpec "
        "with initial state, propagation duration, and truth state; dispatch "
        "to gmat-sweep with LocalJoblibPool; aggregate manifest."
    )


if __name__ == "__main__":
    main()
