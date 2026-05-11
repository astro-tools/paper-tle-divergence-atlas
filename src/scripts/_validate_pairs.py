"""Single-satellite validation of the TLE-pair pipeline.

Picks one densely-tracked Starlink sat, propagates each TLE_i forward to t_j
via SGP4, and compares the propagated position to SGP4(TLE_j, Δt=0) -- the
"next-TLE truth" used throughout the paper. The script emits a plot of |Δr|
vs. Δt and prints summary statistics.

This is methodological sanity, not a manuscript figure (underscored name so
showyourwork ignores it). Acceptance for issue #1: sub-kilometer Δr at
Δt < 1 day on the chosen sat.

Usage:
    python src/scripts/_validate_pairs.py \\
        --raw src/data/tles_raw.parquet \\
        --norad-id 44713 \\
        --out src/data/_validate_pairs.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday


def _propagate(line1: str, line2: str, jd: float, fr: float) -> np.ndarray:
    """SGP4 position at the given Julian day, in km, TEME frame."""
    sat = Satrec.twoline2rv(line1, line2)
    err, r, _ = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 returned error {err} for line1={line1!r}")
    return np.asarray(r)


def _to_jd_fr(epoch: pd.Timestamp) -> tuple[float, float]:
    epoch_utc = epoch.tz_convert("UTC")
    return jday(
        epoch_utc.year,
        epoch_utc.month,
        epoch_utc.day,
        epoch_utc.hour,
        epoch_utc.minute,
        epoch_utc.second + epoch_utc.microsecond / 1e6,
    )


def validate_satellite(tles: pd.DataFrame, norad_id: int) -> pd.DataFrame:
    """For each consecutive pair on a single sat, return |Δr| at t_j."""
    sat_tles = tles[tles["norad_id"] == norad_id].sort_values("epoch").reset_index(drop=True)
    if len(sat_tles) < 2:
        raise ValueError(f"sat {norad_id}: need >=2 TLEs in the window, got {len(sat_tles)}")

    rows = []
    for i in range(len(sat_tles) - 1):
        a, b = sat_tles.iloc[i], sat_tles.iloc[i + 1]
        jd_j, fr_j = _to_jd_fr(b["epoch"])
        r_propagated = _propagate(a["line1"], a["line2"], jd_j, fr_j)
        r_truth = _propagate(b["line1"], b["line2"], jd_j, fr_j)
        rows.append(
            {
                "dt_sec": (b["epoch"] - a["epoch"]).total_seconds(),
                "dt_hours": (b["epoch"] - a["epoch"]).total_seconds() / 3600.0,
                "dr_km": float(np.linalg.norm(r_propagated - r_truth)),
            },
        )
    return pd.DataFrame(rows)


def _pick_dense_sat(tles: pd.DataFrame) -> int:
    counts = tles["norad_id"].value_counts()
    return int(counts.index[0])


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("src/data/tles_raw.parquet"))
    parser.add_argument(
        "--norad-id",
        type=int,
        default=None,
        help="NORAD ID to validate. Default: the densely-tracked sat in the cache.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/data/_validate_pairs.png"),
        help="Where to write the diagnostic PNG.",
    )
    args = parser.parse_args()

    tles = pd.read_parquet(args.raw)
    norad_id = args.norad_id or _pick_dense_sat(tles)
    df = validate_satellite(tles, norad_id)

    print(f"sat {norad_id}: {len(df)} pairs", file=sys.stderr)
    print(df.describe()[["dt_hours", "dr_km"]], file=sys.stderr)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["dt_hours"], df["dr_km"], s=20, alpha=0.7)
    ax.set_xlabel("Δt (hours)")
    ax.set_ylabel("|Δr| at t_j (km)")
    ax.set_yscale("log")
    ax.set_title(f"SGP4 prediction vs. next-TLE truth — NORAD {norad_id}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
