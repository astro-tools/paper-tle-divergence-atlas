"""TLE corpus pipeline: fetch, pair, filter, sample.

Day 2 of the paper plan. Pure data preparation: no GMAT, no SGP4 propagation.

The four public functions are sequenced like a pipeline:

    raw_tles  = fetch_tles(window)         # one-time; needs Space-Track creds
    pairs     = build_pairs(raw_tles)      # consecutive (TLE_i, TLE_j) per sat
    filtered  = filter_maneuvers(pairs)    # drop pairs that bridge maneuvers
    corpus    = stratified_sample(filtered) # 500 sats across altitude shells

`build_corpus` runs the last three together against an already-fetched cache.
Run `python -m sweep.tle_pipeline --help` for the CLI.

Why a separate fetch step. Space-Track has an API allowance and requires
credentials; we fetch once locally and commit the resulting Parquet so the
rest of the work, and anyone reproducing the paper, never needs Space-Track.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

MU_EARTH_KM3_S2: Final = 398600.4418
EARTH_RADIUS_KM: Final = 6378.137

# Starlink shell stratification (km altitude). Bin widths chosen to be
# disjoint, cover the bulk of the active fleet, and have enough margin to
# absorb short-term Keplerian wobble without misclassifying a satellite.
ALTITUDE_SHELLS_KM: Final = {
    "540": (533.0, 547.0),
    "550": (547.0, 557.0),
    "560": (557.0, 573.0),
}

DEFAULT_MANEUVER_THRESHOLD_KM: Final = 0.1  # 100 m SMA jump
DEFAULT_SATS_PER_SHELL: Final = 167
DEFAULT_SEED: Final = 20260401


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    @classmethod
    def from_json(cls, path: Path) -> Window:
        data = json.loads(path.read_text())
        return cls(
            start=datetime.fromisoformat(data["start"].replace("Z", "+00:00")),
            end=datetime.fromisoformat(data["end"].replace("Z", "+00:00")),
        )


def sma_km_from_mean_motion(mean_motion_rev_per_day: float) -> float:
    """Semi-major axis in km from TLE mean motion (rev/day)."""
    n_rad_per_s = mean_motion_rev_per_day * 2.0 * math.pi / 86400.0
    return (MU_EARTH_KM3_S2 / n_rad_per_s**2) ** (1.0 / 3.0)


def altitude_shell(sma_km: float) -> str | None:
    altitude_km = sma_km - EARTH_RADIUS_KM
    for shell, (low, high) in ALTITUDE_SHELLS_KM.items():
        if low <= altitude_km < high:
            return shell
    return None


def fetch_tles(
    window: Window,
    out_path: Path,
    *,
    username: str | None = None,
    password: str | None = None,
) -> int:
    """Fetch all Starlink TLEs in `window` from Space-Track. Returns row count.

    Credentials read from `username`/`password` args if given, else from
    SPACETRACK_USERNAME / SPACETRACK_PASSWORD env vars. The result is written
    as a Parquet with columns: norad_id, object_name, epoch (UTC), line1, line2.
    """
    import requests

    username = (username or os.environ.get("SPACETRACK_USERNAME", "")).strip()
    password = (password or os.environ.get("SPACETRACK_PASSWORD", "")).strip()
    if not username or not password:
        raise RuntimeError(
            "Space-Track credentials missing. Set SPACETRACK_USERNAME and "
            "SPACETRACK_PASSWORD env vars or pass --username/--password.",
        )

    base = "https://www.space-track.org"
    session = requests.Session()
    auth = session.post(
        f"{base}/ajaxauth/login",
        data={"identity": username, "password": password},
        timeout=30,
    )
    auth.raise_for_status()
    # Space-Track returns HTTP 200 with `{"Login":"Failed"}` on bad creds —
    # the status code alone is not enough. The auth cookie ("chocolatechip")
    # is present either way, so check the body explicitly.
    if "Failed" in auth.text:
        raise RuntimeError(
            "Space-Track auth failed (status 200 but body indicates failure). "
            "Check SPACETRACK_USERNAME / SPACETRACK_PASSWORD; common cause is "
            "trailing whitespace or Windows line endings in the .env file.",
        )

    epoch_range = f"{window.start.strftime('%Y-%m-%d')}--{window.end.strftime('%Y-%m-%d')}"
    query = (
        f"{base}/basicspacedata/query"
        f"/class/gp_history"
        f"/EPOCH/{epoch_range}"
        f"/OBJECT_NAME/~~STARLINK"
        f"/orderby/NORAD_CAT_ID,EPOCH%20asc"
        f"/format/json"
    )
    resp = session.get(query, timeout=300)
    resp.raise_for_status()

    rows = resp.json()
    df = pd.DataFrame(
        [
            {
                "norad_id": int(r["NORAD_CAT_ID"]),
                "object_name": r["OBJECT_NAME"],
                "epoch": pd.Timestamp(r["EPOCH"], tz="UTC"),
                "line1": r["TLE_LINE1"],
                "line2": r["TLE_LINE2"],
            }
            for r in rows
        ],
    )
    df = df.sort_values(["norad_id", "epoch"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


def _mean_motion_from_line2(line2: str) -> float:
    """Parse mean motion (rev/day) from a TLE line 2 — columns 53..63."""
    return float(line2[52:63])


def build_pairs(tles: pd.DataFrame) -> pd.DataFrame:
    """Build consecutive (TLE_i, TLE_j) pairs per satellite.

    Input must have columns `norad_id`, `epoch`, `line1`, `line2`. Output has
    one row per pair with both TLE lines, both epochs, the time delta, and
    SMAs at each endpoint plus the absolute jump in km.
    """
    required = {"norad_id", "epoch", "line1", "line2"}
    missing = required - set(tles.columns)
    if missing:
        raise ValueError(f"build_pairs: tles missing columns {missing}")

    tles = tles.sort_values(["norad_id", "epoch"]).reset_index(drop=True)

    rows = []
    for norad_id, group in tles.groupby("norad_id", sort=False):
        records = group.to_dict(orient="records")
        for i in range(len(records) - 1):
            a, b = records[i], records[i + 1]
            sma_i = sma_km_from_mean_motion(_mean_motion_from_line2(a["line2"]))
            sma_j = sma_km_from_mean_motion(_mean_motion_from_line2(b["line2"]))
            rows.append(
                {
                    "norad_id": norad_id,
                    "epoch_i": a["epoch"],
                    "epoch_j": b["epoch"],
                    "dt_sec": (b["epoch"] - a["epoch"]).total_seconds(),
                    "line1_i": a["line1"],
                    "line2_i": a["line2"],
                    "line1_j": b["line1"],
                    "line2_j": b["line2"],
                    "sma_i_km": sma_i,
                    "sma_j_km": sma_j,
                    "sma_jump_km": abs(sma_j - sma_i),
                },
            )
    return pd.DataFrame(rows)


def filter_maneuvers(
    pairs: pd.DataFrame,
    *,
    sma_jump_threshold_km: float = DEFAULT_MANEUVER_THRESHOLD_KM,
) -> pd.DataFrame:
    """Drop pairs whose SMA jump exceeds the threshold (proxy for maneuvers).

    Starlink ion-drive maneuvers leave a clear discontinuity in mean motion
    between consecutive TLEs; 100 m is the published rule of thumb in the
    operations literature for unambiguous discrimination. The diagnostic plot
    in `src/scripts/fig_maneuver_filter.py` is the calibration check.
    """
    return pairs[pairs["sma_jump_km"] <= sma_jump_threshold_km].reset_index(drop=True)


def stratified_sample(
    pairs: pd.DataFrame,
    *,
    n_per_shell: int = DEFAULT_SATS_PER_SHELL,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Stratified random sample by altitude shell, capped at `n_per_shell`/shell.

    Shell assignment uses the median SMA across a satellite's pairs so a
    sat that ended the window in a different shell from where it started
    is binned by its dominant residence.
    """
    if "sma_i_km" not in pairs.columns:
        raise ValueError("stratified_sample: pairs missing 'sma_i_km' column")

    rng = np.random.default_rng(seed)

    sat_smas = pairs.groupby("norad_id")["sma_i_km"].median()
    sat_shells = sat_smas.apply(altitude_shell).dropna()

    keep_ids: list[int] = []
    for shell in ALTITUDE_SHELLS_KM:
        candidates = sat_shells[sat_shells == shell].index.tolist()
        if not candidates:
            continue
        size = min(n_per_shell, len(candidates))
        chosen = rng.choice(candidates, size=size, replace=False)
        keep_ids.extend(int(x) for x in chosen)

    out = pairs[pairs["norad_id"].isin(keep_ids)].copy()
    out["alt_shell"] = out["norad_id"].map(sat_shells.to_dict())
    return out.reset_index(drop=True)


def build_corpus(
    tles: pd.DataFrame,
    *,
    sma_jump_threshold_km: float = DEFAULT_MANEUVER_THRESHOLD_KM,
    n_per_shell: int = DEFAULT_SATS_PER_SHELL,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """End-to-end: pairs → filter → sample."""
    return stratified_sample(
        filter_maneuvers(
            build_pairs(tles),
            sma_jump_threshold_km=sma_jump_threshold_km,
        ),
        n_per_shell=n_per_shell,
        seed=seed,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="One-time fetch from Space-Track")
    pf.add_argument("--window", type=Path, default=Path("src/data/window.json"))
    pf.add_argument("--out", type=Path, default=Path("src/data/tles_raw.parquet"))

    pb = sub.add_parser("build", help="Pairs + filter + sample → corpus parquet")
    pb.add_argument("--raw", type=Path, default=Path("src/data/tles_raw.parquet"))
    pb.add_argument("--out", type=Path, default=Path("src/data/tles_cache.parquet"))
    pb.add_argument("--sma-threshold-km", type=float, default=DEFAULT_MANEUVER_THRESHOLD_KM)
    pb.add_argument("--per-shell", type=int, default=DEFAULT_SATS_PER_SHELL)
    pb.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args()

    if args.cmd == "fetch":
        window = Window.from_json(args.window)
        n = fetch_tles(window, args.out)
        print(f"fetched {n} TLEs to {args.out}", file=sys.stderr)
        return 0

    if args.cmd == "build":
        raw = pd.read_parquet(args.raw)
        corpus = build_corpus(
            raw,
            sma_jump_threshold_km=args.sma_threshold_km,
            n_per_shell=args.per_shell,
            seed=args.seed,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        corpus.to_parquet(args.out, index=False)
        print(
            f"built corpus: {len(corpus)} pairs across "
            f"{corpus['norad_id'].nunique()} sats → {args.out}",
            file=sys.stderr,
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
