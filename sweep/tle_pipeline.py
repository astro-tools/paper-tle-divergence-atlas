"""TLE corpus pipeline: fetch, pair across multiple Δt targets, filter, sample.

Day 2 of the paper plan. Pure data preparation: no GMAT, no SGP4 propagation.

Pipeline (`build_corpus`):

    raw_tles            = fetch_tles(window)
    sat_to_shell        = sample_sats(raw_tles)        # cap per altitude shell
    subset              = raw_tles[norad in sat_to_shell]
    maneuver_epochs     = detect_maneuver_epochs(subset)
    starting_tles       = subsample_starting_tles(subset)
    pairs               = build_pairs(starting_tles, subset, target_dts)
    no_maneuver_pairs   = filter_maneuvers(pairs, maneuver_epochs)
    corpus              = attach_shell(no_maneuver_pairs, sat_to_shell)

Sat sampling happens first so the pair-construction loop only sees the ~500
sats we keep. On the full 10k-sat Starlink fleet this is ~20× faster than
sampling at the end.

Methodology note. Starlink TLEs are updated ~6 times per day, so strict
consecutive pairs span only ~4 hours, well short of the staleness range
the paper studies. Instead, for each starting TLE we look for the nearest
available TLE at target Δt of 1, 3, and 7 days. Maneuver detection runs
across the *full* TLE history (not just pair endpoints) so a pair is
dropped if any maneuver landed inside its (t_i, t_j] interval.

Why a separate fetch step. Space-Track has an API allowance and requires
credentials; we fetch once locally and commit only the post-sample corpus
(`src/data/tles_cache.parquet`) so the rest of the work, and anyone
reproducing the paper, never needs Space-Track.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
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
# Δt buckets: 6 hours sits just beyond Starlink's ~4-hour operator update
# cadence, making it the most operationally relevant horizon. 1/3/7-day
# targets span the staleness range H1's power-law fit covers.
DEFAULT_TARGET_DTS_SEC: Final = (6 * 3600, 86_400, 3 * 86_400, 7 * 86_400)
DEFAULT_TOLERANCE_SEC: Final = 7_200  # ±2 hours around the target Δt
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
    import time

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

    # Chunk by single-day windows. Starlink TLEs are updated ~6 times per day
    # per sat, so a single-day query is ~36k rows — well within the per-response
    # cap that a multi-day query overflows ("Query range out of bounds").
    all_rows: list[dict] = []
    day = window.start
    while day < window.end:
        next_day = day + timedelta(days=1)
        query = (
            f"{base}/basicspacedata/query"
            f"/class/gp_history"
            f"/EPOCH/{day.strftime('%Y-%m-%d')}--{next_day.strftime('%Y-%m-%d')}"
            f"/OBJECT_NAME/~~STARLINK"
            f"/orderby/EPOCH%20asc"
            f"/format/json"
        )
        resp = session.get(query, timeout=300)
        resp.raise_for_status()
        rows = resp.json()
        # Space-Track surfaces query errors as a one-row list `[{"error": ...}]`.
        if isinstance(rows, list) and rows and "error" in rows[0] and "NORAD_CAT_ID" not in rows[0]:
            raise RuntimeError(f"Space-Track query error on {day:%Y-%m-%d}: {rows[0]['error']}")
        all_rows.extend(rows)
        print(
            f"  fetched {day:%Y-%m-%d}: {len(rows)} TLEs (total {len(all_rows)})", file=sys.stderr
        )
        day = next_day
        time.sleep(1.0)  # be polite to the rate limiter

    df = pd.DataFrame(
        [
            {
                "norad_id": int(r["NORAD_CAT_ID"]),
                "object_name": r["OBJECT_NAME"],
                "epoch": pd.Timestamp(r["EPOCH"], tz="UTC"),
                "line1": r["TLE_LINE1"],
                "line2": r["TLE_LINE2"],
            }
            for r in all_rows
        ],
    )
    df = df.sort_values(["norad_id", "epoch"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


def _mean_motion_from_line2(line2: str) -> float:
    """Parse mean motion (rev/day) from a TLE line 2 — columns 53..63."""
    return float(line2[52:63])


def detect_maneuver_epochs(
    tles: pd.DataFrame,
    *,
    sma_jump_threshold_km: float = DEFAULT_MANEUVER_THRESHOLD_KM,
) -> pd.DataFrame:
    """For each consecutive TLE pair per sat, flag epochs where SMA jumped.

    Returns a long DataFrame with columns (norad_id, maneuver_epoch) — one row
    per detected event. Pairs in `build_pairs` whose (t_i, t_j] interval covers
    any of these epochs are dropped by `filter_maneuvers`.
    """
    required = {"norad_id", "epoch", "line2"}
    missing = required - set(tles.columns)
    if missing:
        raise ValueError(f"detect_maneuver_epochs: tles missing columns {missing}")

    tles = tles.sort_values(["norad_id", "epoch"]).reset_index(drop=True)
    smas = tles["line2"].map(lambda x: sma_km_from_mean_motion(_mean_motion_from_line2(x)))
    sma_diff = smas.groupby(tles["norad_id"]).diff().abs()
    maneuver_mask = sma_diff > sma_jump_threshold_km
    return (
        tles.loc[maneuver_mask, ["norad_id", "epoch"]]
        .rename(
            columns={"epoch": "maneuver_epoch"},
        )
        .reset_index(drop=True)
    )


def subsample_starting_tles(tles: pd.DataFrame, *, per_day: int = 1) -> pd.DataFrame:
    """Subsample to `per_day` starting TLEs per sat per UTC day (first-N each day).

    Starlink updates ~6×/day; sampling at the day level keeps the corpus
    tractable while spanning the full window of starting epochs.
    """
    if per_day < 1:
        raise ValueError("subsample_starting_tles: per_day must be >= 1")
    tles = tles.sort_values(["norad_id", "epoch"]).reset_index(drop=True)
    tles["_day"] = tles["epoch"].dt.floor("D")
    out = tles.groupby(["norad_id", "_day"], sort=False).head(per_day)
    return out.drop(columns="_day").reset_index(drop=True)


def build_pairs(
    starting_tles: pd.DataFrame,
    all_tles: pd.DataFrame,
    *,
    target_dts_sec: tuple[int, ...] = DEFAULT_TARGET_DTS_SEC,
    tolerance_sec: int = DEFAULT_TOLERANCE_SEC,
) -> pd.DataFrame:
    """Build (TLE_i, TLE_j) pairs at multiple target Δt values.

    For each row in `starting_tles` and each target Δt, search `all_tles` for
    the same sat's TLE with epoch closest to t_i + Δt, within ±`tolerance_sec`.
    Emits one row per (starting TLE, target Δt) when a match is found.
    """
    required = {"norad_id", "epoch", "line1", "line2"}
    for label, df in (("starting_tles", starting_tles), ("all_tles", all_tles)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"build_pairs: {label} missing columns {missing}")

    by_sat = {
        nid: group.sort_values("epoch").reset_index(drop=True)
        for nid, group in all_tles.groupby("norad_id", sort=False)
    }
    tol = pd.Timedelta(seconds=tolerance_sec)

    rows = []
    for _, start in starting_tles.iterrows():
        sat = by_sat.get(start["norad_id"])
        if sat is None:
            continue
        epoch_i = start["epoch"]
        sma_i = sma_km_from_mean_motion(_mean_motion_from_line2(start["line2"]))

        for target_dt in target_dts_sec:
            target_epoch = epoch_i + pd.Timedelta(seconds=target_dt)
            window_mask = (sat["epoch"] >= target_epoch - tol) & (
                sat["epoch"] <= target_epoch + tol
            )
            candidates = sat.loc[window_mask]
            if candidates.empty:
                continue
            # Pick the TLE closest to the target epoch.
            j_idx = (candidates["epoch"] - target_epoch).abs().idxmin()
            j = candidates.loc[j_idx]
            sma_j = sma_km_from_mean_motion(_mean_motion_from_line2(j["line2"]))
            rows.append(
                {
                    "norad_id": int(start["norad_id"]),
                    "target_dt_sec": target_dt,
                    "epoch_i": epoch_i,
                    "epoch_j": j["epoch"],
                    "actual_dt_sec": (j["epoch"] - epoch_i).total_seconds(),
                    "line1_i": start["line1"],
                    "line2_i": start["line2"],
                    "line1_j": j["line1"],
                    "line2_j": j["line2"],
                    "sma_i_km": sma_i,
                    "sma_j_km": sma_j,
                },
            )
    return pd.DataFrame(rows)


def filter_maneuvers(
    pairs: pd.DataFrame,
    maneuver_epochs: pd.DataFrame,
) -> pd.DataFrame:
    """Drop pairs whose (t_i, t_j] interval covers any detected maneuver epoch.

    Maneuver detection (`detect_maneuver_epochs`) walks the *full* TLE history;
    this filter applies the result at the pair level. A pair survives only if
    no maneuver landed inside it.
    """
    if pairs.empty:
        return pairs

    # Group maneuver epochs per sat as pandas Index objects so comparisons
    # stay tz-aware (np.datetime64 strips the UTC offset and breaks against
    # tz-aware pd.Timestamps).
    by_sat: dict[int, pd.DatetimeIndex] = {
        nid: pd.DatetimeIndex(group["maneuver_epoch"])
        for nid, group in maneuver_epochs.groupby("norad_id", sort=False)
    }
    keep = np.ones(len(pairs), dtype=bool)
    for i, p in enumerate(pairs.itertuples(index=False)):
        events = by_sat.get(p.norad_id)
        if events is None or len(events) == 0:
            continue
        # An event lands inside (t_i, t_j] if t_i < event <= t_j.
        if ((events > p.epoch_i) & (events <= p.epoch_j)).any():
            keep[i] = False
    return pairs.loc[keep].reset_index(drop=True)


def sample_sats(
    tles: pd.DataFrame,
    *,
    n_per_shell: int = DEFAULT_SATS_PER_SHELL,
    seed: int = DEFAULT_SEED,
) -> dict[int, str]:
    """Pick a stratified sample of satellite NORAD IDs by altitude shell.

    Returns a dict mapping norad_id → shell label. Done up-front so the
    expensive pair-construction loop never sees sats we'll discard later.
    """
    required = {"norad_id", "line2"}
    missing = required - set(tles.columns)
    if missing:
        raise ValueError(f"sample_sats: tles missing columns {missing}")

    rng = np.random.default_rng(seed)
    # Median SMA per sat across its entire history → robust against a single
    # outlier TLE near a maneuver.
    sat_smas = (
        tles.assign(
            sma_km=tles["line2"].map(lambda x: sma_km_from_mean_motion(_mean_motion_from_line2(x))),
        )
        .groupby("norad_id")["sma_km"]
        .median()
    )
    sat_shells = sat_smas.apply(altitude_shell).dropna()

    keep: dict[int, str] = {}
    for shell in ALTITUDE_SHELLS_KM:
        candidates = sat_shells[sat_shells == shell].index.tolist()
        if not candidates:
            continue
        size = min(n_per_shell, len(candidates))
        chosen = rng.choice(candidates, size=size, replace=False)
        for sat in chosen:
            keep[int(sat)] = shell
    return keep


def attach_shell(pairs: pd.DataFrame, sat_to_shell: dict[int, str]) -> pd.DataFrame:
    """Attach the alt_shell column to a corpus DataFrame."""
    if pairs.empty:
        return pairs
    out = pairs.copy()
    out["alt_shell"] = out["norad_id"].map(sat_to_shell)
    return out.reset_index(drop=True)


def build_corpus(
    tles: pd.DataFrame,
    *,
    sma_jump_threshold_km: float = DEFAULT_MANEUVER_THRESHOLD_KM,
    target_dts_sec: tuple[int, ...] = DEFAULT_TARGET_DTS_SEC,
    tolerance_sec: int = DEFAULT_TOLERANCE_SEC,
    n_per_shell: int = DEFAULT_SATS_PER_SHELL,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """End-to-end: sat-sample → maneuver-detect → starting-subsample → pair → filter.

    Sat sampling runs first so the pair-construction loop only sees the
    ~500 sats we'll keep. On the full 10k-sat Starlink fleet this is ~20×
    faster than sampling after pair construction.
    """
    sat_to_shell = sample_sats(tles, n_per_shell=n_per_shell, seed=seed)
    subset = tles[tles["norad_id"].isin(sat_to_shell)]

    maneuvers = detect_maneuver_epochs(subset, sma_jump_threshold_km=sma_jump_threshold_km)
    starts = subsample_starting_tles(subset)
    pairs = build_pairs(
        starts,
        subset,
        target_dts_sec=target_dts_sec,
        tolerance_sec=tolerance_sec,
    )
    clean = filter_maneuvers(pairs, maneuvers)
    return attach_shell(clean, sat_to_shell)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="One-time fetch from Space-Track")
    pf.add_argument("--window", type=Path, default=Path("src/data/window.json"))
    pf.add_argument("--out", type=Path, default=Path("src/data/tles_raw.parquet"))

    pb = sub.add_parser("build", help="maneuver detect + pair + filter + sample → corpus")
    pb.add_argument("--raw", type=Path, default=Path("src/data/tles_raw.parquet"))
    pb.add_argument("--out", type=Path, default=Path("src/data/tles_cache.parquet"))
    pb.add_argument("--sma-threshold-km", type=float, default=DEFAULT_MANEUVER_THRESHOLD_KM)
    pb.add_argument("--per-shell", type=int, default=DEFAULT_SATS_PER_SHELL)
    pb.add_argument("--seed", type=int, default=DEFAULT_SEED)
    pb.add_argument(
        "--satcat",
        type=Path,
        default=None,
        help="If given, parse GCAT satcat.tsv and attach per-sat dry_mass_kg / "
        "span_m / drag_area_m2 / srp_area_m2 / gcat_pl_name columns.",
    )

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
        if args.satcat is not None:
            from sweep.spacecraft_props import attach_spacecraft_props, parse_gcat_satcat

            satcat = parse_gcat_satcat(args.satcat)
            corpus = attach_spacecraft_props(corpus, satcat)
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
