"""Space-weather artifacts for the sweep.

CelesTrak's ``sw19571001.txt`` (CssiSpaceWeather v1.2) feeds two
independent consumers in this repository:

1. An analysis-annotation parquet (``src/static/sw_cache.parquet``) keyed
   by UTC date, so per-run sweep parquets can carry real ``f107`` /
   ``ap`` values for the H3 regression instead of NaNs. Built by the
   ``fetch`` subcommand below.

2. The CSSI-format text file itself (``src/static/SpaceWeather-All-v1.2.txt``),
   read by GMAT's NRLMSISE-00 drag model via the ``FM.Drag.CSSISpaceWeatherFile``
   script-level override (see ``sweep/mission.script``). GMAT's bundled
   ``SpaceWeather-All-v1.2.txt`` is stamped 2025-03-21 — its observed-data
   horizon falls 13 months before the corpus window, so without this
   override the hi-fid arm would consume Schatten predictions for the
   entire April 2026 sweep. Built by the ``fetch-raw`` subcommand.

Pipeline (annotation parquet)::

    fetch_sw_file(out_text)                    # one-time HTTP, ~7 MB plain text
    df = parse_sw_text(text)                   # OBSERVED + DAILY_PREDICTED sections
    df_slice = df[in window margin]
    df_slice.to_parquet(out_parquet)

    # at sweep time:
    lookup = load_sw_cache(out_parquet)
    row = lookup[epoch_i_utc.date()]           # KeyError on miss (not silent NaN)

Pipeline (GMAT-facing CSSI file)::

    fetch_sw_file(out_path)                    # write the raw CelesTrak text verbatim
    verify_gmat_sw_coverage(out_path, window)  # observed horizon must reach window end

Column mapping for the parquet builder
(CelesTrak FORMAT(I4,I3,I3,I5,I3,8I3,I4,8I4,I4,F4.1,I2,I4,F6.1,I2,5F6.1)):

    field 0  = year
    field 1  = month
    field 2  = day
    field 22 = Ap Avg            -> ap_daily       (planetary daily Ap, dimensionless)
    field 30 = Obs F10.7         -> f107_obs       (daily observed 10.7 cm flux, sfu)
    field 31 = Obs Ctr81         -> f107_avg81     (81-day centered avg of obs F10.7, sfu)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

SW_URL: Final = "https://celestrak.org/SpaceData/sw19571001.txt"

# Default committed-cache window. The corpus window is April 2026
# (src/static/window.json); we keep ~9 months on each side so window
# extensions or follow-up analyses don't force a re-fetch.
DEFAULT_WINDOW_START: Final = dt.date(2026, 1, 1)
DEFAULT_WINDOW_END: Final = dt.date(2027, 1, 1)


@dataclass(frozen=True, slots=True)
class SwRow:
    """One day of space-weather annotation."""

    f107_obs: float
    f107_avg81: float
    ap_daily: float
    is_observed: bool


def fetch_sw_file(out_path: Path, *, url: str = SW_URL, timeout: int = 60) -> int:
    """Download the CelesTrak SW text file. Returns byte count."""
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return len(resp.content)


def parse_sw_text(text: str) -> pd.DataFrame:
    """Parse CelesTrak `sw19571001.txt` into a tidy DataFrame.

    Both OBSERVED and DAILY_PREDICTED sections are returned; rows carry
    an `is_observed` flag. Returns columns: `date`, `f107_obs`,
    `f107_avg81`, `ap_daily`, `is_observed`. Sorted by date.

    Whitespace-tokenised rather than fixed-width because CelesTrak
    occasionally pads columns differently between observed and predicted
    blocks. The token positions follow the published FORMAT string.
    """
    rows: list[tuple[dt.date, float, float, float, bool]] = []
    section: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("BEGIN OBSERVED"):
            section = "observed"
            continue
        if stripped.startswith("BEGIN DAILY_PREDICTED"):
            section = "daily_predicted"
            continue
        if stripped.startswith("END "):
            section = None
            continue
        if section is None or not stripped or stripped.startswith("#"):
            continue

        tokens = stripped.split()
        if len(tokens) < 32:
            continue

        date = dt.date(int(tokens[0]), int(tokens[1]), int(tokens[2]))
        ap_daily = float(tokens[22])
        f107_obs = float(tokens[30])
        f107_avg81 = float(tokens[31])
        rows.append((date, f107_obs, f107_avg81, ap_daily, section == "observed"))

    df = pd.DataFrame(
        rows,
        columns=["date", "f107_obs", "f107_avg81", "ap_daily", "is_observed"],
    )
    return df.sort_values("date").reset_index(drop=True)


def build_sw_cache(
    text: str,
    *,
    window_start: dt.date = DEFAULT_WINDOW_START,
    window_end: dt.date = DEFAULT_WINDOW_END,
) -> pd.DataFrame:
    """Parse SW text and slice to [window_start, window_end)."""
    df = parse_sw_text(text)
    mask = (df["date"] >= window_start) & (df["date"] < window_end)
    return df.loc[mask].reset_index(drop=True)


def load_sw_cache(path: Path) -> dict[dt.date, SwRow]:
    """Load `sw_cache.parquet` into a date-keyed dict for O(1) lookup."""
    df = pd.read_parquet(path)
    lookup: dict[dt.date, SwRow] = {}
    for record in df.to_dict(orient="records"):
        date = record["date"]
        if isinstance(date, pd.Timestamp):
            date = date.date()
        lookup[date] = SwRow(
            f107_obs=float(record["f107_obs"]),
            f107_avg81=float(record["f107_avg81"]),
            ap_daily=float(record["ap_daily"]),
            is_observed=bool(record["is_observed"]),
        )
    return lookup


def parse_observed_end_date(text: str) -> dt.date:
    """Return the UTC date of the last row inside the OBSERVED section.

    Parses the same CssiSpaceWeather v1.2 layout as :func:`parse_sw_text`
    but only scans the OBSERVED block — the GMAT-facing consumer cares
    about the observed horizon, not the daily/monthly predicted rows
    that follow.

    Raises ``ValueError`` if the file has no OBSERVED block or the block
    is empty.
    """
    section: str | None = None
    last_date: dt.date | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("BEGIN OBSERVED"):
            section = "observed"
            continue
        if stripped.startswith("END OBSERVED"):
            break
        if section != "observed" or not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 3:
            continue
        last_date = dt.date(int(tokens[0]), int(tokens[1]), int(tokens[2]))

    if last_date is None:
        raise ValueError("space-weather text has no OBSERVED rows")
    return last_date


def verify_gmat_sw_coverage(path: Path, window_end: dt.date) -> dt.date:
    """Confirm the GMAT-facing CSSI file's OBSERVED block covers `window_end`.

    Returns the file's last observed date. Raises ``ValueError`` if the
    observed horizon falls strictly before ``window_end`` — without this
    check, GMAT would silently fall back to the DAILY_PREDICTED or
    MONTHLY_PREDICTED sections for epochs past the horizon, which is the
    exact failure mode this whole override exists to prevent.
    """
    text = path.read_text(encoding="utf-8")
    end = parse_observed_end_date(text)
    if end < window_end:
        raise ValueError(
            f"{path}: observed-data horizon is {end.isoformat()} but the "
            f"corpus window ends {window_end.isoformat()}. Refresh the "
            f"snapshot with `make fetch-gmat-sw` (the CelesTrak source file "
            f"updates daily with new observations)."
        )
    return end


def lookup_for_epoch(lookup: dict[dt.date, SwRow], epoch_utc: pd.Timestamp) -> SwRow:
    """Look up SW values for the UTC date of `epoch_utc`.

    Raises KeyError with a descriptive message if the date isn't in the
    cache — surfaced rather than silently NaN-ed so a corpus window
    extension that outruns the cached SW window fails loudly.
    """
    date = epoch_utc.tz_convert("UTC").date()
    row = lookup.get(date)
    if row is None:
        raise KeyError(
            f"no space-weather entry for {date} in cache "
            f"(extend the cache window or re-run `make fetch-sw`)"
        )
    return row


# --- CLI -------------------------------------------------------------------


def _cli_fetch_raw(args: argparse.Namespace) -> int:
    print(f"fetching {SW_URL} -> {args.out}", file=sys.stderr)
    n_bytes = fetch_sw_file(args.out)
    print(f"  {n_bytes:,} bytes", file=sys.stderr)
    end = parse_observed_end_date(args.out.read_text(encoding="utf-8"))
    print(f"  last observed date: {end.isoformat()}", file=sys.stderr)
    return 0


def _cli_fetch(args: argparse.Namespace) -> int:
    text_path = args.text or args.out.with_suffix(".txt")
    print(f"fetching {SW_URL} -> {text_path}", file=sys.stderr)
    n_bytes = fetch_sw_file(text_path)
    print(f"  {n_bytes:,} bytes", file=sys.stderr)

    text = text_path.read_text(encoding="utf-8")
    df = build_sw_cache(text, window_start=args.window_start, window_end=args.window_end)
    print(
        f"  {len(df)} rows in [{args.window_start}, {args.window_end}) "
        f"({df['is_observed'].sum()} observed, {(~df['is_observed']).sum()} predicted)",
        file=sys.stderr,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"  wrote {args.out}", file=sys.stderr)

    if args.keep_text is False:
        text_path.unlink()
    return 0


def _parse_iso_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    fetch = sub.add_parser("fetch", help="Fetch CelesTrak SW and build sw_cache.parquet")
    fetch.add_argument("--out", type=Path, required=True, help="Output parquet path")
    fetch.add_argument(
        "--text",
        type=Path,
        default=None,
        help="Path to write the raw text (default: alongside --out as .txt; "
        "removed unless --keep-text)",
    )
    fetch.add_argument(
        "--window-start",
        type=_parse_iso_date,
        default=DEFAULT_WINDOW_START,
        help=f"Cache window start (default {DEFAULT_WINDOW_START.isoformat()})",
    )
    fetch.add_argument(
        "--window-end",
        type=_parse_iso_date,
        default=DEFAULT_WINDOW_END,
        help=f"Cache window end, exclusive (default {DEFAULT_WINDOW_END.isoformat()})",
    )
    fetch.add_argument(
        "--keep-text",
        action="store_true",
        help="Keep the downloaded raw text file (useful for debugging the parser)",
    )
    fetch.set_defaults(func=_cli_fetch)

    fetch_raw = sub.add_parser(
        "fetch-raw",
        help="Fetch the CelesTrak CssiSpaceWeather v1.2 text verbatim "
        "(for GMAT's FM.Drag.CSSISpaceWeatherFile override)",
    )
    fetch_raw.add_argument("--out", type=Path, required=True, help="Output text-file path")
    fetch_raw.set_defaults(func=_cli_fetch_raw)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
