"""Per-satellite spacecraft properties from McDowell's GCAT satcat.

Drives per-run mass and cross-section overrides in the GMAT sweep so each
satellite is propagated with its own ballistic coefficient instead of a single
Starlink v2-mini placeholder applied uniformly across the corpus.

Pipeline:

    fetch_gcat_satcat(out)                # one-time HTTP, ~18 MB tsv
    satcat = parse_gcat_satcat(path)
    corpus = attach_spacecraft_props(corpus, satcat)
                                          # adds dry_mass_kg, span_m,
                                          #      drag_area_m2, srp_area_m2,
                                          #      gcat_pl_name

`dry_mass_kg` comes directly from GCAT and resolves at the launch-batch
level (~75 distinct values across the 76 launches in our corpus, ranging
220-700 kg). `drag_area_m2` / `srp_area_m2` are bucketed from McDowell's
structural `Span`: span <= 15 m yields 5.0 m^2 (Starlink v1.x; ~9 m span),
span > 15 m yields 3.5 m^2 (v2-mini / v2-full; ~29 m span). These are
attitude-averaged effective cross-sections; SpaceX does not publish them
per-sat. Cd and Cr are free-molecular-flow defaults that do not vary
meaningfully across the Starlink fleet.

Citation: McDowell, J. C. 2020, AJ, 159, 5 — General Catalog of Artificial
Space Objects (https://planet4589.org/space/gcat/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import pandas as pd

GCAT_SATCAT_URL: Final = "https://planet4589.org/space/gcat/tsv/cat/satcat.tsv"

SPAN_BUCKET_BOUNDARY_M: Final = 15.0
AREA_SMALL_M2: Final = 5.0
AREA_LARGE_M2: Final = 3.5
CD: Final = 2.2
CR: Final = 1.5


def drag_area_from_span(span_m: float) -> float:
    """Attitude-averaged effective area from GCAT's structural Span."""
    return AREA_SMALL_M2 if span_m <= SPAN_BUCKET_BOUNDARY_M else AREA_LARGE_M2


def fetch_gcat_satcat(out_path: Path, *, url: str = GCAT_SATCAT_URL, timeout: int = 60) -> int:
    """Download GCAT satcat.tsv to `out_path`. Returns byte count."""
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return len(resp.content)


def parse_gcat_satcat(satcat_path: Path) -> pd.DataFrame:
    """Parse a GCAT satcat.tsv into a DataFrame keyed by Satcat (NORAD ID).

    Columns kept: satcat_id (int), dry_mass_kg (float, NaN if blank),
    span_m (float, NaN if blank), pl_name (str).
    """
    with satcat_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().lstrip("#").rstrip("\n").split("\t")

    raw = pd.read_csv(satcat_path, sep="\t", comment="#", names=header, dtype=str)

    satcat_num = pd.to_numeric(raw["Satcat"], errors="coerce")
    keep = satcat_num.notna()
    out = pd.DataFrame(
        {
            "satcat_id": satcat_num[keep].astype(int).to_numpy(),
            "dry_mass_kg": pd.to_numeric(raw.loc[keep, "DryMass"].str.strip(), errors="coerce"),
            "span_m": pd.to_numeric(raw.loc[keep, "Span"].str.strip(), errors="coerce"),
            "pl_name": raw.loc[keep, "PLName"].fillna("").str.strip(),
        },
    )
    return out.reset_index(drop=True)


def attach_spacecraft_props(corpus: pd.DataFrame, satcat: pd.DataFrame) -> pd.DataFrame:
    """Attach per-NORAD-ID dry mass, span, and derived areas to a pair corpus.

    Raises `KeyError` if any sat in `corpus` is missing from `satcat`, or
    `ValueError` if a matched row has NaN mass/span. The static paper
    window expects 100% coverage; surfacing the gap is the right move.
    """
    if "norad_id" not in corpus.columns:
        raise ValueError("attach_spacecraft_props: corpus missing 'norad_id'")

    requested = {int(n) for n in corpus["norad_id"].unique()}
    available = set(satcat["satcat_id"].tolist())
    missing = requested - available
    if missing:
        sample = sorted(missing)[:10]
        raise KeyError(
            f"GCAT satcat missing {len(missing)} NORAD IDs from corpus: "
            f"{sample}{'...' if len(missing) > 10 else ''}",
        )

    props = (
        satcat[satcat["satcat_id"].isin(requested)]
        .drop_duplicates("satcat_id")
        .rename(columns={"satcat_id": "norad_id", "pl_name": "gcat_pl_name"})
        .reset_index(drop=True)
    )
    nan_rows = props[props[["dry_mass_kg", "span_m"]].isna().any(axis=1)]
    if not nan_rows.empty:
        raise ValueError(
            f"GCAT rows with missing mass/span for NORAD IDs: "
            f"{nan_rows['norad_id'].head(10).tolist()}",
        )

    props["drag_area_m2"] = props["span_m"].map(drag_area_from_span)
    props["srp_area_m2"] = props["drag_area_m2"]

    return corpus.merge(
        props[["norad_id", "dry_mass_kg", "span_m", "drag_area_m2", "srp_area_m2", "gcat_pl_name"]],
        on="norad_id",
        how="left",
        validate="many_to_one",
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch-satcat", help="Download McDowell's GCAT satcat.tsv")
    pf.add_argument("--out", type=Path, default=Path("src/data/gcat_satcat.tsv"))

    args = parser.parse_args()

    if args.cmd == "fetch-satcat":
        n = fetch_gcat_satcat(args.out)
        print(f"fetched {n} bytes to {args.out}", file=sys.stderr)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
