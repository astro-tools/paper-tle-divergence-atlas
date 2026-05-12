"""Per-satellite spacecraft properties for the GMAT sweep.

Drives per-run mass and cross-section overrides so each Starlink is
propagated with a generation-appropriate ballistic coefficient instead of
a single v2-mini placeholder applied uniformly across the corpus.

Pipeline:

    fetch_gcat_satcat(out)                # one-time HTTP, ~18 MB tsv
    satcat = parse_gcat_satcat(path)
    corpus = attach_spacecraft_props(corpus, satcat)
                                          # adds dry_mass_kg, span_m,
                                          #      generation, drag_area_m2,
                                          #      srp_area_m2, gcat_pl_name

Property sources:

- **`dry_mass_kg`** comes directly from McDowell's GCAT and resolves at the
  launch-batch level (~75 distinct values across the 76 launches in the
  corpus, ranging 220-700 kg). McDowell flags all entries as estimates.

- **`generation`** is classified from McDowell's `PLName` field (e.g.
  `"Starlink V1.0-L20-03"` -> `v1.0`; `"Starlink Group 4-1-3"` -> `v1.5`;
  `"Starlink Group 15-1-22"` -> `v2-mini`). This is more accurate per-sat
  than year-bucketing from the COSPAR launch ID, and captures sub-variants
  (TSP / Direct-to-Cell) that share a launch-year cohort with their base
  generation but differ structurally.

- **`drag_area_m2`** / **`srp_area_m2`** are looked up per-generation from
  `DRAG_AREA_M2` / `SRP_AREA_M2`. The drag values are shark-fin nominal
  station-keeping attitude-averaged effective areas.

  For **v1.0 / v1.5**, the value is anchored to Baruah et al. 2024
  (https://doi.org/10.1029/2023SW003716), which models the loss of 38
  Starlink v1.5 satellites during the February 2022 geomagnetic storm
  using a shark-fin ram area of **4.48 m^2 with Cd = 1** (their
  "operational orientation"). Since we use Cd = 2.2, we set DragArea =
  4.48 / 2.2 ≈ 2.0 m^2 so the resulting (Cd * A) matches Baruah's
  effective drag cross-section. Baruah's reported open-book (collision-
  avoidance) value of 1.00 m^2 maps to ~0.45 m^2 at Cd = 2.2; we use the
  shark-fin value since station-keeping is the dominant duty cycle.

  For **v2-mini**, no published ballistic-coefficient analysis with the
  same rigor exists yet. We scale Baruah's v1.5 shark-fin ram area by
  the v2-mini bus-size ratio reported on Gunter's Space Page (v2-mini
  bus is "twice the size" of v1.5; we apply a 1.5x linear / ~2.25x area
  factor for the deployed-panel cross-section), yielding 4.5 m^2 at
  Cd = 2.2. This is the largest single modelling assumption in the
  spacecraft-properties layer.

  For **v2-full** (Starship-launched, no in-corpus sats), we use 10.0
  m^2 as an extrapolation; not used in this paper's window.

- **`Cd`** (2.2) and **`Cr`** (1.5) are free-molecular-flow defaults that
  don't vary meaningfully across the Starlink fleet.

Citations:

- **GCAT (dry mass + PLName classification):** McDowell, J. C. 2020,
  *General Catalog of Artificial Space Objects*, AJ, 159, 5.
  https://planet4589.org/space/gcat/
- **v1.5 drag cross-section:** Baruah, Y., Roy, S., Sinha, S., Palmerio,
  E., Pal, S., Oliveira, D. M., & Nandy, D. 2024, *The Loss of Starlink
  Satellites in February 2022*, Space Weather, 22, e2023SW003716.
  https://doi.org/10.1029/2023SW003716
- **v2-mini bus scaling:** Gunter Krebs, *Starlink Block v2-Mini*,
  Gunter's Space Page. https://space.skyrocket.de/doc_sdat/starlink-v2-mini.htm
"""

from __future__ import annotations

import argparse
import re
import sys
from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd

GCAT_SATCAT_URL: Final = "https://planet4589.org/space/gcat/tsv/cat/satcat.tsv"

CD: Final = 2.2
CR: Final = 1.5


class Generation(StrEnum):
    """Starlink hardware cohort, classified from McDowell's PLName."""

    V1_0 = "v1.0"
    V1_5 = "v1.5"
    V2_MINI = "v2-mini"
    V2_FULL = "v2-full"


# Per-generation attitude-averaged effective drag area (m^2). Shark-fin
# nominal station-keeping orientation. v1.0/v1.5 anchored to Baruah et al.
# 2024's 4.48 m^2 (Cd=1) -> 2.0 m^2 at Cd=2.2. v2-mini scaled by bus-size
# ratio (Gunter's). See module docstring for the citation chain.
DRAG_AREA_M2: Final[dict[Generation, float]] = {
    Generation.V1_0: 2.0,
    Generation.V1_5: 2.0,
    Generation.V2_MINI: 4.5,
    Generation.V2_FULL: 10.0,
}

# Per-generation effective SRP area (m^2). Larger than drag area because
# SRP integrates over the deployed-panel surface regardless of ram direction.
# Approximated as ~2.5x drag area, matching the panel-face-on / panel-edge-on
# geometry ratio for Starlink's single-panel design. v2-mini and v2-full
# scaled by the same bus-size argument.
SRP_AREA_M2: Final[dict[Generation, float]] = {
    Generation.V1_0: 5.0,
    Generation.V1_5: 5.0,
    Generation.V2_MINI: 10.0,
    Generation.V2_FULL: 25.0,
}

_GROUP_RE: Final = re.compile(r"^Starlink\s+Group\s+(\d+)-")


def generation_from_pl_name(pl_name: str) -> Generation:
    """Map McDowell GCAT `PLName` to a Starlink generation.

    Patterns:
      - "Starlink V0.9-..."   -> V1_0 (pre-production lumped with v1.0)
      - "Starlink V1.0-..."   -> V1_0
      - "Starlink TSP..."     -> V1_0 (Tranche Service Provider; DOD-modified v1.0 bus)
      - "Starlink Group N-..." with N in 1..5  -> V1_5
      - "Starlink Group N-..." with N >= 6     -> V2_MINI
    """
    s = pl_name.strip()
    if not s.startswith("Starlink "):
        raise ValueError(f"unrecognized PLName for Starlink classification: {pl_name!r}")
    rest = s[len("Starlink ") :].lstrip()
    if rest.startswith(("V0.9", "V1.0", "TSP")):
        return Generation.V1_0
    m = _GROUP_RE.match(s)
    if m is not None:
        return Generation.V1_5 if int(m.group(1)) <= 5 else Generation.V2_MINI
    raise ValueError(f"unrecognized PLName for Starlink classification: {pl_name!r}")


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
    """Attach per-NORAD-ID dry mass + per-generation drag/SRP area to a corpus.

    Raises `KeyError` if any sat in `corpus` is missing from `satcat`, or
    `ValueError` if a matched row has NaN dry mass, or if a PLName fails
    generation classification. The static paper window expects 100%
    coverage; surfacing the gap is the right move.
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
    nan_rows = props[props["dry_mass_kg"].isna()]
    if not nan_rows.empty:
        raise ValueError(
            f"GCAT rows with missing dry mass for NORAD IDs: "
            f"{nan_rows['norad_id'].head(10).tolist()}",
        )

    props["generation"] = props["gcat_pl_name"].map(generation_from_pl_name).astype(str)
    props["drag_area_m2"] = props["generation"].map(lambda g: DRAG_AREA_M2[Generation(g)])
    props["srp_area_m2"] = props["generation"].map(lambda g: SRP_AREA_M2[Generation(g)])

    return corpus.merge(
        props[
            [
                "norad_id",
                "dry_mass_kg",
                "span_m",
                "generation",
                "drag_area_m2",
                "srp_area_m2",
                "gcat_pl_name",
            ]
        ],
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
