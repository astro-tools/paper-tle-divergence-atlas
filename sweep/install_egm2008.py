"""Install an EGM2008 70x70 potential file into GMAT.

Pipeline:

    1. Acquire NGA's flat-text spherical-harmonic coefficient file
       (`EGM2008_to2190_TideFree`), either by download or from a
       user-supplied local path.
    2. Parse it into (n, m, C_nm, S_nm) rows, truncated to
       `--max-degree` (default 70 — what the paper's force model reads).
    3. Emit a GMAT `.cof` text file with byte-for-byte the same layout
       GMAT R2026a ships for `EGM96.cof`: CRLF line endings, fixed-width
       columns, COMMENT/POTFIELD/RECOEF records.
    4. Install at `${GMAT_ROOT}/data/gravity/earth/EGM2008.cof` and write
       a small `.sha256` sidecar so re-runs are no-ops when up to date.

EGM2008 vs. EGM96 at degree 70 agree to sub-mm at LEO over multi-day
arcs (Pavlis et al. 2012, JGR Solid Earth, 10.1029/2011JB008916). The
swap is methodological fidelity, not numerical accuracy — but the
manuscript's methods section needs to cite the model actually loaded.

Usage::

    # Download from NGA and install:
    python -m sweep.install_egm2008

    # Use a pre-downloaded coefficient file:
    python -m sweep.install_egm2008 --source /path/to/EGM2008_to2190_TideFree

    # Override install target:
    python -m sweep.install_egm2008 --out /custom/path/EGM2008.cof

    # Re-emit even if up to date:
    python -m sweep.install_egm2008 --force
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# NGA's spherical-harmonic distribution page for EGM2008 has moved a few
# times; this URL has been stable since the WGS84 portal refresh and is
# what `make install-egm2008` hits by default. If NGA rotates it again,
# pass `--source <local-path>` or `--source-url <new-url>` instead.
DEFAULT_SOURCE_URL: Final = "https://earth-info.nga.mil/php/download.php?file=egm-08spherical"

# Constants from the NGA EGM2008 distribution header. Values match EGM96
# to the displayed precision; using EGM2008's own constants is the
# correct choice for an EGM2008 force model.
EGM2008_MU: Final = 3.986004415e14  # m^3 / s^2
EGM2008_RE: Final = 6.378136300e6  # m

# Local cache for the downloaded NGA source file. Not committed; the
# checksum-pinned re-emission below means stale caches are detected
# automatically.
DEFAULT_CACHE_DIR: Final = Path.home() / ".cache" / "egm2008"

# Truncation default: matches the paper's `FM.GravityField.Earth.Degree`
# in `sweep/mission.script`. Bigger files are wasted disk; smaller files
# would force a re-install if the sweep ever raised the degree.
DEFAULT_MAX_DEGREE: Final = 70

# Width of the columns in the .cof RECOEF records (Fortran-D format).
# Pinned to match EGM96.cof byte-for-byte; see `_format_signed_e_field`.
_C_FIELD_WIDTH: Final = 24
_S_FIELD_WIDTH: Final = 21


@dataclass(frozen=True)
class Coefficient:
    """A single (n, m) spherical-harmonic coefficient pair."""

    n: int
    m: int
    c: float
    s: float


def _default_install_path() -> Path:
    gmat_root = os.environ.get("GMAT_ROOT", str(Path.home() / "gmat-R2026a"))
    return Path(gmat_root) / "data" / "gravity" / "earth" / "EGM2008.cof"


def parse_nga_coefficients(text: str, *, max_degree: int) -> list[Coefficient]:
    """Parse NGA EGM2008 flat-text coefficients.

    Each non-blank line has whitespace-separated tokens
    ``n  m  C_nm  S_nm  [sigmaC  sigmaS]``. Lines with ``n < 2`` describe
    the central-body term and (unmodelled) tidal asymmetries — both
    handled separately by GMAT, so we drop them. Fortran 'D' exponents
    (e.g. ``1.23D-04``) are normalised to 'E'.
    """
    out: list[Coefficient] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            n = int(parts[0])
            m = int(parts[1])
        except ValueError:
            continue
        if n < 2 or n > max_degree:
            continue
        c = float(parts[2].replace("D", "E").replace("d", "E"))
        s = float(parts[3].replace("D", "E").replace("d", "E"))
        out.append(Coefficient(n=n, m=m, c=c, s=s))
    out.sort(key=lambda r: (r.n, r.m))
    return out


def _format_e20(value: float) -> str:
    """Format ``abs(value)`` as a 20-char unsigned scientific string.

    Pattern: ``D.DDDDDDDDDDDDDDE±NN`` (one integer digit, 14 fraction
    digits, signed 2-digit exponent). Mirrors Fortran ``E20.14`` minus
    the leading sign — sign handling is the caller's job so the same
    digits can be padded into either the 24-wide C field or the 21-wide
    S field without re-formatting.
    """
    # Python's "{:.14E}" emits e.g. "2.43914352398000E-06" (20 chars on
    # most platforms) or "2.43914352398000E-006" (21 chars on platforms
    # that emit 3-digit exponents). Normalise to 2-digit exponents.
    raw = f"{abs(value):.14E}"
    mantissa, exponent = raw.split("E")
    sign = "+" if int(exponent) >= 0 else "-"
    return f"{mantissa}E{sign}{abs(int(exponent)):02d}"


def _format_signed_e_field(value: float, width: int) -> str:
    """Render ``value`` into a fixed-width Fortran-D field.

    The numeric body is 20 chars; a leading sign character ('-' for
    negatives, ' ' for non-negatives) brings it to 21, then it is
    left-padded with spaces to ``width``. Matches the EGM96.cof
    convention where a negative S coefficient elides the inter-field
    space because its minus sign already separates the two columns.
    """
    body = _format_e20(value)
    signed = ("-" if value < 0 else " ") + body  # 21 chars
    return signed.rjust(width)


def _potfield_line(degree: int, order: int, mu: float, re: float, scale: float = 1.0) -> str:
    """Emit the single ``POTFIELD`` record (exactly 80 visible chars).

    Layout: ``POTFIELD<DDD><OOO>  <flag> <mu> <Re> <scale>`` with
    degree/order zero-padded to 3 digits and the normalisation flag set
    to 1 (fully-normalised coefficients — same as EGM96).
    """
    deg_s = f"{degree:03d}"
    ord_s = f"{order:03d}"
    mu_s = _format_e20(mu)
    re_s = _format_e20(re)
    scale_s = _format_e20(scale)
    line = f"POTFIELD{deg_s}{ord_s}  1 {mu_s} {re_s} {scale_s}"
    assert len(line) == 80, f"POTFIELD line is {len(line)} chars, expected 80"
    return line


def _recoef_line(coef: Coefficient) -> str:
    """Emit one ``RECOEF`` record.

    Zonal (``m == 0``): 38 visible chars, C only. Tesseral (``m > 0``):
    59 visible chars, C and S concatenated — the S field's sign char
    doubles as the inter-column separator.
    """
    prefix = f"RECOEF{coef.n:5d}{coef.m:3d}"
    c_field = _format_signed_e_field(coef.c, width=_C_FIELD_WIDTH)
    if coef.m == 0:
        return f"{prefix}{c_field}"
    s_field = _format_signed_e_field(coef.s, width=_S_FIELD_WIDTH)
    return f"{prefix}{c_field}{s_field}"


def emit_cof(coefficients: list[Coefficient], *, max_degree: int) -> bytes:
    """Build the full .cof file content as bytes with CRLF line endings.

    Header layout matches EGM96.cof: a ``COMMENT`` count line followed
    by three 80-char banner/description lines, then the POTFIELD record,
    then one RECOEF per coefficient.
    """
    lines: list[str] = []
    lines.append("COMMENT   3")
    lines.append("C" * 80)
    desc = f"EGM2008_to_{max_degree}x{max_degree} : truncated from NGA spherical-harmonic file"
    lines.append(f"CCCCC  {desc.ljust(66)}  CCCCC")
    lines.append("C" * 80)
    lines.append(_potfield_line(max_degree, max_degree, EGM2008_MU, EGM2008_RE))
    for coef in coefficients:
        lines.append(_recoef_line(coef))
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_NGA_INNER_FILE: Final = "EGM2008_to2190_TideFree"


def _fetch_source(url: str, cache_dir: Path) -> Path:
    """Download NGA's spherical-harmonic archive and return the local
    path to the inner flat-text coefficient file.

    NGA's ``egm-08spherical`` endpoint serves a ~109 MB ZIP containing
    ``EGM2008_to2190_TideFree`` (and a few unused Fortran-synthesis
    extras). We extract the inner file on first fetch and cache only
    that — discarding the ZIP keeps the cache footprint at ~240 MB
    instead of ~350 MB.
    """
    import requests  # local import: only needed for the download path

    cache_dir.mkdir(parents=True, exist_ok=True)
    inner = cache_dir / _NGA_INNER_FILE
    if inner.exists() and inner.stat().st_size > 0:
        return inner

    zip_path = cache_dir / "EGM2008_Spherical_Harmonics.zip"
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print(f"downloading {url} → {zip_path}", file=sys.stderr)
        with requests.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with zip_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)

    import zipfile

    print(f"extracting {_NGA_INNER_FILE} → {inner}", file=sys.stderr)
    with (
        zipfile.ZipFile(zip_path) as zf,
        zf.open(_NGA_INNER_FILE) as src,
        inner.open("wb") as dst,
    ):
        while chunk := src.read(1 << 20):
            dst.write(chunk)
    # ZIP not needed once the inner file is on disk.
    zip_path.unlink(missing_ok=True)
    return inner


def install(
    *,
    source: Path,
    out_path: Path,
    max_degree: int,
    force: bool,
) -> tuple[int, bool]:
    """Convert ``source`` to ``out_path``. Returns (n_coefficients, wrote).

    ``wrote`` is False if the install was skipped because the target was
    already up to date. Idempotency key: the SHA-256 of the .cof bytes.
    A sidecar ``<out>.sha256`` records the digest of the last write so
    we don't have to re-hash the .cof on every invocation (cheap, but
    skipping it makes the no-op path log-clean).
    """
    text = source.read_text(encoding="ascii", errors="strict")
    coefs = parse_nga_coefficients(text, max_degree=max_degree)
    if not coefs:
        raise RuntimeError(
            f"no usable coefficients parsed from {source} (max_degree={max_degree})",
        )
    payload = emit_cof(coefs, max_degree=max_degree)
    digest = _sha256_bytes(payload)

    sidecar = out_path.with_suffix(out_path.suffix + ".sha256")
    up_to_date = (
        not force
        and out_path.exists()
        and sidecar.exists()
        and sidecar.read_text().strip() == digest
    )
    if up_to_date:
        return len(coefs), False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    sidecar.write_text(digest + "\n")
    return len(coefs), True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Pre-downloaded NGA coefficient file. If omitted, the file "
        "is fetched from --source-url into --cache-dir.",
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help=f"NGA download URL (default: {DEFAULT_SOURCE_URL}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache for downloaded source files (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination .cof path (default: ${GMAT_ROOT}/data/gravity/earth/EGM2008.cof).",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=DEFAULT_MAX_DEGREE,
        help=f"Truncate coefficients to this degree (default: {DEFAULT_MAX_DEGREE}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-emit even if the target .cof is already up to date.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source or _fetch_source(args.source_url, args.cache_dir)
    out_path = args.out or _default_install_path()

    if not source.exists():
        print(f"source file not found: {source}", file=sys.stderr)
        print(
            f"  download it manually from {args.source_url} and pass it via --source, "
            f"or point --cache-dir at a directory containing EGM2008_to2190_TideFree.",
            file=sys.stderr,
        )
        return 2

    n_coefs, wrote = install(
        source=source,
        out_path=out_path,
        max_degree=args.max_degree,
        force=args.force,
    )
    if wrote:
        print(f"wrote {out_path} ({n_coefs} coefficients, degree {args.max_degree})")
    else:
        print(f"up to date: {out_path} ({n_coefs} coefficients, degree {args.max_degree})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
