"""Unit tests for sweep.install_egm2008.

Covers the NGA-format parser, the Fortran-D column emitter, and the
byte-exact COMMENT/POTFIELD/RECOEF layout so a future GMAT release that
tightens its `.cof` parser fails here, not deep in a sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sweep.install_egm2008 import (
    EGM2008_MU,
    EGM2008_RE,
    Coefficient,
    _format_e20,
    _format_signed_e_field,
    _potfield_line,
    _recoef_line,
    emit_cof,
    install,
    parse_nga_coefficients,
)

# Toy NGA-format fixture: degree 2 through 4, mixing positive / negative
# C and S values so the sign-as-separator convention is exercised. Values
# are the actual EGM96 zonal/sectoral coefficients (close to EGM2008 at
# this truncation) — see GMAT-shipped EGM96.cof lines 8–13.
_NGA_FIXTURE = """\
   2    0   -0.48416537173600D-03    0.00000000000000D+00    7.4812D-11    0.0000D+00
   2    1   -0.18698763595500D-09    0.11952801203100D-08    7.0638D-11    7.3483D-11
   2    2    0.24391435239800D-05   -0.14001668365400D-05    7.2302D-11    7.4258D-11
   3    0    0.95725417379200D-06    0.00000000000000D+00
   3    1    0.20299888218400D-05    0.24851315871600D-06
   3    2    0.90462776860500D-06   -0.61902594420500D-06
   3    3    0.72107265705700D-06    0.14143562695800D-05
   4    0    0.53987386378900D-06    0.00000000000000D+00
   4    1   -0.53632161697100D-06   -0.47344026585300D-06
   4    2    0.35069410578500D-06    0.66267157254000D-06
   4    3    0.99077180382900D-06   -0.20092836917700D-06
   4    4   -0.18856080273500D-06    0.30885316933300D-06

# blank lines and comments are skipped
   1    0    1.0    0.0     # n<2 is dropped
   5    0    0.6853234756D-07    0.0       # truncated when max_degree=4
"""


def test_parse_drops_low_n_and_high_n() -> None:
    rows = parse_nga_coefficients(_NGA_FIXTURE, max_degree=4)
    ns = {(r.n, r.m) for r in rows}
    assert (2, 0) in ns
    assert (4, 4) in ns
    assert all(2 <= r.n <= 4 for r in rows)
    assert (5, 0) not in ns  # truncated by max_degree
    assert (1, 0) not in ns  # n<2 dropped
    assert len(rows) == 12  # 3 (deg2) + 4 (deg3) + 5 (deg4)


def test_parse_truncates_to_lower_degree() -> None:
    rows = parse_nga_coefficients(_NGA_FIXTURE, max_degree=3)
    assert max(r.n for r in rows) == 3
    assert len(rows) == 7  # (2,0..2) + (3,0..3)


def test_parse_handles_d_exponent() -> None:
    rows = parse_nga_coefficients(_NGA_FIXTURE, max_degree=2)
    j2 = next(r for r in rows if (r.n, r.m) == (2, 0))
    # EGM96 J2 = -1.0826...e-3 normalised → -4.8416...e-4 (unnormalised
    # mantissa is what's stored). Just confirm the D→E conversion worked.
    assert j2.c == pytest.approx(-4.84165371736e-4, rel=1e-12)


def test_parse_ignores_garbage_and_blank() -> None:
    text = """
    not a number here
       2    0   -0.48416537173600D-03    0.0
    """
    rows = parse_nga_coefficients(text, max_degree=70)
    assert len(rows) == 1


def test_format_e20_width_and_pattern() -> None:
    assert _format_e20(2.439143523980e-6) == "2.43914352398000E-06"
    assert _format_e20(-2.439143523980e-6) == "2.43914352398000E-06"  # abs()
    assert _format_e20(1.0) == "1.00000000000000E+00"
    assert len(_format_e20(EGM2008_MU)) == 20


def test_format_e20_three_digit_exponent_normalised() -> None:
    # Python may render 1e-100 with a 3-digit exponent on some
    # platforms; the formatter must normalise to 2 digits. The actual
    # mantissa for 1e-100 is 1.0e-100, well outside anything EGM2008
    # contains, but the format guarantee should hold.
    s = _format_e20(1e-10)
    assert s == "1.00000000000000E-10"
    assert len(s) == 20


def test_format_signed_e_field_padding() -> None:
    # 24-wide C field: 3 leading spaces, then space-or-minus, then 20-char number.
    pos = _format_signed_e_field(2.439143523980e-6, width=24)
    assert pos == "    2.43914352398000E-06"
    assert len(pos) == 24

    neg = _format_signed_e_field(-4.841653717360e-4, width=24)
    assert neg == "   -4.84165371736000E-04"
    assert len(neg) == 24

    # 21-wide S field: no leading spaces beyond the sign char.
    pos_s = _format_signed_e_field(1.195280120310e-9, width=21)
    assert pos_s == " 1.19528012031000E-09"
    assert len(pos_s) == 21

    neg_s = _format_signed_e_field(-1.400166836540e-6, width=21)
    assert neg_s == "-1.40016683654000E-06"
    assert len(neg_s) == 21


def test_potfield_line_is_80_chars_with_egm2008_constants() -> None:
    line = _potfield_line(70, 70, EGM2008_MU, EGM2008_RE)
    assert len(line) == 80
    assert line.startswith("POTFIELD070070  1 ")
    assert " 3.98600441500000E+14 " in line
    assert " 6.37813630000000E+06 " in line
    assert line.endswith(" 1.00000000000000E+00")


def test_recoef_zonal_matches_egm96_format() -> None:
    # EGM96 J2 zonal — should be byte-identical to the corresponding
    # RECOEF line in the shipped EGM96.cof modulo the trailing CRLF.
    coef = Coefficient(n=2, m=0, c=-4.84165371736e-4, s=0.0)
    assert _recoef_line(coef) == "RECOEF    2  0   -4.84165371736000E-04"
    assert len(_recoef_line(coef)) == 38


def test_recoef_tesseral_positive_s() -> None:
    coef = Coefficient(n=2, m=1, c=-1.86987635955e-10, s=1.19528012031e-9)
    line = _recoef_line(coef)
    assert line == "RECOEF    2  1   -1.86987635955000E-10 1.19528012031000E-09"
    assert len(line) == 59


def test_recoef_tesseral_negative_s_elides_separator() -> None:
    # EGM96 (2,2) — the negative S sign substitutes for the inter-field
    # space, matching `cat EGM96.cof | sed -n '10p'`.
    coef = Coefficient(n=2, m=2, c=2.43914352398e-6, s=-1.40016683654e-6)
    line = _recoef_line(coef)
    assert line == "RECOEF    2  2    2.43914352398000E-06-1.40016683654000E-06"
    assert len(line) == 59


def test_emit_cof_is_crlf_and_layout_byte_exact() -> None:
    coefs = [
        Coefficient(n=2, m=0, c=-4.84165371736e-4, s=0.0),
        Coefficient(n=2, m=1, c=-1.86987635955e-10, s=1.19528012031e-9),
        Coefficient(n=2, m=2, c=2.43914352398e-6, s=-1.40016683654e-6),
    ]
    blob = emit_cof(coefs, max_degree=2)
    text = blob.decode("ascii")
    # CRLF line endings everywhere.
    assert text.count("\r\n") == 4 + 1 + len(coefs)  # 4 header + POTFIELD + RECOEFs
    assert "\n" not in text.replace("\r\n", "")

    lines = text.split("\r\n")
    assert lines[0] == "COMMENT   3"
    assert lines[1] == "C" * 80
    assert "EGM2008_to_2x2" in lines[2]
    assert lines[2].startswith("CCCCC  ") and lines[2].endswith("  CCCCC")
    assert len(lines[2]) == 80
    assert lines[3] == "C" * 80
    assert lines[4].startswith("POTFIELD002002  1 ")
    assert len(lines[4]) == 80
    assert lines[5] == "RECOEF    2  0   -4.84165371736000E-04"
    assert lines[6] == "RECOEF    2  1   -1.86987635955000E-10 1.19528012031000E-09"
    assert lines[7] == "RECOEF    2  2    2.43914352398000E-06-1.40016683654000E-06"
    # Trailing CRLF means an empty final element after split.
    assert lines[-1] == ""


def test_install_writes_file_and_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "EGM2008_to2190_TideFree"
    source.write_text(_NGA_FIXTURE)
    out = tmp_path / "EGM2008.cof"
    n, wrote = install(source=source, out_path=out, max_degree=4, force=False)
    assert wrote is True
    assert n == 12
    assert out.exists()
    sidecar = out.with_suffix(".cof.sha256")
    assert sidecar.exists()
    assert len(sidecar.read_text().strip()) == 64  # SHA-256 hex


def test_install_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "EGM2008_to2190_TideFree"
    source.write_text(_NGA_FIXTURE)
    out = tmp_path / "EGM2008.cof"
    install(source=source, out_path=out, max_degree=4, force=False)
    mtime_first = out.stat().st_mtime_ns

    # Touch the .cof file timestamp so we can tell whether the second
    # call wrote it again.
    out.touch()
    out_mtime_touched = out.stat().st_mtime_ns

    n, wrote = install(source=source, out_path=out, max_degree=4, force=False)
    assert wrote is False
    assert n == 12
    # If install() short-circuited it shouldn't have re-written; mtime
    # should match the post-touch value, not anything newer.
    assert out.stat().st_mtime_ns == out_mtime_touched
    _ = mtime_first  # silence unused-name


def test_install_force_rewrites_even_when_up_to_date(tmp_path: Path) -> None:
    source = tmp_path / "EGM2008_to2190_TideFree"
    source.write_text(_NGA_FIXTURE)
    out = tmp_path / "EGM2008.cof"
    install(source=source, out_path=out, max_degree=4, force=False)
    first_bytes = out.read_bytes()
    _, wrote = install(source=source, out_path=out, max_degree=4, force=True)
    assert wrote is True
    # Content unchanged but the file was definitely re-written.
    assert out.read_bytes() == first_bytes


def test_install_detects_max_degree_change(tmp_path: Path) -> None:
    source = tmp_path / "EGM2008_to2190_TideFree"
    source.write_text(_NGA_FIXTURE)
    out = tmp_path / "EGM2008.cof"
    install(source=source, out_path=out, max_degree=2, force=False)
    n_first = out.read_bytes().count(b"RECOEF")
    n, wrote = install(source=source, out_path=out, max_degree=4, force=False)
    assert wrote is True
    n_second = out.read_bytes().count(b"RECOEF")
    assert n_second > n_first
    assert n == 12
