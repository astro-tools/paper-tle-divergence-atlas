"""Unit tests for sweep.strip_arxiv_tarball."""

from __future__ import annotations

import tarfile
from pathlib import Path

from sweep.strip_arxiv_tarball import strip_tarball


def _build_dirty_tarball(tar_path: Path) -> None:
    """Construct a tarball mimicking showyourwork v0.4.3's output.

    The dirty shape: ``figures/`` and ``tables/`` subdirs are correct,
    plus every file under them appears a second time at the root under
    just the basename, plus a ``.snakemake_timestamp`` dotfile.
    """
    staging = tar_path.parent / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    (staging / "ms.tex").write_text(r"\documentclass{article}\begin{document}hi\end{document}")
    (staging / "ms.bbl").write_text("bbl stub")
    (staging / "showyourwork.sty").write_text("syw stub")
    (staging / "lineno.sty").write_text("lineno stub")
    (staging / ".snakemake_timestamp").write_text("0")

    figures = staging / "figures"
    figures.mkdir()
    (figures / "fig_one.pdf").write_bytes(b"PDF1")
    (figures / "fig_two.pdf").write_bytes(b"PDF2")

    tables = staging / "tables"
    tables.mkdir()
    (tables / "tab_one.tex").write_text("table one")

    # Root-level dupes by basename.
    (staging / "fig_one.pdf").write_bytes(b"PDF1")
    (staging / "fig_two.pdf").write_bytes(b"PDF2")
    (staging / "tab_one.tex").write_text("table one")

    with tarfile.open(tar_path, "w:gz") as tf:
        for entry in sorted(staging.iterdir()):
            tf.add(entry, arcname=entry.name, recursive=True)


def _entries(tar_path: Path) -> list[str]:
    with tarfile.open(tar_path, "r:gz") as tf:
        return sorted(m.name for m in tf.getmembers())


def test_strip_removes_dotfiles_and_root_dupes(tmp_path: Path):
    src = tmp_path / "dirty.tar.gz"
    _build_dirty_tarball(src)

    dst = tmp_path / "clean.tar.gz"
    kept, dropped = strip_tarball(src, dst)

    # Three dropped: .snakemake_timestamp + two fig dupes + one tab dupe = 4.
    assert dropped == 4
    assert kept >= 1

    names = _entries(dst)
    # No dotfiles at the root of the clean archive.
    assert not any(n.startswith(".") or "/." in n for n in names), names
    # No root-level duplicates of files that also live under figures/ or tables/.
    assert "fig_one.pdf" not in names
    assert "fig_two.pdf" not in names
    assert "tab_one.tex" not in names
    # figures/ and tables/ contents survive.
    assert "figures/fig_one.pdf" in names
    assert "figures/fig_two.pdf" in names
    assert "tables/tab_one.tex" in names
    # Non-conflicting root entries are preserved.
    assert "ms.tex" in names
    assert "ms.bbl" in names
    assert "showyourwork.sty" in names
    assert "lineno.sty" in names


def test_strip_in_place(tmp_path: Path):
    """When --dst equals --src the helper still produces a clean tarball."""
    src = tmp_path / "in_place.tar.gz"
    _build_dirty_tarball(src)

    # Mimic the CLI's in-place codepath by routing through an intermediate.
    intermediate = src.with_suffix(src.suffix + ".tmp")
    strip_tarball(src, intermediate)
    intermediate.replace(src)

    names = _entries(src)
    assert "fig_one.pdf" not in names
    assert "figures/fig_one.pdf" in names
    assert ".snakemake_timestamp" not in names


def test_strip_handles_already_clean(tmp_path: Path):
    """Round-tripping a clean tarball is a no-op on the root level."""
    src = tmp_path / "clean.tar.gz"

    # Build a clean tarball (figures/ and tables/ only, no root dupes).
    staging = tmp_path / "_clean_staging"
    staging.mkdir()
    (staging / "ms.tex").write_text("x")
    (staging / "figures").mkdir()
    (staging / "figures" / "fig.pdf").write_bytes(b"P")
    with tarfile.open(src, "w:gz") as tf:
        for entry in sorted(staging.iterdir()):
            tf.add(entry, arcname=entry.name, recursive=True)

    dst = tmp_path / "round.tar.gz"
    kept, dropped = strip_tarball(src, dst)

    assert dropped == 0
    assert kept == 2  # ms.tex + figures/
    assert _entries(dst) == ["figures", "figures/fig.pdf", "ms.tex"]
