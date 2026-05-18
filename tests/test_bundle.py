"""Unit tests for sweep.bundle."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from sweep.bundle import BUNDLE_FILES, build_bundle


def _materialise(repo_root: Path, relative_paths: tuple[str, ...]) -> None:
    """Touch every relative path under `repo_root` with a tiny payload."""
    for rel in relative_paths:
        target = repo_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")


def test_canonical_list_is_unique_by_basename():
    # Flat-layout requirement: every entry must have a distinct basename
    # or the archive collides.
    basenames = [Path(p).name for p in BUNDLE_FILES]
    assert len(set(basenames)) == len(basenames), basenames


def test_build_bundle_writes_flat_archive(tmp_path: Path):
    _materialise(tmp_path, BUNDLE_FILES)

    out = build_bundle(tmp_path, tmp_path / "bundle.zip")

    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        names = sorted(zf.namelist())
    expected = sorted(Path(p).name for p in BUNDLE_FILES)
    assert names == expected


def test_build_bundle_errors_on_missing(tmp_path: Path):
    # Materialise everything except the last file.
    _materialise(tmp_path, BUNDLE_FILES[:-1])

    with pytest.raises(FileNotFoundError) as excinfo:
        build_bundle(tmp_path, tmp_path / "bundle.zip")

    assert BUNDLE_FILES[-1] in str(excinfo.value)
    assert not (tmp_path / "bundle.zip").exists()


def test_build_bundle_refuses_basename_collision(tmp_path: Path):
    # Two sources sharing a basename → collision in the flat archive.
    colliding = ("a/notes.txt", "b/notes.txt")
    _materialise(tmp_path, colliding)

    with pytest.raises(ValueError, match="basenames"):
        build_bundle(tmp_path, tmp_path / "bundle.zip", files=colliding)
