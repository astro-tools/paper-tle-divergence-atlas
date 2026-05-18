"""Strip dotfiles and root-level figure/table dupes from an arXiv tarball.

showyourwork v0.4.3's ``arxiv.py`` builds the submission tarball by
walking ``.showyourwork/compile/`` with ``Path.rglob("*")`` and calling
``tarball.add(file, arcname=file.name)`` for every entry, including
directories. ``tarfile.add()`` defaults to ``recursive=True``, so the
``figures/`` directory is added with its contents under the
``figures/`` prefix *and* each file inside it is added a second time at
the archive root under just the basename. Same for ``tables/``. The
result is an otherwise-clean tarball with ~17 root-level duplicate
``fig_*.pdf`` / ``tab_*.tex`` entries plus a ``.snakemake_timestamp``
dotfile that confuses the arXiv pdflatex auto-detect.

This module post-processes the showyourwork output: extract to a temp
dir, delete root-level dotfiles, delete root-level files whose
basename also appears under ``figures/`` or ``tables/``, then re-tar.
The figure/table subdirectories themselves and every non-conflicting
root entry (``ms.tex``, ``ms.bbl``, ``showyourwork.sty``, ``.otf``
fonts, ``-stamp.pdf``, ``-logo.pdf``, ``-metadata.tex``, ``lineno.sty``,
``xstring.sty``, ``xstring.tex``, ``listofitems.tex``) are preserved.

The Makefile ``arxiv-tarball`` target wraps the showyourwork build and
this post-processor, so a single ``make arxiv-tarball`` produces an
arXiv-ready ``arxiv.tar.gz`` at the repo root.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path


def _children_basenames(directory: Path) -> set[str]:
    """Return the set of file basenames directly under `directory`, or empty."""
    if not directory.is_dir():
        return set()
    return {p.name for p in directory.iterdir() if p.is_file()}


def strip_tarball(src: Path, dst: Path) -> tuple[int, int]:
    """Re-emit `src` into `dst` with dotfiles and root-level fig/tab dupes removed.

    Returns ``(kept, dropped)`` entry counts at the root level.
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with tarfile.open(src, "r:gz") as tf:
            # The "data" filter strips suid bits and rejects absolute /
            # parent-relative member paths — the conservative default for
            # untrusted archives (PEP 706). showyourwork emits clean
            # archives but the filter is the right idiom for Python 3.12+.
            tf.extractall(tmp, filter="data")

        bad_basenames = _children_basenames(tmp / "figures") | _children_basenames(tmp / "tables")

        kept: list[Path] = []
        dropped = 0
        for entry in sorted(tmp.iterdir()):
            if entry.is_file() and (entry.name.startswith(".") or entry.name in bad_basenames):
                entry.unlink()
                dropped += 1
                continue
            kept.append(entry)

        with tarfile.open(dst, "w:gz") as tf:
            for entry in kept:
                tf.add(entry, arcname=entry.name, recursive=True)

        return len(kept), dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("arxiv.tar.gz"),
        help="Input tarball (defaults to ./arxiv.tar.gz, where showyourwork emits it).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output tarball (defaults to overwriting --src in place).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.src.is_file():
        print(f"error: {args.src} not found", file=sys.stderr)
        return 1
    dst = args.dst if args.dst is not None else args.src
    # When dst == src we still need a real intermediate path since
    # tarfile reads and writes the same handle would race; use a sibling.
    if dst == args.src:
        intermediate = args.src.with_suffix(args.src.suffix + ".tmp")
        kept, dropped = strip_tarball(args.src, intermediate)
        intermediate.replace(dst)
    else:
        kept, dropped = strip_tarball(args.src, dst)
    print(f"wrote {dst} ({kept} root entr(ies) kept, {dropped} dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
