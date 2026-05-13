"""pytest discovery glue.

Adds ``src/scripts`` to `sys.path` so figure-script modules (which are
not a package and live under `src/scripts/`) can be imported directly
from tests. The script directory is on the path used by `showyourwork`
at build time; this mirrors that import context.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src" / "scripts"))
