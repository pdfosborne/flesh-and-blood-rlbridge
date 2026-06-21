#!/usr/bin/env python3
"""Backward-compatible entry point — use ``sage_deckbuilder.py`` instead."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runscripts.sage_deckbuilder import main

if __name__ == "__main__":
    raise SystemExit(main())
