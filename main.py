#!/usr/bin/env python3
"""Flesh and Blood RL Bridge — interactive experiment launcher.

Prefer the installed commands::

    fab-gui          # web GUI
    fab-tui          # terminal UI
    fab-bridge init  # first-time setup

This module remains for ``python main.py`` compatibility.
"""

from __future__ import annotations

from fab_bridge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
