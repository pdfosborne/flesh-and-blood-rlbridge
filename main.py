#!/usr/bin/env python3
"""Flesh and Blood RL Bridge — interactive experiment launcher.

Launch the terminal UI (default):

    python main.py

Or run a preset non-interactively:

    python main.py preset aurora-fixed
    python main.py preset simulate-matchup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fab_tui.app import main as tui_main, run_tui  # noqa: E402
from fab_tui.runner import run_runscript  # noqa: E402


PRESET_ALIASES: dict[str, str] = {
    "aurora-fixed": "aurora_vs_briar_fixed_opponent.py",
    "aurora-dual": "sage_aurora_vs_briar_deckbuild.py",
    "briar-dorinthea": "sage_briar_vs_dorinthea_play.py",
    "simulate-matchup": "simulate_deck_matchup.py",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("tui", help="Launch the interactive menu (default)")

    preset = sub.add_parser("preset", help="Run a named preset runscript")
    preset.add_argument(
        "name",
        choices=sorted(PRESET_ALIASES),
        help="Preset alias",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return run_tui()

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "preset":
        script = PRESET_ALIASES[args.name]
        return run_runscript(script)
    if args.command == "tui":
        return tui_main()
    return tui_main()


if __name__ == "__main__":
    raise SystemExit(main())
