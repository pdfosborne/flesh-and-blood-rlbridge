#!/usr/bin/env python3
"""Project launcher for flesh-and-blood-rlbridge.

Run with no arguments for an interactive menu, or pass a tool name to run it
directly (forwarding any remaining arguments):

    python main.py                       # interactive menu
    python main.py play --format silver_age
    python main.py update-db --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

# Make the in-repo package importable when running from a source checkout.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _run_play(args: list[str]) -> int:
    from flesh_and_blood_rlbridge.cli import main as play_main

    return int(play_main(args) or 0)


def _run_talishar(args: list[str]) -> int:
    import runpy

    argv_backup = sys.argv
    sys.argv = ["cli_talishar.py", *args]
    try:
        runpy.run_path(
            str(Path(__file__).resolve().parent / "scripts" / "cli_talishar.py"),
            run_name="__main__",
        )
        return 0
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    finally:
        sys.argv = argv_backup


def _run_train_eval_render(args: list[str]) -> int:
    import runpy

    argv_backup = sys.argv
    sys.argv = ["train_eval_render_pipeline.py", *args]
    try:
        runpy.run_path(
            str(
                Path(__file__).resolve().parent
                / "scripts"
                / "train_eval_render_pipeline.py"
            ),
            run_name="__main__",
        )
        return 0
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    finally:
        sys.argv = argv_backup


def _run_update_db(args: list[str]) -> int:
    from flesh_and_blood_rlbridge.card_db import update_cards_db_from_fabtcg as updater

    # The updater parses sys.argv directly, so present its own argv to it.
    argv_backup = sys.argv
    sys.argv = ["update_cards_db_from_fabtcg.py", *args]
    try:
        return int(updater.main() or 0)
    finally:
        sys.argv = argv_backup


_TOOLS: dict[str, tuple[str, Callable[[list[str]], int]]] = {
    "play": (
        "Play a Flesh and Blood match (pick deck + opponent, agent suggestions)",
        _run_play,
    ),
    "talishar": (
        "Play interactively via the Talishar engine (human or agent vs CombatDummy AI)",
        _run_talishar,
    ),
    "train-eval-render": (
        "Run full pipeline: train agents, evaluate head-to-head, render optimal policy states",
        _run_train_eval_render,
    ),
    "update-db": (
        "Update the card database from the official FAB Card Vault API",
        _run_update_db,
    ),
}


def _print_help() -> None:
    print("Flesh and Blood RL Bridge launcher\n")
    print("Usage: python main.py [tool] [tool-args...]\n")
    print("Tools:")
    for name, (desc, _) in _TOOLS.items():
        print(f"  {name:<12} {desc}")
    print("\nRun with no tool for an interactive menu.")


def _interactive_choice() -> Optional[str]:
    keys = list(_TOOLS)
    print("Flesh and Blood RL Bridge - choose a tool:\n")
    for i, name in enumerate(keys, 1):
        print(f"  [{i}] {name}: {_TOOLS[name][0]}")
    while True:
        raw = input(f"\nSelect [1-{len(keys)}] or name (q to quit): ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        if raw in _TOOLS:
            return raw
        print("Invalid selection; try again.")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    if argv:
        tool = argv[0]
        if tool not in _TOOLS:
            print(f"Unknown tool: {tool!r}\n")
            _print_help()
            return 2
        rest = argv[1:]
    else:
        try:
            tool = _interactive_choice()
        except (KeyboardInterrupt, EOFError):
            print()
            return 0
        if tool is None:
            print("Nothing selected.")
            return 0
        rest = []

    return _TOOLS[tool][1](rest)


if __name__ == "__main__":
    raise SystemExit(main())
