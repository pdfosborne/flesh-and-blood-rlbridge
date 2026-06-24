"""Unified CLI for FAB RL Bridge (TUI, GUI, setup, Talishar)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fab_bridge.paths import configure_import_paths

configure_import_paths()

from fab_gui.server import run_gui  # noqa: E402
from fab_tui.app import main as tui_main, run_tui  # noqa: E402
from fab_tui.runner import run_runscript  # noqa: E402
from fab_tui.sage_picker import (  # noqa: E402
    DEFAULT_P1_HERO,
    DEFAULT_P2_HERO,
    prompt_sage_matchup,
)

PRESET_ALIASES: dict[str, str] = {
    "aurora-fixed": "aurora_vs_briar_fixed_opponent.py",
    "aurora-dual": "sage_deckbuilder.py",
    "sage-deckbuilder": "sage_deckbuilder.py",
    "briar-dorinthea": "sage_briar_vs_dorinthea_play.py",
    "simulate-matchup": "simulate_deck_matchup.py",
}

SAGE_RUNSCRIPT = "sage_deckbuilder.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fab-bridge",
        description="Flesh and Blood RL Bridge — launcher for GUI, TUI, and pipelines.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("tui", help="Launch the interactive terminal menu (default)")

    gui = sub.add_parser("gui", help="Launch the web GUI for sideboard comparison")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")

    doctor = sub.add_parser("doctor", help="Check dependencies and asset layout")
    doctor.add_argument(
        "--no-docker",
        action="store_true",
        help="Do not require Docker to be reachable",
    )

    init = sub.add_parser("init", help="Verify layout, create dirs, start Talishar backend")
    init.add_argument(
        "--no-talishar",
        action="store_true",
        help="Skip starting Talishar containers",
    )
    init.add_argument(
        "--with-fe",
        action="store_true",
        help="Also start Talishar-FE (Node) for live play / GIF rendering",
    )

    talishar = sub.add_parser("talishar", help="Start or stop the Talishar stack")
    talishar.add_argument("--backend-only", action="store_true")
    talishar.add_argument("--fe-only", action="store_true")
    talishar.add_argument("--down", action="store_true")

    preset = sub.add_parser("preset", help="Run a named preset runscript")
    preset.add_argument("name", choices=sorted(PRESET_ALIASES), help="Preset alias")
    preset.add_argument(
        "runscript_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the runscript (prefix with --)",
    )

    sage = sub.add_parser(
        "sage",
        help="Silver Age dual deckbuilder pipeline (default: aurora vs briar)",
    )
    sage.add_argument("p1_hero", nargs="?", default=DEFAULT_P1_HERO)
    sage.add_argument("p2_hero", nargs="?", default=DEFAULT_P2_HERO)
    sage.add_argument("--p1-hero", dest="p1_hero_flag", default=None)
    sage.add_argument("--p2-hero", dest="p2_hero_flag", default=None)
    sage.add_argument("--p1-deck", default=None, help="FaBrary slug/URL or JSON path")
    sage.add_argument("--p2-deck", default=None, help="FaBrary slug/URL or JSON path")
    sage.add_argument("--out-dir", default=None)
    sage.add_argument(
        "--format",
        default="silver_age",
        choices=["silver_age", "sage", "classic_constructed", "blitz", "upf"],
    )
    sage.add_argument("--defaults", "-y", action="store_true")
    sage.add_argument("--list-decks", action="store_true")
    sage.add_argument("--workers", type=int, default=None)
    return parser


def _clean_remainder(args: list[str]) -> list[str]:
    return [a for a in args if a != "--"]


def _sage_positional_heroes(raw_argv: list[str]) -> bool:
    positional = [token for token in raw_argv if token != "sage" and not token.startswith("-")]
    return bool(positional)


def _sage_argv(
    ns: argparse.Namespace,
    *,
    p1_hero: str | None = None,
    p2_hero: str | None = None,
    p1_deck: Path | None = None,
    p2_deck: Path | None = None,
) -> list[str]:
    argv: list[str] = []
    p1 = p1_hero or ns.p1_hero_flag or ns.p1_hero
    p2 = p2_hero or ns.p2_hero_flag or ns.p2_hero
    argv.extend([p1, p2])

    deck1 = p1_deck or (Path(ns.p1_deck) if ns.p1_deck else None)
    deck2 = p2_deck or (Path(ns.p2_deck) if ns.p2_deck else None)
    if deck1:
        argv.extend(["--p1-deck", str(deck1)])
    if deck2:
        argv.extend(["--p2-deck", str(deck2)])
    if ns.out_dir:
        argv.extend(["--out-dir", ns.out_dir])
    if ns.format:
        argv.extend(["--format", ns.format])
    if ns.workers is not None:
        argv.extend(["--workers", str(ns.workers)])
    return argv


def _sage_should_prompt(ns: argparse.Namespace, raw_argv: list[str]) -> bool:
    if ns.defaults or ns.list_decks:
        return False
    if not sys.stdin.isatty():
        return False
    if _sage_positional_heroes(raw_argv):
        return False
    if ns.p1_hero_flag or ns.p2_hero_flag or ns.p1_deck or ns.p2_deck:
        return False
    return True


def _run_sage(ns: argparse.Namespace, raw_argv: list[str]) -> int:
    if ns.list_decks:
        from fab_tui.config import EnvironmentSettings
        from fab_tui.sage_picker import list_sage_precon_options

        assets = Path(EnvironmentSettings().assets_path)
        options = list_sage_precon_options(assets)
        if not options:
            print(f"No SAGE precons found under {assets}")
            return 1
        print("Available SAGE precon decks:")
        for index, option in enumerate(options, start=1):
            print(f"  {index:2}. {option.label}")
        print()
        print(f"Default matchup: {DEFAULT_P1_HERO} vs {DEFAULT_P2_HERO}")
        print("  fab-bridge sage")
        print("  fab-bridge sage --defaults")
        return 0

    p1_hero = ns.p1_hero_flag or ns.p1_hero
    p2_hero = ns.p2_hero_flag or ns.p2_hero
    p1_deck: Path | None = Path(ns.p1_deck) if ns.p1_deck else None
    p2_deck: Path | None = Path(ns.p2_deck) if ns.p2_deck else None

    if _sage_should_prompt(ns, raw_argv):
        choice = prompt_sage_matchup()
        if choice is None:
            return 1
        p1_hero = choice.p1_hero
        p2_hero = choice.p2_hero
        p1_deck = choice.p1_deck
        p2_deck = choice.p2_deck

    return run_runscript(
        SAGE_RUNSCRIPT,
        *_sage_argv(
            ns,
            p1_hero=p1_hero,
            p2_hero=p2_hero,
            p1_deck=p1_deck,
            p2_deck=p2_deck,
        ),
    )


def _dispatch(args: argparse.Namespace, raw_argv: list[str]) -> int:
    if args.command == "doctor":
        from fab_bridge.doctor import print_report, run_doctor

        report = run_doctor(require_docker=not args.no_docker)
        return print_report(report)

    if args.command == "init":
        from fab_bridge.init_app import run_init

        return run_init(
            start_talishar=not args.no_talishar,
            backend_only=not args.with_fe,
        )

    if args.command == "talishar":
        from start_talishar import run as start_talishar_run

        if sum([args.backend_only, args.fe_only, args.down]) > 1:
            raise SystemExit("use at most one of --backend-only, --fe-only, or --down")
        return start_talishar_run(
            backend_only=args.backend_only,
            fe_only=args.fe_only,
            down=args.down,
        )

    if args.command == "preset":
        script = PRESET_ALIASES[args.name]
        extra = _clean_remainder(list(args.runscript_args or []))
        return run_runscript(script, *extra)

    if args.command == "sage":
        return _run_sage(args, raw_argv)

    if args.command == "tui":
        return tui_main()

    if args.command == "gui":
        return run_gui(host=args.host, port=args.port, open_browser=not args.no_browser)

    return run_tui()


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        return run_tui()

    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    return _dispatch(args, raw_argv)


def gui_main() -> int:
    return main(["gui"])


def tui_main_entry() -> int:
    return main(["tui"])


if __name__ == "__main__":
    raise SystemExit(main())
