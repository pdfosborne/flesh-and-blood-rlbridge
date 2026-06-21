"""Interactive Silver Age (SAGE) precon selection for ``main.py sage``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from fab_tui.config import EnvironmentSettings, slugify
from fab_tui.decks import DECK_CACHE, export_precon_deck_json

console = Console()

DEFAULT_P1_HERO = "aurora"
DEFAULT_P2_HERO = "briar"


@dataclass(frozen=True)
class SagePreconOption:
    hero_slug: str
    deck_name: str
    label: str


@dataclass(frozen=True)
class SageMatchupChoice:
    p1_hero: str
    p2_hero: str
    p1_deck: Path | None = None
    p2_deck: Path | None = None


def list_sage_precon_options(assets_path: Path) -> list[SagePreconOption]:
    """Return SAGE precons whose Talishar Assets ``.txt`` files exist."""
    if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from flesh_and_blood_rlbridge.talishar_deck_assets import SAGE_PRECON_BY_HERO

    options: list[SagePreconOption] = []
    for hero, deck_name in sorted(SAGE_PRECON_BY_HERO.items()):
        if (assets_path / f"{deck_name}.txt").is_file():
            options.append(
                SagePreconOption(
                    hero_slug=hero,
                    deck_name=deck_name,
                    label=f"{hero.replace('_', ' ').title()} ({deck_name})",
                )
            )
    return options


def _export_precon(deck_name: str, assets_path: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"precon_{slugify(deck_name)}.json"
    export_precon_deck_json(deck_name, assets_path, out, game_format="sage")
    return out


def _print_precon_table(options: list[SagePreconOption], *, title: str) -> None:
    table = Table(title=title, box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Hero / precon")
    for index, option in enumerate(options, start=1):
        table.add_row(str(index), option.label)
    console.print(table)


def prompt_sage_matchup(
    *,
    assets_path: Path | None = None,
    default_p1: str = DEFAULT_P1_HERO,
    default_p2: str = DEFAULT_P2_HERO,
) -> SageMatchupChoice | None:
    """Prompt for P1/P2 SAGE precons. Returns ``None`` if the user cancels."""
    env = EnvironmentSettings()
    assets = Path(assets_path or env.assets_path)
    options = list_sage_precon_options(assets)
    if not options:
        console.print(
            "[red]No SAGE precon decks found in Talishar Assets.[/red]\n"
            f"[dim]Expected under: {assets}[/dim]"
        )
        return None

    console.print()
    console.print(
        "[bold cyan]Silver Age (SAGE) deckbuilder[/bold cyan] — choose player decks"
    )
    console.print(
        f"[dim]Default matchup: {default_p1.title()} vs {default_p2.title()}[/dim]"
    )
    console.print()

    use_default = Prompt.ask(
        f"Use default ({default_p1.title()} vs {default_p2.title()})?",
        choices=["y", "n"],
        default="y",
    )
    if use_default.lower() == "y":
        return SageMatchupChoice(p1_hero=default_p1, p2_hero=default_p2)

    _print_precon_table(options, title="Player 1 (P1) — select deck")
    p1_choice = IntPrompt.ask("P1 deck #", default=1)
    if p1_choice < 1 or p1_choice > len(options):
        console.print("[red]Invalid P1 selection.[/red]")
        return None
    p1_opt = options[p1_choice - 1]

    _print_precon_table(options, title="Player 2 (P2) — select deck")
    p2_default = next(
        (index for index in range(1, len(options) + 1) if index != p1_choice),
        1,
    )
    p2_choice = IntPrompt.ask("P2 deck #", default=p2_default)
    if p2_choice < 1 or p2_choice > len(options):
        console.print("[red]Invalid P2 selection.[/red]")
        return None
    if p2_choice == p1_choice:
        console.print("[yellow]P2 must differ from P1 — pick another deck.[/yellow]")
        return None
    p2_opt = options[p2_choice - 1]

    cache_dir = DECK_CACHE
    try:
        p1_deck = _export_precon(p1_opt.deck_name, assets, cache_dir)
        p2_deck = _export_precon(p2_opt.deck_name, assets, cache_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return None

    console.print()
    console.print(
        f"[green]Matchup:[/green] {p1_opt.label} [dim]vs[/dim] {p2_opt.label}"
    )
    console.print(f"[dim]P1 deck → {p1_deck}[/dim]")
    console.print(f"[dim]P2 deck → {p2_deck}[/dim]")
    console.print()

    return SageMatchupChoice(
        p1_hero=p1_opt.hero_slug,
        p2_hero=p2_opt.hero_slug,
        p1_deck=p1_deck,
        p2_deck=p2_deck,
    )
