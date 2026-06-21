"""Interactive sideboard swap picker for the TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from fab_tui.card_search import CardHit, CardSearchIndex
from fab_tui.decks import read_deck_hero_info


@dataclass
class ManualSwapVariant:
    candidate_id: str
    label: str
    game_deck: dict[str, int]
    swaps: list[tuple[str, str]]


def load_deck_and_pool(deck_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    data = json.loads(deck_path.read_text(encoding="utf-8"))
    game_deck = {
        str(k): int(v) for k, v in (data.get("deck") or {}).items() if int(v) > 0
    }
    sideboard = {
        str(k): int(v) for k, v in (data.get("sideboard") or {}).items() if int(v) > 0
    }
    pool = dict(game_deck)
    for cid, count in sideboard.items():
        pool[cid] = pool.get(cid, 0) + count
    return game_deck, pool


def apply_manual_swap(
    game_deck: dict[str, int],
    card_pool: dict[str, int],
    out_card: str,
    in_card: str,
) -> tuple[dict[str, int], dict[str, int]] | None:
    """Swap one copy out → in, expanding the registered pool when needed."""
    deck = {str(k): int(v) for k, v in game_deck.items() if int(v) > 0}
    pool = {str(k): int(v) for k, v in card_pool.items() if int(v) >= 0}
    if deck.get(out_card, 0) <= 0:
        return None

    inventory = {
        cid: pool.get(cid, 0) - deck.get(cid, 0)
        for cid in set(pool) | set(deck)
    }
    inventory = {cid: count for cid, count in inventory.items() if count > 0}

    if inventory.get(in_card, 0) <= 0:
        pool[in_card] = pool.get(in_card, 0) + 1

    deck[out_card] -= 1
    if deck[out_card] <= 0:
        del deck[out_card]
    deck[in_card] = deck.get(in_card, 0) + 1
    return deck, pool


def _deck_entries(game_deck: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(
        ((cid, count) for cid, count in game_deck.items() if count > 0),
        key=lambda item: item[0],
    )


def _show_default_deck(
    console: Console,
    *,
    game_deck: dict[str, int],
    card_pool: dict[str, int],
    card_index: CardSearchIndex,
    title: str = "Default game deck",
) -> None:
    inventory = {
        cid: card_pool.get(cid, 0) - game_deck.get(cid, 0)
        for cid in set(card_pool) | set(game_deck)
    }
    inventory = {cid: c for cid, c in inventory.items() if c > 0}

    table = Table(title=title, box=box.ROUNDED)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Card")
    table.add_column("ID", style="dim")
    table.add_column("×", justify="right")
    for index, (cid, count) in enumerate(_deck_entries(game_deck), start=1):
        hit = card_index.lookup(cid)
        pitch = f"P{hit.pitch} " if hit and hit.pitch else ""
        name = hit.name if hit else card_index.display_name(cid)
        table.add_row(str(index), f"{pitch}{name}", cid, str(count))
    console.print(table)
    console.print(
        f"[dim]{sum(game_deck.values())} cards in deck  |  "
        f"{sum(inventory.values())} in sideboard inventory[/dim]"
    )


def _pick_from_search_results(
    console: Console,
    hits: list[CardHit],
    *,
    prompt: str,
) -> CardHit | None:
    if not hits:
        console.print("[yellow]No matches.[/yellow]")
        return None

    table = Table(title=prompt, box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Pitch", justify="center")
    table.add_column("Name")
    table.add_column("ID", style="dim")
    for index, hit in enumerate(hits, start=1):
        pitch = str(hit.pitch) if hit.pitch is not None else "-"
        table.add_row(str(index), pitch, hit.name, hit.card_id)
    console.print(table)

    choice = Prompt.ask("Select # (or Enter to skip)", default="").strip()
    if not choice:
        return None
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(hits):
            return hits[idx - 1]
    return None


def _pick_card_out(
    console: Console,
    game_deck: dict[str, int],
    card_index: CardSearchIndex,
) -> str | None:
    entries = _deck_entries(game_deck)
    if not entries:
        return None

    while True:
        query = Prompt.ask(
            "Card to remove [number, name search, or Enter to list deck]",
            default="",
        ).strip()
        if not query:
            _show_default_deck(
                console,
                game_deck=game_deck,
                card_pool=game_deck,
                card_index=card_index,
                title="Pick a card to swap out",
            )
            continue
        if query.isdigit():
            idx = int(query)
            if 1 <= idx <= len(entries):
                return entries[idx - 1][0]
            console.print("[red]Invalid deck number.[/red]")
            continue

        filtered = [
            (cid, count)
            for cid, count in entries
            if query.lower() in cid.lower()
            or query.lower() in card_index.display_name(cid).lower()
        ]
        if len(filtered) == 1:
            return filtered[0][0]
        if filtered:
            hits = [
                CardHit(
                    card_id=cid,
                    name=card_index.display_name(cid),
                )
                for cid, _ in filtered[:12]
            ]
            picked = _pick_from_search_results(
                console, hits, prompt="Matching deck cards"
            )
            if picked:
                return picked.card_id
            continue

        console.print("[yellow]No matching card in deck.[/yellow]")


def _pick_card_in(
    console: Console,
    card_index: CardSearchIndex,
    *,
    inventory: dict[str, int],
) -> str | None:
    while True:
        query = Prompt.ask(
            "Card to add [name search — searches full card DB]",
            default="",
        ).strip()
        if not query:
            if inventory:
                table = Table(title="Sideboard inventory", box=box.SIMPLE)
                table.add_column("#", style="cyan", justify="right")
                table.add_column("Card")
                table.add_column("×", justify="right")
                inv_entries = sorted(inventory.items())
                for index, (cid, count) in enumerate(inv_entries, start=1):
                    table.add_row(
                        str(index),
                        card_index.display_name(cid),
                        str(count),
                    )
                console.print(table)
                choice = Prompt.ask("Pick inventory # or type a search", default="").strip()
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(inv_entries):
                        return inv_entries[idx - 1][0]
            continue

        hits = card_index.search(query, limit=12)
        picked = _pick_from_search_results(console, hits, prompt=f"Matches for '{query}'")
        if picked:
            return picked.card_id


def configure_manual_swap_variants(
    console: Console,
    deck_path: Path,
    *,
    game_format: str,
    max_variants: int = 2,
    max_swaps_per_variant: int = 1,
) -> tuple[list[ManualSwapVariant], dict[str, int]]:
    """Build baseline + user-defined swap variants from the default deck."""
    baseline_deck, card_pool = load_deck_and_pool(deck_path)
    card_index = CardSearchIndex(game_format)

    _header(console, "Default deck")
    info = read_deck_hero_info(deck_path)
    if info:
        console.print(f"[bold]{info.name or info.hero_id}[/bold]  ({info.hero_id})")
    _show_default_deck(
        console,
        game_deck=baseline_deck,
        card_pool=card_pool,
        card_index=card_index,
    )

    max_variants = max(0, min(max_variants, 10))
    max_swaps = max(1, min(max_swaps_per_variant, 5))
    if max_variants == 0:
        return [], card_pool

    console.print(
        f"\nConfigure up to [bold]{max_variants}[/bold] alternate list(s) "
        f"([bold]{max_swaps}[/bold] swap(s) each)."
    )

    variants: list[ManualSwapVariant] = []
    expanded_pool = dict(card_pool)

    for variant_no in range(1, max_variants + 1):
        if not Confirm.ask(f"Configure variant {variant_no}?", default=variant_no == 1):
            if variant_no == 1 and not variants:
                continue
            break

        working_deck = dict(baseline_deck)
        working_pool = dict(expanded_pool)
        swaps: list[tuple[str, str]] = []

        for swap_no in range(1, max_swaps + 1):
            console.print(f"\n[cyan]Variant {variant_no} — swap {swap_no}/{max_swaps}[/cyan]")
            out_card = _pick_card_out(console, working_deck, card_index)
            if not out_card:
                console.print("[yellow]Swap skipped.[/yellow]")
                break

            inventory = {
                cid: working_pool.get(cid, 0) - working_deck.get(cid, 0)
                for cid in set(working_pool) | set(working_deck)
            }
            inventory = {cid: c for cid, c in inventory.items() if c > 0}

            in_card = _pick_card_in(
                console,
                card_index,
                inventory=inventory,
            )
            if not in_card:
                console.print("[yellow]Swap skipped.[/yellow]")
                break

            result = apply_manual_swap(working_deck, working_pool, out_card, in_card)
            if result is None:
                console.print("[red]Invalid swap — try again.[/red]")
                continue
            working_deck, working_pool = result
            swaps.append((out_card, in_card))
            console.print(
                f"[green]Swapped[/green] {card_index.display_name(out_card)} "
                f"→ {card_index.display_name(in_card)}"
            )

        if not swaps:
            continue

        expanded_pool = working_pool
        swap_label = ", ".join(
            f"{card_index.display_name(o)}→{card_index.display_name(i)}"
            for o, i in swaps
        )
        variants.append(
            ManualSwapVariant(
                candidate_id=f"manual_{variant_no:02d}",
                label=f"Manual: {swap_label}",
                game_deck=dict(working_deck),
                swaps=swaps,
            )
        )

    return variants, expanded_pool


def _header(console: Console, title: str) -> None:
    console.print(f"\n[bold cyan]{title}[/bold cyan]")


def variants_to_candidate_payload(
    baseline_deck: dict[str, int],
    variants: list[ManualSwapVariant],
    *,
    include_baseline: bool = True,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if include_baseline:
        candidates.append(
            {
                "candidate_id": "baseline",
                "label": "Default deck",
                "game_deck": dict(baseline_deck),
                "swaps": [],
            }
        )
    for variant in variants:
        candidates.append(
            {
                "candidate_id": variant.candidate_id,
                "label": variant.label,
                "game_deck": dict(variant.game_deck),
                "swaps": [list(pair) for pair in variant.swaps],
            }
        )
    return {"candidates": candidates}


def write_candidates_manifest(
    path: Path,
    *,
    baseline_deck: dict[str, int],
    card_pool: dict[str, int],
    variants: list[ManualSwapVariant],
) -> Path:
    payload = variants_to_candidate_payload(baseline_deck, variants)
    payload["card_pool"] = dict(card_pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
