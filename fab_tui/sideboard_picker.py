"""Interactive sideboard swap picker for the TUI."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from fab_tui.card_search import CardHit, CardSearchIndex
from fab_tui.decks import read_deck_hero_info
from fab_tui.equipment import (
    EquipmentSearchIndex,
    parse_equipment_header,
    slot_display_name,
    suggest_guide_equipment_header,
)
from flesh_and_blood_rlbridge.sideboard_guide_policy import simulate_guide_sideboard_deck

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAINING_ROOT = _REPO_ROOT / "scripts" / "training"
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from train_pipeline_common import (  # noqa: E402
    PhaseAgents,
    ensure_pool_metadata,
    greedy_game_deck_cut,
    min_deck_size_for_format,
)


@dataclass
class PolicyBaselineChoice:
    """Baseline game deck chosen before play training starts."""

    baseline_deck: dict[str, int]
    card_pool: dict[str, int]
    baseline_label: str
    source: str
    equipment_header: str = ""


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


def _pool_by_id_for(
    card_pool: dict[str, int],
    *,
    hero_id: str,
    hero_class: str,
    game_format: str,
) -> dict[str, Any]:
    agents = PhaseAgents(player="p1", card_pool=dict(card_pool))
    ensure_pool_metadata(
        agents,
        hero_id=hero_id,
        hero_class=hero_class,
        game_format=game_format,
    )
    return agents.pool_by_id


def compute_guide_policy_deck(
    card_pool: dict[str, int],
    *,
    opponent_hero_id: str,
    hero_id: str,
    game_format: str,
    hero_class: str,
) -> dict[str, int]:
    """Return the SideboardGuidePolicy game deck for a matchup."""
    pool_by_id = _pool_by_id_for(
        card_pool,
        hero_id=hero_id,
        hero_class=hero_class,
        game_format=game_format,
    )
    guide_deck = simulate_guide_sideboard_deck(
        card_pool,
        opponent_hero_id,
        hero_id=hero_id,
        game_format=game_format,
        pool_by_id=pool_by_id,
    )
    min_size = min_deck_size_for_format(game_format)
    if guide_deck:
        total = sum(int(v) for v in guide_deck.values())
        if total > min_size:
            return greedy_game_deck_cut(guide_deck, min_size)
        if total >= min_size:
            return guide_deck
    return greedy_game_deck_cut(card_pool, min_size)


def _show_equipment_loadout(
    console: Console,
    *,
    equipment_header: str,
    hero_id: str,
    equip_index: EquipmentSearchIndex,
    title: str = "Equipment loadout",
) -> None:
    entries = parse_equipment_header(
        equipment_header,
        hero_id=hero_id,
        display_name=equip_index.display_name,
    )
    if not entries:
        console.print(f"[dim]{title}: (none)[/dim]")
        return

    table = Table(title=title, box=box.ROUNDED)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Slot")
    table.add_column("Card")
    table.add_column("ID", style="dim")
    for entry in entries:
        table.add_row(
            str(entry.index),
            slot_display_name(entry.slot),
            entry.label,
            entry.card_id,
        )
    console.print(table)


def _pick_equipment_replacement(
    console: Console,
    equip_index: EquipmentSearchIndex,
    *,
    slot: str,
    slot_label: str,
) -> str | None:
    while True:
        query = Prompt.ask(
            f"Replacement for {slot_label} [name search, or Enter to list]",
            default="",
        ).strip()
        if not query:
            hits = equip_index.search("", slot=slot, limit=16)
        else:
            hits = equip_index.search(query, slot=slot, limit=12)
        if not hits:
            console.print("[yellow]No matching equipment.[/yellow]")
            if not Confirm.ask("Try another search?", default=True):
                return None
            continue

        table = Table(title=f"Equipment matches ({slot_label})", box=box.SIMPLE)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Name")
        table.add_column("ID", style="dim")
        for index, hit in enumerate(hits, start=1):
            table.add_row(str(index), hit.name, hit.card_id)
        console.print(table)

        choice = Prompt.ask("Select # (or Enter to skip)", default="").strip()
        if not choice:
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(hits):
                return hits[idx - 1].card_id
        console.print("[red]Invalid selection.[/red]")


def _refine_equipment_until_happy(
    console: Console,
    equipment_header: str,
    *,
    hero_id: str,
    equip_index: EquipmentSearchIndex,
) -> tuple[str, list[tuple[str, str]]]:
    """Let the user swap equipment until they confirm the loadout."""
    working_header = equipment_header
    changes: list[tuple[str, str]] = []

    while True:
        _header(console, "Equipment loadout")
        _show_equipment_loadout(
            console,
            equipment_header=working_header,
            hero_id=hero_id,
            equip_index=equip_index,
            title="Current equipment",
        )
        if Confirm.ask("Happy with this equipment loadout?", default=True):
            return working_header, changes

        entries = parse_equipment_header(
            working_header,
            hero_id=hero_id,
            display_name=equip_index.display_name,
        )
        if not entries:
            console.print("[yellow]No equipment to edit — continuing.[/yellow]")
            return working_header, changes

        choice = Prompt.ask(
            "Slot # to replace (hero row cannot be changed)",
            default="",
        ).strip()
        if not choice or not choice.isdigit():
            console.print("[yellow]Enter a slot number from the table, or confirm when ready.[/yellow]")
            continue
        idx = int(choice)
        entry = next((row for row in entries if row.index == idx), None)
        if entry is None:
            console.print("[red]Invalid slot number.[/red]")
            continue
        if entry.slot == "hero":
            console.print("[yellow]The hero card cannot be swapped here.[/yellow]")
            continue

        replacement = _pick_equipment_replacement(
            console,
            equip_index,
            slot=entry.slot,
            slot_label=slot_display_name(entry.slot),
        )
        if not replacement:
            continue

        parts = working_header.split()
        if 1 <= idx <= len(parts):
            old_id = parts[idx - 1]
            parts[idx - 1] = replacement
            working_header = " ".join(parts)
            changes.append((old_id, replacement))
            console.print(
                f"[green]Equipped[/green] {equip_index.display_name(replacement)} "
                f"in {slot_display_name(entry.slot)}"
            )


def _refine_deck_until_happy(
    console: Console,
    baseline_deck: dict[str, int],
    card_pool: dict[str, int],
    card_index: CardSearchIndex,
) -> tuple[dict[str, int], dict[str, int], list[tuple[str, str]]]:
    """Let the user swap deck cards until they confirm the list."""
    working_deck = dict(baseline_deck)
    working_pool = dict(card_pool)
    swaps: list[tuple[str, str]] = []

    while True:
        _header(console, "Baseline deck")
        _show_default_deck(
            console,
            game_deck=working_deck,
            card_pool=working_pool,
            card_index=card_index,
            title="Current deck",
        )
        if Confirm.ask("Happy with this deck?", default=True):
            return working_deck, working_pool, swaps

        console.print("[cyan]Swap one card out of the deck, then pick a replacement.[/cyan]")
        out_card = _pick_card_out(console, working_deck, card_index)
        if not out_card:
            console.print("[yellow]Swap cancelled — review the deck and confirm when ready.[/yellow]")
            continue

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
            console.print("[yellow]Swap cancelled — review the deck and confirm when ready.[/yellow]")
            continue

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


def prompt_policy_baseline_deck(
    console: Console,
    deck_path: Path,
    *,
    opponent_hero_id: str,
    hero_id: str,
    hero_class: str,
    game_format: str,
    saved_list: bool = False,
) -> PolicyBaselineChoice:
    """Show guide-policy recommendations, then refine equipment and deck."""
    file_deck, card_pool = load_deck_and_pool(deck_path)
    card_index = CardSearchIndex(game_format)
    equip_index = EquipmentSearchIndex(game_format, hero_id=hero_id)
    info = read_deck_hero_info(deck_path)
    file_equipment = info.equipment_header if info else ""

    skip_loadout_refinement = False

    if saved_list:
        _header(console, "Saved sideboard list")
        if info:
            console.print(f"[bold]{info.name or info.hero_id}[/bold]  ({info.hero_id})")
        console.print(f"Opponent matchup: [bold]{opponent_hero_id}[/bold]\n")
        console.print(
            "[dim]Adjust equipment first, then deck cards — confirm each when you are happy.[/dim]\n"
        )
        baseline_deck = dict(file_deck)
        equipment_header = file_equipment
        source = "saved"
        baseline_label = (info.name if info else "") or "Saved list"
    else:
        pool_by_id = _pool_by_id_for(
            card_pool,
            hero_id=hero_id,
            hero_class=hero_class,
            game_format=game_format,
        )
        guide_deck = compute_guide_policy_deck(
            card_pool,
            opponent_hero_id=opponent_hero_id,
            hero_id=hero_id,
            game_format=game_format,
            hero_class=hero_class,
        )
        guide_equipment = suggest_guide_equipment_header(
            file_equipment,
            hero_id=hero_id,
            opponent_hero_id=opponent_hero_id,
            game_format=game_format,
            pool_by_id=pool_by_id,
        )

        _header(console, "Sideboard guide policy")
        if info:
            console.print(f"[bold]{info.name or info.hero_id}[/bold]  ({info.hero_id})")
        console.print(f"Opponent matchup: [bold]{opponent_hero_id}[/bold]\n")
        console.print(
            "[dim]Review the starting list vs guide recommendations. "
            "If you accept the guide, equipment and deck are used as-is; "
            "otherwise adjust each loadout and confirm when happy.[/dim]\n"
        )
        _show_default_deck(
            console,
            game_deck=file_deck,
            card_pool=card_pool,
            card_index=card_index,
            title="Starting deck (from file)",
        )
        _show_equipment_loadout(
            console,
            equipment_header=file_equipment,
            hero_id=hero_id,
            equip_index=equip_index,
            title="Starting equipment (from file)",
        )
        _show_default_deck(
            console,
            game_deck=guide_deck,
            card_pool=card_pool,
            card_index=card_index,
            title="Guide policy deck (recommended)",
        )
        _show_equipment_loadout(
            console,
            equipment_header=guide_equipment,
            hero_id=hero_id,
            equip_index=equip_index,
            title="Guide policy equipment (recommended)",
        )

        use_guide = Confirm.ask(
            "Start from guide policy recommendations?",
            default=True,
        )
        if use_guide:
            baseline_deck = dict(guide_deck)
            equipment_header = guide_equipment
            source = "guide_policy"
            baseline_label = "Guide policy deck"
            skip_loadout_refinement = True
        else:
            baseline_deck = dict(file_deck)
            equipment_header = file_equipment
            source = "deck_file"
            baseline_label = "Starting deck"

    if skip_loadout_refinement:
        equip_changes: list[tuple[str, str]] = []
        swaps: list[tuple[str, str]] = []
    else:
        console.print(
            "\n[bold]Step 1 — Equipment[/bold]  "
            "[dim]Swap slots until you are happy with the loadout.[/dim]"
        )
        equipment_header, equip_changes = _refine_equipment_until_happy(
            console,
            equipment_header,
            hero_id=hero_id,
            equip_index=equip_index,
        )

        console.print(
            "\n[bold]Step 2 — Deck cards[/bold]  "
            "[dim]Swap cards until you are happy with the list.[/dim]"
        )
        baseline_deck, card_pool, swaps = _refine_deck_until_happy(
            console,
            baseline_deck,
            card_pool,
            card_index,
        )

    if equip_changes:
        source = f"{source}_equip_edited"
        equip_label = ", ".join(
            f"{equip_index.display_name(o)}→{equip_index.display_name(i)}"
            for o, i in equip_changes
        )
        baseline_label = f"{baseline_label} (equip {equip_label})"
    if swaps:
        source = f"{source}_edited"
        swap_label = ", ".join(
            f"{card_index.display_name(o)}→{card_index.display_name(i)}"
            for o, i in swaps
        )
        if equip_changes:
            baseline_label = f"{baseline_label}; {swap_label}"
        else:
            baseline_label = f"{baseline_label} ({swap_label})"

    return PolicyBaselineChoice(
        baseline_deck=baseline_deck,
        card_pool=card_pool,
        baseline_label=baseline_label,
        source=source,
        equipment_header=equipment_header,
    )


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
    baseline_deck: dict[str, int] | None = None,
    card_pool: dict[str, int] | None = None,
) -> tuple[list[ManualSwapVariant], dict[str, int]]:
    """Build alternate swap variants from the chosen baseline deck."""
    if baseline_deck is None or card_pool is None:
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
    else:
        baseline_deck = dict(baseline_deck)
        card_pool = dict(card_pool)
        card_index = CardSearchIndex(game_format)

    max_variants = max(0, min(max_variants, 10))
    max_swaps = max(1, min(max_swaps_per_variant, 5))
    if max_variants == 0:
        return [], card_pool

    console.print(
        f"\nConfigure up to [bold]{max_variants}[/bold] alternate list(s) "
        f"([bold]{max_swaps}[/bold] swap(s) each), starting from the baseline above."
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
    baseline_label: str = "Default deck",
    equipment_header: str = "",
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if include_baseline:
        baseline_entry: dict[str, Any] = {
            "candidate_id": "baseline",
            "label": baseline_label,
            "game_deck": dict(baseline_deck),
            "swaps": [],
        }
        if equipment_header:
            baseline_entry["equipment_header"] = equipment_header
        candidates.append(baseline_entry)
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
    baseline_label: str = "Default deck",
    equipment_header: str = "",
) -> Path:
    payload = variants_to_candidate_payload(
        baseline_deck,
        variants,
        baseline_label=baseline_label,
        equipment_header=equipment_header,
    )
    payload["card_pool"] = dict(card_pool)
    if equipment_header:
        payload["equipment_header"] = equipment_header
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
