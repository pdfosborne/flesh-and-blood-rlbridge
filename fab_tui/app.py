"""Rich-based terminal UI for FAB RL experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from fab_tui.config import (
    RESULTS_ROOT,
    EnvironmentSettings,
    EvalSpec,
    MatchupSimSpec,
    SideboardCompareSpec,
    slugify,
)
from fab_tui.decks import (
    DECK_CACHE,
    export_precon_deck_json,
    list_precon_options,
    read_deck_format,
    read_deck_hero_info,
    resolve_deck_link,
)
from fab_tui.sideboard_picker import (
    configure_manual_swap_variants,
    load_deck_and_pool,
    write_candidates_manifest,
)
from fab_tui.results import discover_evaluable_results, list_sideboard_candidate_ids
from fab_tui.runner import (
    fetch_fabrary_deck,
    run_eval_dashboard,
    run_matchup_simulation,
    run_sideboard_compare,
)

console = Console()

PLAYER_DECK_CHOICES = {
    "1": ("precon", "Talishar SAGE precon"),
    "2": ("fabrary", "FaBrary URL or slug"),
}

EVAL_MODE_CHOICES = {
    "1": ("eval", "Run evaluation (win-rate episodes + GIF replay)"),
    "2": ("render", "Render optimal policy only (GIF replay, no eval episodes)"),
}


def _header(title: str, subtitle: str = "") -> None:
    console.clear()
    body = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        body += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(body, border_style="cyan", box=box.ROUNDED))


def _pause() -> None:
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def _choose_mapping(mapping: dict[str, tuple[str, str]], prompt: str) -> str:
    table = Table(title=prompt, box=box.SIMPLE)
    table.add_column("Key", style="cyan", justify="center")
    table.add_column("Option")
    for key, (_, label) in mapping.items():
        table.add_row(key, label)
    console.print(table)
    while True:
        choice = Prompt.ask("Select", choices=list(mapping.keys()), default="1")
        return mapping[choice][0]


def _pick_precon_deck(env: EnvironmentSettings, *, label: str) -> Path | None:
    assets = Path(env.assets_path)
    options = list_precon_options(assets)
    if not options:
        console.print("[red]No precon deck files found in Assets.[/red]")
        return None

    table = Table(title=f"{label} — precon decks", box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Deck")
    for index, (option_label, _) in enumerate(options, start=1):
        table.add_row(str(index), option_label)
    console.print(table)

    choice = IntPrompt.ask("Select precon", default=1)
    if choice < 1 or choice > len(options):
        return None

    _, deck_name = options[choice - 1]
    cache_dir = DECK_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"precon_{slugify(deck_name)}.json"
    export_precon_deck_json(deck_name, assets, out, game_format="sage")
    console.print(f"[green]Exported precon → {out}[/green]")
    return out


def _pick_player_deck(env: EnvironmentSettings) -> Path | None:
    """Your deck — SAGE precon or FaBrary link."""
    source = _choose_mapping(PLAYER_DECK_CHOICES, "Your deck")
    if source == "precon":
        return _pick_precon_deck(env, label="Your deck")

    raw = Prompt.ask("FaBrary URL, slug, or local JSON path").strip()
    if not raw:
        return None
    cache_dir = DECK_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print("[yellow]Fetching deck…[/yellow]")
    path = resolve_deck_link(
        raw,
        label="player",
        cache_dir=cache_dir,
        fetch_fn=fetch_fabrary_deck,
    )
    if path is None:
        console.print("[red]Could not resolve deck.[/red]")
    else:
        console.print(f"[green]Deck ready → {path}[/green]")
    return path


def _pick_opponent_precon(env: EnvironmentSettings) -> tuple[str, str] | None:
    """Return ``(opponent_hero_id, opponent_deck_asset)`` from a precon."""
    assets = Path(env.assets_path)
    options = list_precon_options(assets)
    if not options:
        console.print("[red]No opponent precons found in Assets.[/red]")
        return None

    table = Table(title="Opponent precon", box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Deck")
    for index, (option_label, _) in enumerate(options, start=1):
        table.add_row(str(index), option_label)
    console.print(table)

    choice = IntPrompt.ask("Select opponent", default=2 if len(options) > 1 else 1)
    if choice < 1 or choice > len(options):
        return None

    _, deck_name = options[choice - 1]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from flesh_and_blood_rlbridge.opponent_deck import read_talishar_asset_hero_info

    info = read_talishar_asset_hero_info(env.assets_path, deck_name)
    if info is not None:
        return info.hero_id, info.asset_stem

    console.print("[yellow]Could not read hero from Assets — enter manually.[/yellow]")
    hero_id = Prompt.ask("Opponent hero id", default=deck_name.split("SAGE")[0].lower())
    return hero_id, deck_name


def _resolve_deck_link_prompt(label: str) -> Path | None:
    raw = Prompt.ask(f"{label} deck (FaBrary URL, slug, or local JSON)").strip()
    if not raw:
        return None
    cache_dir = DECK_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[yellow]Resolving {label} deck…[/yellow]")
    path = resolve_deck_link(
        raw,
        label=label.lower(),
        cache_dir=cache_dir,
        fetch_fn=fetch_fabrary_deck,
    )
    if path is None:
        console.print(f"[red]Could not resolve {label} deck.[/red]")
    else:
        console.print(f"[green]{label} deck → {path}[/green]")
    return path


def _show_sideboard_results(out_dir: Path) -> None:
    summary_path = out_dir / "sideboard_compare_results.json"
    if not summary_path.is_file():
        console.print(f"[yellow]No summary at {summary_path}[/yellow]")
        return

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    ranking = data.get("ranking") or []
    baseline_wr = data.get("baseline_final_eval_win_rate")
    has_final = any(row.get("final_eval_win_rate") is not None for row in ranking)

    if has_final:
        table = Table(title="Final eval comparison", box=box.ROUNDED)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Candidate")
        table.add_column("Train", justify="right")
        table.add_column("Final", justify="right")
        table.add_column("Δ vs base", justify="right")
        table.add_column("Label")
        for rank, row in enumerate(ranking, start=1):
            train = float(row.get("play_win_rate", 0.0))
            final = row.get("final_eval_win_rate")
            delta = row.get("final_eval_delta_vs_baseline")
            final_txt = f"{float(final) * 100:.1f}%" if final is not None else "n/a"
            if delta is not None:
                delta_txt = f"{float(delta) * 100:+.1f}%"
            elif baseline_wr is not None and final is not None:
                delta_txt = f"{(float(final) - float(baseline_wr)) * 100:+.1f}%"
            else:
                delta_txt = "n/a"
            marker = " ← winner" if rank == 1 else ""
            table.add_row(
                str(rank),
                str(row.get("candidate_id", "")),
                f"{train * 100:.1f}%",
                final_txt,
                delta_txt,
                str(row.get("label", "")) + marker,
            )
        console.print(table)
    else:
        table = Table(title="Training ranking", box=box.ROUNDED)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Candidate")
        table.add_column("Win rate")
        table.add_column("Label")
        for rank, row in enumerate(ranking, start=1):
            marker = " ← winner" if rank == 1 else ""
            table.add_row(
                str(rank),
                str(row.get("candidate_id", "")),
                f"{float(row.get('play_win_rate', 0.0)) * 100:.1f}%{marker}",
                str(row.get("label", "")),
            )
        console.print(table)

    winner = data.get("winner") or {}
    console.print(
        f"[dim]Full results: {summary_path}\n"
        f"Winning deck asset: {data.get('winning_deck_asset', 'n/a')}[/dim]"
    )
    if winner.get("out_dir"):
        console.print(f"[dim]Winner run: {winner['out_dir']}[/dim]")


def wizard_sideboard_compare(env: EnvironmentSettings) -> None:
    _header(
        "Sideboard comparison",
        "Default deck → manual swaps (card DB search) → parallel play training",
    )

    player_deck = _pick_player_deck(env)
    if player_deck is None:
        console.print("[yellow]Deck not selected.[/yellow]")
        _pause()
        return

    opponent = _pick_opponent_precon(env)
    if opponent is None:
        console.print("[yellow]Opponent not selected.[/yellow]")
        _pause()
        return
    opponent_hero_id, opponent_deck = opponent

    player_info = read_deck_hero_info(player_deck)
    game_format = read_deck_format(player_deck)
    spec = SideboardCompareSpec(
        starting_deck=str(player_deck),
        opponent_hero_id=opponent_hero_id,
        opponent_deck=opponent_deck,
        hero_id=player_info.hero_id if player_info else "",
        hero_class=player_info.hero_class if player_info else "",
        equipment_header=player_info.equipment_header if player_info else "",
        game_format=game_format,  # type: ignore[arg-type]
    )

    spec.max_swap_variants = IntPrompt.ask(
        "Alternate lists to test (besides default)",
        default=spec.max_swap_variants,
    )
    spec.max_swaps_per_variant = IntPrompt.ask(
        "Max card swaps per alternate list",
        default=spec.max_swaps_per_variant,
    )
    spec.max_parallel = IntPrompt.ask("Train in parallel (max at once)", default=spec.max_parallel)
    spec.play_episodes = IntPrompt.ask("Play episodes per list", default=spec.play_episodes)
    spec.final_eval_episodes = IntPrompt.ask(
        "Final eval games per list",
        default=spec.final_eval_episodes,
    )

    baseline_deck, _ = load_deck_and_pool(player_deck)
    variants, expanded_pool = configure_manual_swap_variants(
        console,
        player_deck,
        game_format=game_format,
        max_variants=spec.max_swap_variants,
        max_swaps_per_variant=spec.max_swaps_per_variant,
    )

    out_dir = spec.resolved_out_dir()
    candidates_path = write_candidates_manifest(
        out_dir / "candidates_manifest.json",
        baseline_deck=baseline_deck,
        card_pool=expanded_pool,
        variants=variants,
    )
    spec.candidates_json = str(candidates_path)
    spec.num_options = 1 + len(variants)
    spec.build_cpp_engine = Confirm.ask("Build/use C++ engine if available?", default=True)

    table = Table(title="Review", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    for key, value in [
        ("Your deck", str(player_deck)),
        ("Opponent", f"{opponent_hero_id} ({opponent_deck})"),
        ("Lists to compare", str(spec.num_options)),
        ("  default + alternates", f"1 + {len(variants)} manual"),
        ("Parallel", str(spec.max_parallel)),
        ("Play episodes", str(spec.play_episodes)),
        ("Final eval games", str(spec.final_eval_episodes)),
        ("Output", str(out_dir)),
    ]:
        table.add_row(key, value)
    console.print(table)

    if not variants:
        console.print(
            "[yellow]No alternate lists configured — only the default deck will be tested.[/yellow]"
        )
    if not Confirm.ask("\nStart sideboard comparison?", default=True):
        return

    console.print("\n[bold]Running sideboard comparison…[/bold]\n")
    rc = run_sideboard_compare(
        spec,
        env,
        starting_deck=player_deck,
        candidates_json=candidates_path,
    )
    _header("Sideboard comparison finished", f"Exit code {rc}")
    _show_sideboard_results(out_dir)
    _pause()


def wizard_simulate_decks(env: EnvironmentSettings) -> None:
    _header(
        "Fixed deck simulation",
        "Train play agents on two fixed decks (FaBrary links or local JSON)",
    )

    deck1 = _resolve_deck_link_prompt("Player")
    if deck1 is None:
        _pause()
        return
    deck2 = _resolve_deck_link_prompt("Opponent")
    if deck2 is None:
        _pause()
        return

    fmt1 = read_deck_format(deck1)
    fmt2 = read_deck_format(deck2)
    game_format = fmt1 if fmt1 == fmt2 else fmt1  # type: ignore[assignment]
    if fmt1 != fmt2:
        console.print(
            f"[yellow]Deck formats differ ({fmt1} vs {fmt2}); using {game_format}.[/yellow]"
        )

    spec = MatchupSimSpec(
        deck1_source=str(deck1),
        deck2_source=str(deck2),
        game_format=game_format,
    )
    spec.play_episodes = IntPrompt.ask("Play training episodes", default=spec.play_episodes)
    spec.final_eval_episodes = IntPrompt.ask(
        "Final evaluation games",
        default=spec.final_eval_episodes,
    )
    spec.build_cpp_engine = Confirm.ask("Build/use C++ engine if available?", default=True)

    table = Table(title="Review matchup", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    for key, value in [
        ("Player deck", str(deck1)),
        ("Opponent deck", str(deck2)),
        ("Format", spec.game_format),
        ("Play episodes", str(spec.play_episodes)),
        ("Final eval games", str(spec.final_eval_episodes)),
    ]:
        table.add_row(key, value)
    console.print(table)

    if not Confirm.ask("Start simulation?", default=True):
        return

    console.print("\n[bold]Running deck matchup simulation…[/bold]\n")
    rc = run_matchup_simulation(spec, env, deck1_json=deck1, deck2_json=deck2)

    from runscripts._common import read_deck_meta

    p1_meta = read_deck_meta(deck1, spec.game_format)
    p2_meta = read_deck_meta(deck2, spec.game_format)
    results_path = (
        RESULTS_ROOT
        / "matchup_sims"
        / f"{p1_meta.short_name}_vs_{p2_meta.short_name}"
        / "results.json"
    )
    _header("Simulation finished", f"Exit code {rc}")
    _show_results_summary(results_path)
    _pause()


def _show_results_summary(results_path: Path) -> None:
    if not results_path.is_file():
        console.print(f"[yellow]No results at {results_path}[/yellow]")
        return
    data = json.loads(results_path.read_text(encoding="utf-8"))
    table = Table(title="Results", box=box.ROUNDED)
    table.add_column("Player")
    table.add_column("Final eval")
    for player_key, label in (("p1", "P1"), ("p2", "P2")):
        player = data.get(player_key, {})
        final = player.get("final_eval")
        if isinstance(final, dict):
            rate = float(final.get("win_rate", 0.0))
            wins = int(final.get("wins", 0))
            losses = int(final.get("losses", 0))
            draws = int(final.get("draws", 0))
            table.add_row(label, f"{rate * 100:.1f}% ({wins}W/{losses}L/{draws}D)")
        else:
            rates = player.get("win_rates") or []
            table.add_row(label, f"training: {rates[-1] if rates else 'n/a'}")
    console.print(table)
    console.print(f"[dim]Full JSON: {results_path}[/dim]")


def _choose_results_dir(
    *,
    title: str = "Select results",
    manual_hint: str = "Enter results directory path",
) -> str:
    entries = discover_evaluable_results()
    if not entries:
        console.print(
            "[yellow]No results with phase-3 checkpoints found under results/.[/yellow]"
        )
        return Prompt.ask(manual_hint, default=str(RESULTS_ROOT))

    table = Table(title=title, box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Category")
    table.add_column("Matchup")
    table.add_column("Run started")
    table.add_column("Checkpoints")
    table.add_column("Folder")
    for index, entry in enumerate(entries, start=1):
        table.add_row(
            str(index),
            entry.category,
            entry.label,
            entry.run_started,
            entry.checkpoints_summary,
            entry.path.name,
        )
    console.print(table)
    console.print("[dim]m = enter path manually[/dim]")

    while True:
        choice = Prompt.ask("Select", default="1").strip()
        if choice.lower() == "m":
            return Prompt.ask(manual_hint, default=entries[0].display_path)
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(entries):
                return str(entries[idx - 1].path)
        console.print("[red]Invalid selection — enter a number or m.[/red]")


def wizard_evaluate(env: EnvironmentSettings) -> None:
    _header(
        "Evaluate checkpoints",
        "Run evaluation or render optimal policy from saved checkpoints",
    )
    mode = _choose_mapping(EVAL_MODE_CHOICES, "What would you like to do?")
    render_only = mode == "render"

    results_dir = _choose_results_dir(
        title="Results with phase-3 checkpoints",
        manual_hint="Results directory",
    )
    candidate_id: str | None = None
    sideboard_candidates = list_sideboard_candidate_ids(Path(results_dir))
    if sideboard_candidates:
        console.print(
            "\n[dim]Sideboard compare run — pick a candidate to watch, "
            "or leave blank for the latest checkpoint across all candidates.[/dim]"
        )
        table = Table(title="Candidates", box=box.SIMPLE)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Candidate ID")
        for index, cid in enumerate(sideboard_candidates, start=1):
            table.add_row(str(index), cid)
        console.print(table)
        choice = Prompt.ask(
            "Candidate (number, id, or blank for all)",
            default="",
        ).strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sideboard_candidates):
                candidate_id = sideboard_candidates[idx - 1]
        elif choice:
            candidate_id = choice

    if render_only:
        spec = EvalSpec(
            results_dir=results_dir,
            candidate_id=candidate_id,
            max_steps=IntPrompt.ask("Max steps for render replay", default=1000),
            watch=Confirm.ask("Watch for new checkpoints?", default=False),
            render_only=True,
        )
        if spec.watch:
            spec.poll_seconds = IntPrompt.ask("Poll interval (seconds)", default=30)
        console.print("\n[bold]Rendering optimal policy…[/bold]\n")
    else:
        spec = EvalSpec(
            results_dir=results_dir,
            candidate_id=candidate_id,
            episodes=IntPrompt.ask("Evaluation episodes", default=20),
            parallel_workers=IntPrompt.ask("Parallel workers", default=4),
            max_steps=IntPrompt.ask("Max steps per game", default=1000),
            watch=Confirm.ask("Watch for new checkpoints?", default=True),
        )
        if spec.watch:
            spec.poll_seconds = IntPrompt.ask("Poll interval (seconds)", default=30)
        console.print("\n[bold]Starting evaluation dashboard…[/bold]\n")

    rc = run_eval_dashboard(spec, env)
    finished = "Render finished" if render_only else "Evaluation finished"
    console.print(f"\n[bold]{finished} (exit {rc}).[/bold]")
    _pause()


def wizard_settings(env: EnvironmentSettings) -> EnvironmentSettings:
    _header("Settings", "Talishar and FaBrary connection defaults")

    env.talishar_url = Prompt.ask("Talishar URL", default=env.talishar_url)
    env.talishar_fe_url = Prompt.ask("Talishar frontend URL", default=env.talishar_fe_url)
    env.assets_path = Prompt.ask("Talishar Assets path", default=env.assets_path)
    env.fabrary_api_key = Prompt.ask(
        "FaBrary API key (optional)",
        default=env.fabrary_api_key,
        password=True,
    )
    env.apply_to_environ()
    console.print("[green]Settings saved for this session.[/green]")
    _pause()
    return env


def run_tui() -> int:
    env = EnvironmentSettings()
    env.apply_to_environ()

    menu_actions = {
        "1": ("Sideboard comparison", wizard_sideboard_compare),
        "2": ("Fixed deck simulation", wizard_simulate_decks),
        "3": ("Evaluate checkpoints", wizard_evaluate),
        "4": ("Settings", wizard_settings),
        "q": ("Quit", None),
    }

    while True:
        _header(
            "Flesh and Blood RL Bridge",
            "Sideboard tuning, fixed-deck matchups, and checkpoint eval",
        )
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Key", style="bold cyan", justify="center")
        table.add_column("Action")
        for key, (label, _) in menu_actions.items():
            if key != "q":
                table.add_row(key, label)
        table.add_row("q", "Quit")
        console.print(table)
        console.print(f"\n[dim]Talishar: {env.talishar_url}[/dim]")

        choice = Prompt.ask("Select", choices=list(menu_actions.keys()), default="1")
        if choice == "q":
            console.print("[dim]Goodbye.[/dim]")
            return 0

        action = menu_actions[choice][1]
        assert action is not None
        if action is wizard_settings:
            env = wizard_settings(env)
        else:
            action(env)

    return 0


def main() -> int:
    try:
        return run_tui()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
