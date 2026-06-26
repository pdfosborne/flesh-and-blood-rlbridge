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
    LivePlaySpec,
    MatchupSimSpec,
    RUNTIME,
    SideboardCompareSpec,
    UnifiedRandomMatchupSpec,
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
from fab_tui.saved_decks import (
    default_saved_deck_label,
    is_saved_user_deck,
    list_saved_user_decks,
    save_user_deck,
)
from fab_tui.sideboard_picker import (
    configure_manual_swap_variants,
    prompt_policy_baseline_deck,
    write_candidates_manifest,
)
from fab_tui.results import (
    CompletedTrainingEntry,
    EvaluableResultsEntry,
    discover_completed_training_runs,
    discover_evaluable_results,
    list_sideboard_candidate_ids,
)
from fab_tui.runner import (
    fetch_fabrary_deck,
    normalize_card_db_for_talishar,
    run_card_db_rescan,
    run_eval_dashboard,
    run_live_talishar_play,
    run_matchup_simulation,
    run_sideboard_compare,
    run_unified_random_matchups,
)
from fab_tui.card_search import CARDS_DB_PATH, clear_card_db_caches

console = Console()

PLAYER_DECK_CHOICES_BASE = {
    "1": ("precon", "Talishar SAGE precon"),
    "2": ("fabrary", "FaBrary URL or slug"),
}


def _player_deck_choices() -> dict[str, tuple[str, str]]:
    saved_count = len(list_saved_user_decks())
    saved_label = (
        f"Saved sideboard lists ({saved_count})"
        if saved_count
        else "Saved sideboard lists"
    )
    return {
        **PLAYER_DECK_CHOICES_BASE,
        "3": ("saved", saved_label),
    }

EVAL_MODE_CHOICES = {
    "1": ("eval", "Run evaluation (win-rate episodes + GIF replay)"),
    "2": ("render", "Render optimal policy only (GIF replay, no eval episodes)"),
}

UNIFIED_FORMAT_CHOICES = {
    "1": ("silver_age", "Silver Age"),
    "2": ("classic_constructed", "Classic Constructed (CC)"),
    "3": ("blitz", "Blitz"),
    "4": ("upf", "Ultimate Pit Fight"),
}


def _header(title: str, subtitle: str = "") -> None:
    console.clear()
    body = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        body += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(body, border_style="cyan", box=box.ROUNDED))


def _pause() -> None:
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def _live_play_deck_labels(
    results_dir: str,
    candidate_id: str | None,
) -> tuple[str, str]:
    """Resolve trained/opponent deck display names for the live-play wizard."""
    repo = Path(__file__).resolve().parents[1]
    scripts_dir = repo / "scripts"
    for path in (repo / "src", scripts_dir):
        token = str(path)
        if token not in sys.path:
            sys.path.insert(0, token)
    import _bootstrap  # noqa: WPS433

    _bootstrap.configure_paths()
    eval_dir = str(scripts_dir / "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from talishar_live_play import (  # noqa: WPS433
        deck_labels_from_bundle,
        resolve_checkpoint_bundles,
    )

    try:
        p1_bundle, p2_bundle = resolve_checkpoint_bundles(
            Path(results_dir).expanduser().resolve(),
            candidate_id=candidate_id,
        )
        return deck_labels_from_bundle(p1_bundle, p2_bundle)
    except Exception:
        return "Trained deck", "Opponent deck"


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


def _pick_saved_deck() -> Path | None:
    """Pick a previously saved sideboard list."""
    saved = list_saved_user_decks()
    if not saved:
        console.print("[yellow]No saved lists yet — refine a deck in Sideboard comparison first.[/yellow]")
        return None

    table = Table(title="Saved sideboard lists", box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("List")
    table.add_column("Hero", style="dim")
    table.add_column("Saved", style="dim")
    for index, entry in enumerate(saved, start=1):
        saved_display = entry.saved_at[:10] if entry.saved_at else "—"
        table.add_row(str(index), entry.label, entry.hero_id or "—", saved_display)
    console.print(table)

    choice = IntPrompt.ask("Select saved list", default=1)
    if choice < 1 or choice > len(saved):
        return None
    path = saved[choice - 1].path
    console.print(f"[green]Using saved list → {path}[/green]")
    return path


def _pick_player_deck(env: EnvironmentSettings) -> Path | None:
    """Your deck — SAGE precon, saved list, or FaBrary link."""
    source = _choose_mapping(_player_deck_choices(), "Your deck")
    if source == "precon":
        return _pick_precon_deck(env, label="Your deck")
    if source == "saved":
        return _pick_saved_deck()

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
        "Guide policy baseline → refine equipment → refine deck → manual swap variants → play training",
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
    spec.max_parallel = IntPrompt.ask(
        "Train in parallel (0 = all candidates)",
        default=spec.max_parallel,
    )
    spec.play_episodes = IntPrompt.ask("Play episodes per list", default=spec.play_episodes)
    spec.final_eval_episodes = IntPrompt.ask(
        "Final eval games per list",
        default=spec.final_eval_episodes,
    )

    baseline_choice = prompt_policy_baseline_deck(
        console,
        player_deck,
        opponent_hero_id=opponent_hero_id,
        hero_id=spec.hero_id,
        hero_class=spec.hero_class,
        game_format=game_format,
        saved_list=is_saved_user_deck(player_deck),
    )

    if Confirm.ask("Save this list for future runs?", default=True):
        default_label = default_saved_deck_label(
            hero_id=spec.hero_id,
            opponent_hero_id=opponent_hero_id,
            baseline_label=baseline_choice.baseline_label,
        )
        save_label = Prompt.ask("Saved list name", default=default_label).strip()
        if save_label:
            saved_path = save_user_deck(
                baseline_deck=baseline_choice.baseline_deck,
                card_pool=baseline_choice.card_pool,
                equipment_header=baseline_choice.equipment_header,
                hero_id=spec.hero_id,
                hero_class=spec.hero_class,
                game_format=game_format,
                label=save_label,
                opponent_hero_id=opponent_hero_id,
                baseline_label=baseline_choice.baseline_label,
            )
            console.print(f"[green]Saved for reuse → {saved_path}[/green]")

    variants, expanded_pool = configure_manual_swap_variants(
        console,
        player_deck,
        game_format=game_format,
        max_variants=spec.max_swap_variants,
        max_swaps_per_variant=spec.max_swaps_per_variant,
        baseline_deck=baseline_choice.baseline_deck,
        card_pool=baseline_choice.card_pool,
    )

    out_dir = spec.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = write_candidates_manifest(
        out_dir / "candidates_manifest.json",
        baseline_deck=baseline_choice.baseline_deck,
        card_pool=expanded_pool,
        variants=variants,
        baseline_label=baseline_choice.baseline_label,
        equipment_header=baseline_choice.equipment_header,
    )
    if baseline_choice.equipment_header:
        spec.equipment_header = baseline_choice.equipment_header
    spec.candidates_json = str(candidates_path)
    spec.num_options = 1 + len(variants)
    spec.build_cpp_engine = Confirm.ask("Build/use C++ engine if available?", default=True)

    table = Table(title="Review", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    for key, value in [
        ("Your deck", str(player_deck)),
        ("Opponent", f"{opponent_hero_id} ({opponent_deck})"),
        ("Baseline", baseline_choice.baseline_label),
        ("Equipment", baseline_choice.equipment_header or spec.equipment_header or "—"),
        ("Lists to compare", str(spec.num_options)),
        ("  baseline + alternates", f"1 + {len(variants)} manual"),
        ("Parallel", "all" if spec.max_parallel <= 0 else str(spec.max_parallel)),
        ("Play episodes", str(spec.play_episodes)),
        ("Final eval games", str(spec.final_eval_episodes)),
        ("Output", str(out_dir)),
    ]:
        table.add_row(key, value)
    console.print(table)

    if not variants:
        console.print(
            "[yellow]No alternate lists configured — only the baseline deck will be tested.[/yellow]"
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


def wizard_unified_random_matchups(env: EnvironmentSettings) -> None:
    _header(
        "Unified random matchups",
        "Train the shared unified agent on random fabrary deck pairs",
    )

    console.print("\n[bold]Format[/bold]")
    for key, (_, label) in UNIFIED_FORMAT_CHOICES.items():
        console.print(f"  [{key}] {label}")
    fmt_choice = Prompt.ask(
        "Select format",
        choices=list(UNIFIED_FORMAT_CHOICES.keys()),
        default="1",
    )
    game_format = UNIFIED_FORMAT_CHOICES[fmt_choice][0]  # type: ignore[assignment]

    spec = UnifiedRandomMatchupSpec(game_format=game_format)  # type: ignore[arg-type]
    spec.matchups = IntPrompt.ask("Random matchups this run", default=spec.matchups)
    spec.episodes = IntPrompt.ask("Episodes per matchup", default=spec.episodes)
    spec.max_steps = IntPrompt.ask("Max steps per episode", default=spec.max_steps)
    spec.warmup_episodes = IntPrompt.ask("Warmup episodes", default=spec.warmup_episodes)
    spec.checkpoint_interval_pct = IntPrompt.ask(
        "Checkpoint interval (% of episodes)",
        default=int(spec.checkpoint_interval_pct),
    )
    spec.checkpoint_eval_episodes = IntPrompt.ask(
        "Eval games per checkpoint",
        default=spec.checkpoint_eval_episodes,
    )
    spec.workers = IntPrompt.ask("Parallel workers", default=spec.workers)
    spec.skip_converged = Confirm.ask(
        "Skip already-converged deck pairs in cache?",
        default=spec.skip_converged,
    )
    spec.build_cpp_engine = Confirm.ask(
        "Build/use C++ engine if needed?",
        default=spec.build_cpp_engine,
    )
    if Confirm.ask("Set a random seed?", default=False):
        spec.seed = IntPrompt.ask("Seed", default=0)

    out_dir = spec.resolved_out_dir()
    table = Table(title="Review", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    for key, value in [
        ("Format", spec.game_format),
        ("Matchups", str(spec.matchups)),
        ("Episodes / matchup", str(spec.episodes)),
        ("Max steps", str(spec.max_steps)),
        ("Warmup episodes", str(spec.warmup_episodes)),
        ("Checkpoint interval", f"{spec.checkpoint_interval_pct:g}%"),
        ("Checkpoint eval games", str(spec.checkpoint_eval_episodes)),
        ("Workers", str(spec.workers)),
        ("C++ engine", "build if needed" if spec.build_cpp_engine else "HTTP fallback"),
        ("Skip converged", str(spec.skip_converged)),
        ("Seed", str(spec.seed) if spec.seed is not None else "random"),
        ("Output", str(out_dir)),
    ]:
        table.add_row(key, value)
    console.print(table)

    if not Confirm.ask("\nStart unified random matchup training?", default=True):
        return

    console.print("\n[bold]Running unified random matchup training…[/bold]\n")
    rc = run_unified_random_matchups(spec, env)
    _header("Unified random matchups finished", f"Exit code {rc}")
    summary_path = out_dir / "training_summary.json"
    if summary_path.is_file():
        console.print(f"[dim]Summary: {summary_path}[/dim]")
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


def _choose_results_entry(
    entries: list[EvaluableResultsEntry] | list[CompletedTrainingEntry],
    *,
    title: str,
    manual_hint: str,
    status_column: bool = False,
) -> str:
    if not entries:
        console.print("[yellow]No matching runs found under results/.[/yellow]")
        return Prompt.ask(manual_hint, default=str(RESULTS_ROOT))

    table = Table(title=title, box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Category")
    table.add_column("Matchup")
    table.add_column("Run started")
    if status_column:
        table.add_column("Status")
    else:
        table.add_column("Checkpoints")
    table.add_column("Folder")
    for index, entry in enumerate(entries, start=1):
        row = [
            str(index),
            entry.category,
            entry.label,
            entry.run_started,
        ]
        if status_column and isinstance(entry, CompletedTrainingEntry):
            row.append(entry.summary)
        else:
            row.append(entry.checkpoints_summary)
        row.append(entry.path.name)
        table.add_row(*row)
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


def _choose_results_dir(
    *,
    title: str = "Select results",
    manual_hint: str = "Enter results directory path",
) -> str:
    entries = discover_evaluable_results()
    return _choose_results_entry(
        entries,
        title=title,
        manual_hint=manual_hint,
    )


def _choose_completed_training_dir(
    *,
    title: str = "Completed training runs",
    manual_hint: str = "Enter results directory path",
) -> str:
    entries = discover_completed_training_runs()
    return _choose_results_entry(
        entries,
        title=title,
        manual_hint=manual_hint,
        status_column=True,
    )


def _pick_sideboard_candidate_id(results_dir: str) -> str | None:
    sideboard_candidates = list_sideboard_candidate_ids(Path(results_dir))
    if not sideboard_candidates:
        return None

    console.print(
        "\n[dim]Sideboard compare run — pick the trained candidate to evaluate, "
        "or leave blank for the latest checkpoint across all candidates.[/dim]"
    )
    table = Table(title="Candidates", box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Candidate ID")
    for index, cid in enumerate(sideboard_candidates, start=1):
        table.add_row(str(index), cid)
    console.print(table)
    choice = Prompt.ask(
        "Candidate (number, id, or blank for latest)",
        default="",
    ).strip()
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(sideboard_candidates):
            return sideboard_candidates[idx - 1]
    if choice:
        return choice
    return None


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
    candidate_id = _pick_sideboard_candidate_id(results_dir)

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


def wizard_evaluate_trained_agent(env: EnvironmentSettings) -> None:
    _header(
        "Evaluate trained agent",
        "Run more Talishar evaluation games on a finished training run",
    )
    results_dir = _choose_completed_training_dir(
        title="Completed training runs",
        manual_hint="Results directory",
    )
    candidate_id = _pick_sideboard_candidate_id(results_dir)

    default_episodes = (
        RUNTIME.play.checkpoint_eval_episodes
        or RUNTIME.meta.final_eval_episodes
        or RUNTIME.meta.eval_episodes
    )
    default_workers = RUNTIME.meta.eval_parallel_workers or RUNTIME.play.workers or 4
    default_max_steps = RUNTIME.meta.eval_max_steps or RUNTIME.meta.max_play_steps

    spec = EvalSpec(
        results_dir=results_dir,
        candidate_id=candidate_id,
        episodes=IntPrompt.ask("Evaluation episodes on Talishar", default=default_episodes),
        parallel_workers=IntPrompt.ask("Parallel workers", default=default_workers),
        max_steps=IntPrompt.ask("Max steps per game", default=default_max_steps),
        watch=False,
    )
    console.print("\n[bold]Running Talishar evaluation on latest checkpoint…[/bold]\n")
    rc = run_eval_dashboard(spec, env)
    console.print(f"\n[bold]Evaluation finished (exit {rc}).[/bold]")
    _pause()


def wizard_realtime_talishar_play(env: EnvironmentSettings) -> None:
    _header(
        "Real-time Talishar play",
        "Watch the agent or play against it on the live Talishar frontend",
    )
    results_dir = _choose_completed_training_dir(
        title="Completed training runs",
        manual_hint="Results directory",
    )
    candidate_id = _pick_sideboard_candidate_id(results_dir)

    human_vs_agent = Confirm.ask(
        "Play against the agent? (You play on the board; the agent acts automatically)",
        default=False,
    )

    human_deck = "opponent"
    if human_vs_agent:
        trained_label, opponent_label = _live_play_deck_labels(
            results_dir,
            candidate_id,
        )
        console.print(
            f"\n  [1] Trained deck — [cyan]{trained_label}[/cyan]"
        )
        console.print(
            f"  [2] Opponent deck — [cyan]{opponent_label}[/cyan]"
        )
        choice = Prompt.ask(
            "Which deck do you want to play?",
            choices=["1", "2"],
            default="2",
        )
        human_deck = "trained" if choice == "1" else "opponent"

    default_max_steps = RUNTIME.meta.eval_max_steps or RUNTIME.meta.max_play_steps
    spec = LivePlaySpec(
        results_dir=results_dir,
        candidate_id=candidate_id,
        games=IntPrompt.ask("Number of games", default=1),
        max_steps=IntPrompt.ask("Max steps per game", default=default_max_steps),
        step_delay_ms=IntPrompt.ask(
            "Step delay (ms, 0 = fastest)",
            default=0,
        ),
        human_vs_agent=human_vs_agent,
        human_deck=human_deck,
        enable_action_coach=Confirm.ask(
            "Show agent coach overlay on the board?",
            default=True,
        ) if human_vs_agent else True,
    )
    if human_vs_agent and spec.enable_action_coach:
        spec.coach_rollouts_per_action = IntPrompt.ask(
            "C++ rollout games per action (win % estimate)",
            default=spec.coach_rollouts_per_action,
        )
    if human_vs_agent:
        trained_label, opponent_label = _live_play_deck_labels(
            results_dir,
            candidate_id,
        )
        your_deck = trained_label if spec.human_deck == "trained" else opponent_label
        agent_deck = opponent_label if spec.human_deck == "trained" else trained_label
        console.print(
            "\n[bold]Starting human vs agent session…[/bold]\n"
            f"[dim]You play [cyan]{your_deck}[/cyan] — "
            "click your actions in the Talishar window.[/dim]\n"
            f"[dim]The trained agent plays [cyan]{agent_deck}[/cyan] automatically.[/dim]\n"
            + (
                "[dim]Agent coach overlay shows policy % and C++ win estimates on your turns.[/dim]\n"
                if spec.enable_action_coach
                else ""
            )
            + f"[dim]Frontend: {env.talishar_fe_url}[/dim]\n"
        )
    else:
        console.print(
            "\n[bold]Starting live Talishar session…[/bold]\n"
            "[dim]A Chromium window will open with the Talishar board (GDPR consent auto-handled).[/dim]\n"
            f"[dim]Frontend: {env.talishar_fe_url}[/dim]\n"
        )
    rc = run_live_talishar_play(spec, env)
    console.print(f"\n[bold]Live play finished (exit {rc}).[/bold]")
    _pause()


def wizard_settings(env: EnvironmentSettings) -> EnvironmentSettings:
    _header("Settings", "Talishar, FaBrary, and card database maintenance")

    env.talishar_url = Prompt.ask("Talishar URL", default=env.talishar_url)
    env.talishar_fe_url = Prompt.ask("Talishar frontend URL", default=env.talishar_fe_url)
    env.assets_path = Prompt.ask("Talishar Assets path", default=env.assets_path)
    env.fabrary_api_key = Prompt.ask(
        "FaBrary API key (optional)",
        default=env.fabrary_api_key,
        password=True,
    )
    env.apply_to_environ()
    console.print("[green]Connection settings saved for this session.[/green]")

    card_count = "?"
    if CARDS_DB_PATH.is_file():
        try:
            card_count = str(len(json.loads(CARDS_DB_PATH.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError):
            pass
    console.print(f"\n[dim]Local card DB: {CARDS_DB_PATH} ({card_count} cards)[/dim]")
    console.print(
        "[dim]Rescan downloads missing deck cards and refreshes metadata/legality "
        "from the official FAB Card Vault API, then normalizes card IDs for Talishar.[/dim]"
    )

    if Confirm.ask("Run full card database rescan now?", default=False):
        console.print(
            "\n[bold]Rescanning card database…[/bold]\n"
            "[dim]This may take several minutes and requires internet access.[/dim]\n"
        )
        try:
            rc = run_card_db_rescan(legality_scope="all")
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            _pause()
            return env
        if rc == 0:
            clear_card_db_caches()
            console.print(
                "[green]Card database rescan and Talishar ID normalization complete.[/green]"
            )
        else:
            console.print(f"[red]Card database rescan failed (exit {rc}).[/red]")
    elif Confirm.ask(
        "Normalize local card IDs for Talishar only? (fixes blank/missing cards in play)",
        default=True,
    ):
        try:
            norm_rc, summary = normalize_card_db_for_talishar()
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            _pause()
            return env
        if norm_rc == 0:
            clear_card_db_caches()
            console.print(
                "[green]Card ID normalization complete "
                f"({summary.get('remapped', 0)} remapped).[/green]"
            )
        else:
            console.print(f"[red]Card ID normalization failed (exit {norm_rc}).[/red]")

    _pause()
    return env


def run_tui() -> int:
    env = EnvironmentSettings()
    env.apply_to_environ()

    menu_actions = {
        "1": ("Sideboard comparison", wizard_sideboard_compare),
        "2": ("Fixed deck simulation", wizard_simulate_decks),
        "3": ("Evaluate checkpoints", wizard_evaluate),
        "4": ("Evaluate trained agent", wizard_evaluate_trained_agent),
        "5": ("Real-time Talishar play", wizard_realtime_talishar_play),
        "6": ("Train unified agent with random matchups", wizard_unified_random_matchups),
        "7": ("Settings", wizard_settings),
        "q": ("Quit", None),
    }

    while True:
        _header(
            "Flesh and Blood RL Bridge",
            "Sideboard tuning, eval, live Talishar play, and settings",
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
