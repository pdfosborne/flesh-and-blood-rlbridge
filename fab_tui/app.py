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
    ExperimentSpec,
    MatchupSimSpec,
    slugify,
)
from fab_tui.decks import (
    DECK_CACHE,
    apply_heroes_from_decks,
    apply_mirror_opponent_from_p1,
    apply_opponent_from_asset,
    apply_player_from_deck,
    discover_saved_decks,
    export_precon_deck_json,
    list_precon_options,
    read_deck_format,
    read_deck_hero_info,
)
from fab_tui.presets import PRESETS
from fab_tui.runner import (
    build_cpp_engine_for_spec,
    cpp_build_inputs_for_spec,
    discover_cpp_engine_dir,
    fetch_fabrary_deck,
    run_eval_dashboard,
    run_full_pipeline,
    run_matchup_simulation,
    run_runscript,
)

console = Console()

FORMAT_CHOICES = {
    "1": ("silver_age", "Silver Age (40-card game decks)"),
    "2": ("classic_constructed", "Classic Constructed (60-card)"),
    "3": ("blitz", "Blitz"),
    "4": ("upf", "Ultimate Pit Fight"),
}

OPPONENT_MODE_CHOICES = {
    "1": ("dual", "Dual — both players draft and train"),
    "2": ("preset", "Preset — fixed Talishar opponent deck"),
    "3": ("mirror", "Mirror — same built deck both sides"),
}

WORKFLOW_CHOICES = {
    "1": ("full", "Full pipeline — draft, sideboard, play, final eval"),
    "2": ("draft_only", "Draft only — deckbuilder + sideboard (no play training)"),
    "3": ("play_only", "Play only — fixed decks, train play agents + eval"),
}

DECK_SOURCE_CHOICES = {
    "1": ("precon", "Talishar SAGE precon"),
    "2": ("fabrary", "FaBrary URL or slug"),
    "3": ("saved", "Previously drafted / saved deck"),
    "4": ("local", "Local JSON file path"),
}

SIM_FORMAT_CHOICES = {
    **FORMAT_CHOICES,
    "5": ("sage", "SAGE (precon format)"),
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


def _maybe_fetch_deck(label: str, deck_dir: Path) -> str | None:
    if not Confirm.ask(f"Provide a FaBrary deck for {label}?", default=False):
        return None
    source = Prompt.ask(
        f"{label} deck (URL, slug, or local JSON path)",
    ).strip()
    if not source:
        return None
    local = Path(source)
    if local.is_file():
        return str(local.resolve())
    slug = source.rsplit("/", 1)[-1][:26]
    out = deck_dir / f"{slugify(label)}_{slug.lower()}.json"
    console.print(f"[yellow]Fetching deck → {out}[/yellow]")
    rc = fetch_fabrary_deck(source, out)
    if rc != 0:
        console.print("[red]Fetch failed.[/red]")
        return None
    return str(out)


def _pick_deck_source(player_label: str, env: EnvironmentSettings) -> Path | None:
    """Interactive deck picker — precon, FaBrary, saved drafts, or local JSON."""
    source = _choose_mapping(DECK_SOURCE_CHOICES, f"{player_label} deck source")
    assets = Path(env.assets_path)
    cache_dir = DECK_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    if source == "precon":
        options = list_precon_options(assets)
        if not options:
            console.print("[red]No precon deck files found in Assets.[/red]")
            return None
        table = Table(title="Precon decks", box=box.SIMPLE)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Deck")
        for index, (label, _) in enumerate(options, start=1):
            table.add_row(str(index), label)
        console.print(table)
        choice = IntPrompt.ask("Select precon", default=1)
        if choice < 1 or choice > len(options):
            return None
        _, deck_name = options[choice - 1]
        out = cache_dir / f"precon_{slugify(deck_name)}.json"
        export_precon_deck_json(deck_name, assets, out, game_format="sage")
        console.print(f"[green]Exported precon → {out}[/green]")
        return out

    if source == "fabrary":
        raw = Prompt.ask(f"{player_label} FaBrary URL or slug").strip()
        if not raw:
            return None
        local = Path(raw)
        if local.is_file():
            return local.resolve()
        slug = raw.rsplit("/", 1)[-1][:26]
        out = cache_dir / f"{slugify(player_label)}_{slug.lower()}.json"
        console.print(f"[yellow]Fetching deck → {out}[/yellow]")
        if fetch_fabrary_deck(raw, out) != 0:
            console.print("[red]Fetch failed.[/red]")
            return None
        return out

    if source == "saved":
        saved = discover_saved_decks()
        if not saved:
            console.print("[yellow]No saved decks found under results/.[/yellow]")
            return None
        table = Table(title="Saved decks", box=box.SIMPLE)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Deck")
        for index, option in enumerate(saved, start=1):
            table.add_row(str(index), option.label)
        console.print(table)
        choice = IntPrompt.ask("Select deck", default=1)
        if choice < 1 or choice > len(saved):
            return None
        return saved[choice - 1].path

    path_str = Prompt.ask(f"{player_label} deck JSON path").strip()
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_file():
        console.print(f"[red]File not found: {path}[/red]")
        return None
    return path.resolve()


def _show_matchup_summary(
    spec: MatchupSimSpec,
    env: EnvironmentSettings,
    deck1: Path,
    deck2: Path,
) -> None:
    table = Table(title="Matchup simulation", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    for key, value in [
        ("P1 deck", str(deck1)),
        ("P2 deck", str(deck2)),
        ("Format", spec.game_format),
        ("Play episodes", str(spec.play_episodes)),
        ("Sideboard episodes", str(spec.sideboard_episodes)),
        ("Final eval games", str(spec.final_eval_episodes)),
        ("Iterations", str(spec.iterations)),
        ("Build C++ engine", "yes" if spec.build_cpp_engine else "no"),
        ("Talishar URL", env.talishar_url),
    ]:
        table.add_row(key, value)
    console.print(table)


def _configure_opponent(spec: ExperimentSpec, env: EnvironmentSettings) -> None:
    """Configure fixed opponent deck / hero for preset and mirror modes."""
    if spec.opponent_mode == "dual":
        console.print(
            "[dim]Dual mode: draft eval uses each player's built deck as the opponent.[/dim]"
        )
        return

    if spec.opponent_mode == "mirror":
        console.print(
            "[dim]Mirror mode: both sides use your built deck — "
            "opponent hero is copied from P1 after deck selection.[/dim]"
        )
        return

    if spec.p2_fixed_deck:
        apply_player_from_deck(spec, spec.p2_fixed_deck, player="p2")
        info = read_deck_hero_info(Path(spec.p2_fixed_deck))
        if info and info.hero_id:
            spec.opponent_hero_id = info.hero_id
            console.print(
                f"[green]Opponent from pinned deck[/green] ({info.name}): "
                f"{spec.p2_hero_id} ({spec.p2_hero_class})"
            )
            console.print(f"[dim]  equipment: {spec.p2_equipment_header}[/dim]")
        else:
            console.print(
                "[green]Opponent deck pinned[/green] — skipping Talishar preset picker."
            )
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from flesh_and_blood_rlbridge.opponent_deck import list_preset_opponent_options

    options = list_preset_opponent_options(env.assets_path)
    if not options:
        spec.opponent_deck = Prompt.ask(
            "Opponent Talishar deck (Assets stem)",
            default=spec.opponent_deck,
        )
    else:
        table = Table(title="Preset opponent deck", box=box.SIMPLE)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Deck")
        for index, (label, _) in enumerate(options, start=1):
            table.add_row(str(index), label)
        table.add_row("c", "Custom Assets name…")
        console.print(table)
        choice = Prompt.ask("Select opponent", default="1")
        if choice.strip().lower() == "c":
            spec.opponent_deck = Prompt.ask(
                "Opponent Talishar deck (Assets stem)",
                default=spec.opponent_deck,
            )
        elif choice.strip().isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                spec.opponent_deck = options[idx - 1][1]
        else:
            spec.opponent_deck = options[0][1]

    if apply_opponent_from_asset(spec, env.assets_path):
        console.print(
            f"[green]Opponent from Assets[/green] ({spec.opponent_deck}): "
            f"{spec.opponent_hero_id} ({spec.p2_hero_class})"
        )
        console.print(f"[dim]  equipment: {spec.p2_equipment_header}[/dim]")
        if not Confirm.ask("Override opponent hero settings?", default=False):
            return

    default_hero = spec.opponent_hero_id or spec.p2_hero_id
    spec.opponent_hero_id = Prompt.ask(
        "Opponent hero id (sideboard target)",
        default=default_hero,
    )
    spec.p2_hero_id = spec.opponent_hero_id
    spec.p2_hero_class = Prompt.ask("Opponent hero class", default=spec.p2_hero_class)
    spec.p2_equipment_header = Prompt.ask(
        "Opponent equipment header",
        default=spec.p2_equipment_header,
    )


def _configure_heroes(spec: ExperimentSpec) -> None:
    apply_heroes_from_decks(spec)

    p1_deck = spec.p1_fixed_deck or spec.p1_starting_deck
    p2_deck = spec.p2_fixed_deck or spec.p2_starting_deck

    if p1_deck:
        info = read_deck_hero_info(Path(p1_deck))
        if info and info.hero_id:
            console.print(
                f"[green]P1 from deck[/green] ({info.name}): "
                f"{spec.hero_id} ({spec.hero_class})"
            )
            console.print(f"[dim]  equipment: {spec.equipment_header}[/dim]")
            edit_p1 = Confirm.ask("Override P1 hero settings?", default=False)
        else:
            edit_p1 = True
    else:
        edit_p1 = True

    if edit_p1:
        spec.hero_id = Prompt.ask("P1 hero id", default=spec.hero_id)
        spec.hero_class = Prompt.ask("P1 hero class", default=spec.hero_class)
        spec.equipment_header = Prompt.ask(
            "P1 equipment header", default=spec.equipment_header
        )

    if spec.opponent_mode == "mirror":
        apply_mirror_opponent_from_p1(spec)
        console.print(
            f"[green]Mirror opponent[/green] (same as P1): "
            f"{spec.opponent_hero_id} ({spec.p2_hero_class})"
        )
        console.print(f"[dim]  equipment: {spec.p2_equipment_header}[/dim]")
        return

    if spec.opponent_mode == "dual":
        if p2_deck:
            info = read_deck_hero_info(Path(p2_deck))
            if info and info.hero_id:
                console.print(
                    f"[green]P2 from deck[/green] ({info.name}): "
                    f"{spec.p2_hero_id} ({spec.p2_hero_class})"
                )
                console.print(f"[dim]  equipment: {spec.p2_equipment_header}[/dim]")
                edit_p2 = Confirm.ask("Override P2 hero settings?", default=False)
            else:
                edit_p2 = True
        else:
            edit_p2 = True

        if edit_p2:
            spec.p2_hero_id = Prompt.ask("P2 hero id", default=spec.p2_hero_id)
            spec.p2_hero_class = Prompt.ask("P2 hero class", default=spec.p2_hero_class)
            spec.p2_equipment_header = Prompt.ask(
                "P2 equipment header", default=spec.p2_equipment_header
            )
    elif spec.opponent_mode == "preset":
        if p2_deck and spec.p2_hero_id:
            console.print(
                f"[green]Opponent hero from deck:[/green] {spec.p2_hero_id}"
            )
            spec.opponent_hero_id = spec.opponent_hero_id or spec.p2_hero_id
        if not Confirm.ask("Override opponent hero id?", default=False):
            return
        spec.opponent_hero_id = Prompt.ask(
            "Opponent hero id (sideboard target)",
            default=spec.opponent_hero_id or spec.p2_hero_id,
        )


def _configure_volumes(spec: ExperimentSpec) -> None:
    if spec.workflow != "play_only":
        spec.deckbuild_episodes = IntPrompt.ask(
            "Deckbuilder episodes / iteration",
            default=spec.deckbuild_episodes,
        )
        spec.sideboard_episodes = IntPrompt.ask(
            "Sideboard episodes / opponent / iteration",
            default=spec.sideboard_episodes,
        )
        spec.num_eval_games = IntPrompt.ask(
            "Eval games inside deckbuilder finalize",
            default=spec.num_eval_games,
        )
    if spec.workflow != "draft_only":
        spec.play_episodes = IntPrompt.ask(
            "Play training episodes / iteration",
            default=spec.play_episodes,
        )
        spec.final_eval_episodes = IntPrompt.ask(
            "Final evaluation games (post-training)",
            default=spec.final_eval_episodes,
        )
        spec.final_eval_max_steps = IntPrompt.ask(
            "Max steps per final eval game",
            default=spec.final_eval_max_steps,
        )
    spec.iterations = IntPrompt.ask("Outer iterations", default=spec.iterations)


def _show_spec_summary(spec: ExperimentSpec, env: EnvironmentSettings) -> None:
    table = Table(title="Experiment summary", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    rows = [
        ("Name", spec.name),
        ("Workflow", spec.workflow),
        ("Format", spec.game_format),
        ("Opponent mode", spec.opponent_mode),
        ("P1 hero", f"{spec.hero_id} ({spec.hero_class})"),
        ("P2 / opponent", spec.p2_hero_id if spec.opponent_mode == "dual" else spec.opponent_deck),
        ("Deckbuild eps", str(spec.deckbuild_episodes)),
        ("Sideboard eps", str(spec.sideboard_episodes)),
        ("Play eps", str(spec.play_episodes)),
        ("Iterations", str(spec.iterations)),
        ("Final eval games", str(spec.final_eval_episodes)),
        ("Output", str(spec.resolved_out_dir())),
        ("Talishar URL", env.talishar_url),
        ("Build C++ engine", "yes" if spec.build_cpp_engine else "no"),
    ]
    if spec.p1_starting_deck:
        rows.append(("P1 warm-start deck", spec.p1_starting_deck))
    if spec.p2_starting_deck:
        rows.append(("P2 warm-start deck", spec.p2_starting_deck))
    if spec.p1_fixed_deck:
        rows.append(("P1 fixed deck", spec.p1_fixed_deck))
    if spec.p2_fixed_deck:
        rows.append(("P2 fixed deck", spec.p2_fixed_deck))
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


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


def wizard_new_experiment(env: EnvironmentSettings) -> None:
    _header("New experiment", "Configure deck drafting, sideboarding, play training, and evaluation")

    spec = ExperimentSpec(name=Prompt.ask("Experiment name", default="my_experiment"))
    spec.workflow = _choose_mapping(WORKFLOW_CHOICES, "Workflow")
    spec.apply_workflow_defaults()
    spec.game_format = _choose_mapping(FORMAT_CHOICES, "Format")  # type: ignore[assignment]
    spec.opponent_mode = _choose_mapping(OPPONENT_MODE_CHOICES, "Opponent mode")  # type: ignore[assignment]

    deck_dir = spec.resolved_out_dir() / "decks"
    if spec.workflow != "play_only":
        warm1 = _maybe_fetch_deck("P1 warm-start", deck_dir)
        if warm1:
            spec.p1_starting_deck = warm1
        if spec.opponent_mode == "dual":
            warm2 = _maybe_fetch_deck("P2 warm-start", deck_dir)
            if warm2:
                spec.p2_starting_deck = warm2
    else:
        fixed1 = _maybe_fetch_deck("P1 fixed game deck", deck_dir)
        if spec.opponent_mode != "mirror":
            fixed2 = _maybe_fetch_deck("P2 fixed game deck", deck_dir)
            if fixed2:
                spec.p2_fixed_deck = fixed2
        if fixed1:
            spec.p1_fixed_deck = fixed1

    if spec.opponent_mode != "mirror" and Confirm.ask(
        "Pin P2 to a fixed opponent deck (skip P2 draft)?", default=False
    ):
        fixed = _maybe_fetch_deck("P2 fixed opponent", deck_dir)
        if fixed:
            spec.p2_fixed_deck = fixed
            apply_player_from_deck(spec, fixed, player="p2")

    _configure_opponent(spec, env)
    _configure_heroes(spec)
    _configure_volumes(spec)
    spec.build_cpp_engine = Confirm.ask("Build/use C++ engine if available?", default=True)

    _header("Review", "Confirm settings before launching")
    _show_spec_summary(spec, env)
    if not Confirm.ask("\nStart experiment?", default=True):
        return

    cpp_dir: str | None = None
    if spec.build_cpp_engine:
        deck1, deck2, _, _ = cpp_build_inputs_for_spec(spec, env)
        existing = discover_cpp_engine_dir(
            deck1, deck2, assets_path=env.assets_path
        )
        if existing is not None:
            console.print(f"[green]Using C++ engine: {existing}[/green]")
            cpp_dir = str(existing)
        elif Confirm.ask("No cached engine found. Build now?", default=True):
            rc = build_cpp_engine_for_spec(spec, env)
            if rc == 0:
                found = discover_cpp_engine_dir(
                    deck1, deck2, assets_path=env.assets_path
                )
                cpp_dir = str(found) if found else None
            else:
                console.print("[yellow]C++ build failed — continuing with HTTP Talishar.[/yellow]")

    console.print("\n[bold]Running training pipeline...[/bold]\n")
    rc = run_full_pipeline(spec, env, cpp_engine_dir=cpp_dir)
    results_path = Path(spec.results_json or spec.resolved_out_dir() / "results.json")

    _header("Experiment finished", f"Exit code {rc}")
    _show_results_summary(results_path)

    if spec.workflow != "draft_only" and Confirm.ask("Open live eval dashboard on this run?", default=False):
        eval_spec = EvalSpec(results_dir=str(spec.resolved_out_dir()), watch=True)
        run_eval_dashboard(eval_spec, env)
    _pause()


def wizard_simulate_decks(env: EnvironmentSettings) -> None:
    _header(
        "Simulate decks",
        "Fixed-deck matchup — precons, FaBrary links, or prior drafts",
    )

    deck1 = _pick_deck_source("P1", env)
    if deck1 is None:
        console.print("[yellow]P1 deck not selected.[/yellow]")
        _pause()
        return
    deck2 = _pick_deck_source("P2", env)
    if deck2 is None:
        console.print("[yellow]P2 deck not selected.[/yellow]")
        _pause()
        return

    fmt1 = read_deck_format(deck1)
    fmt2 = read_deck_format(deck2)
    if fmt1 == fmt2 and fmt1 in {v[0] for v in SIM_FORMAT_CHOICES.values()}:
        game_format = fmt1  # type: ignore[assignment]
        console.print(f"[dim]Using format from decks: {game_format}[/dim]")
    elif fmt1 != fmt2:
        console.print(
            f"[yellow]Deck formats differ ({fmt1} vs {fmt2}); choose simulation format.[/yellow]"
        )
        game_format = _choose_mapping(SIM_FORMAT_CHOICES, "Simulation format")  # type: ignore[assignment]
    else:
        game_format = _choose_mapping(SIM_FORMAT_CHOICES, "Simulation format")  # type: ignore[assignment]

    spec = MatchupSimSpec(
        deck1_source=str(deck1),
        deck2_source=str(deck2),
        game_format=game_format,
    )

    spec.play_episodes = IntPrompt.ask("Play training episodes", default=spec.play_episodes)
    spec.sideboard_episodes = IntPrompt.ask(
        "Sideboard episodes (fixed-deck tuning)",
        default=spec.sideboard_episodes,
    )
    spec.final_eval_episodes = IntPrompt.ask(
        "Final evaluation games",
        default=spec.final_eval_episodes,
    )
    spec.final_eval_max_steps = IntPrompt.ask(
        "Max steps per final eval game",
        default=spec.final_eval_max_steps,
    )
    spec.iterations = IntPrompt.ask("Iterations", default=spec.iterations)
    spec.build_cpp_engine = Confirm.ask("Build/use C++ engine if available?", default=True)

    _header("Review matchup")
    _show_matchup_summary(spec, env, deck1, deck2)
    if not Confirm.ask("Start simulation?", default=True):
        return

    console.print("\n[bold]Running deck matchup simulation...[/bold]\n")
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


def wizard_draft_decks(env: EnvironmentSettings) -> None:
    _header("Draft decks", "Phase 1 deckbuilder + Phase 2 sideboard")
    spec = ExperimentSpec(
        name=Prompt.ask("Draft run name", default="deck_draft"),
        workflow="draft_only",
    )
    spec.game_format = _choose_mapping(FORMAT_CHOICES, "Format")  # type: ignore[assignment]
    spec.opponent_mode = _choose_mapping(OPPONENT_MODE_CHOICES, "Opponent mode")  # type: ignore[assignment]

    deck_dir = spec.resolved_out_dir() / "decks"
    warm1 = _maybe_fetch_deck("P1 warm-start", deck_dir)
    if warm1:
        spec.p1_starting_deck = warm1
    if spec.opponent_mode == "dual":
        warm2 = _maybe_fetch_deck("P2 warm-start", deck_dir)
        if warm2:
            spec.p2_starting_deck = warm2

    _configure_opponent(spec, env)
    _configure_heroes(spec)
    spec.deckbuild_episodes = IntPrompt.ask("Deckbuilder episodes", default=30)
    spec.sideboard_episodes = IntPrompt.ask("Sideboard episodes", default=30)
    spec.iterations = IntPrompt.ask("Iterations", default=1)
    spec.apply_workflow_defaults()

    _header("Review draft run")
    _show_spec_summary(spec, env)
    if not Confirm.ask("Start drafting?", default=True):
        return

    rc = run_full_pipeline(spec, env)
    console.print(f"\n[bold]Draft finished (exit {rc}).[/bold]")
    _show_results_summary(spec.resolved_out_dir() / "results.json")
    _pause()


def wizard_evaluate(env: EnvironmentSettings) -> None:
    _header("Evaluate checkpoints", "Run phase-3 eval dashboard on saved results")

    default_dir = ""
    experiments_root = RESULTS_ROOT / "experiments"
    if experiments_root.is_dir():
        candidates = sorted(experiments_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            default_dir = str(candidates[0])

    results_dir = Prompt.ask("Results directory", default=default_dir or str(RESULTS_ROOT))
    spec = EvalSpec(
        results_dir=results_dir,
        episodes=IntPrompt.ask("Evaluation episodes", default=20),
        parallel_workers=IntPrompt.ask("Parallel workers", default=4),
        max_steps=IntPrompt.ask("Max steps per game", default=1000),
        watch=Confirm.ask("Watch for new checkpoints?", default=True),
    )
    if spec.watch:
        spec.poll_seconds = IntPrompt.ask("Poll interval (seconds)", default=30)

    console.print("\n[bold]Starting evaluation dashboard...[/bold]\n")
    rc = run_eval_dashboard(spec, env)
    console.print(f"\n[bold]Evaluation finished (exit {rc}).[/bold]")
    _pause()


def wizard_presets(env: EnvironmentSettings) -> None:
    _header("Presets", "Launch a curated experiment or runscript")

    table = Table(box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Name")
    table.add_column("Description")
    for index, preset in enumerate(PRESETS, start=1):
        table.add_row(str(index), preset.label, preset.description)
    console.print(table)

    choice = IntPrompt.ask("Select preset", default=1)
    if choice < 1 or choice > len(PRESETS):
        console.print("[red]Invalid selection.[/red]")
        _pause()
        return

    preset = PRESETS[choice - 1]
    env.apply_to_environ()

    if preset.runscript:
        console.print(f"\n[bold]Running {preset.runscript}...[/bold]\n")
        rc = run_runscript(preset.runscript)
        console.print(f"\n[bold]Finished (exit {rc}).[/bold]")
    elif preset.experiment:
        spec = preset.experiment
        spec.apply_workflow_defaults()
        _show_spec_summary(spec, env)
        if Confirm.ask("Start?", default=True):
            rc = run_full_pipeline(spec, env)
            console.print(f"\n[bold]Finished (exit {rc}).[/bold]")
            _show_results_summary(spec.resolved_out_dir() / "results.json")
    _pause()


def wizard_settings(env: EnvironmentSettings) -> EnvironmentSettings:
    _header("Environment settings", "Talishar and FaBrary connection defaults")

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


def wizard_browse_results() -> None:
    _header("Browse results", "Recent experiment output folders")

    roots = [RESULTS_ROOT / "experiments", RESULTS_ROOT / "matchup_sims", RESULTS_ROOT / "full_pipeline"]
    rows: list[tuple[str, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:8]:
            if path.is_dir():
                rows.append((root.name, path))

    if not rows:
        console.print("[yellow]No result directories found yet.[/yellow]")
        _pause()
        return

    table = Table(box=box.SIMPLE)
    table.add_column("#", style="cyan")
    table.add_column("Category")
    table.add_column("Path")
    for index, (category, path) in enumerate(rows, start=1):
        table.add_row(str(index), category, str(path))
    console.print(table)

    choice = Prompt.ask("Open results JSON for entry # (or Enter to skip)", default="")
    if not choice.strip().isdigit():
        _pause()
        return
    idx = int(choice)
    if idx < 1 or idx > len(rows):
        _pause()
        return
    results_json = rows[idx - 1][1] / "results.json"
    _show_results_summary(results_json)
    _pause()


def run_tui() -> int:
    env = EnvironmentSettings()
    env.apply_to_environ()

    menu_actions = {
        "1": ("New experiment (draft → play → eval)", wizard_new_experiment),
        "2": ("Draft decks only (phases 1–2)", wizard_draft_decks),
        "3": ("Simulate decks (fixed matchup)", wizard_simulate_decks),
        "4": ("Evaluate checkpoints", wizard_evaluate),
        "5": ("Presets & runscripts", wizard_presets),
        "6": ("Browse results", lambda _env: wizard_browse_results()),
        "7": ("Settings", wizard_settings),
        "q": ("Quit", None),
    }

    while True:
        _header(
            "Flesh and Blood RL Bridge",
            "Draft decks, train agents, and evaluate matchups",
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
