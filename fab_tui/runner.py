"""Subprocess runners for experiments and evaluation."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fab_tui.config import (
    CARDS_DB_PATH,
    CARDS_DB_UPDATE_SCRIPT,
    FABRARY_DECKS_PATH,
    REPO_ROOT,
    RESULTS_ROOT,
    RUNSCRIPTS_ROOT,
    SCRIPTS_CPP,
    SCRIPTS_DECK,
    SCRIPTS_EVAL,
    SCRIPTS_TRAINING,
    EnvironmentSettings,
    EvalSpec,
    ExperimentSpec,
    LivePlaySpec,
    MatchupSimSpec,
    SideboardCompareSpec,
    UnifiedRandomMatchupSpec,
    normalize_pipeline_format,
)


def _python() -> str:
    return sys.executable


@contextmanager
def _cwd_repo() -> Iterator[None]:
    previous = Path.cwd()
    try:
        import os

        os.chdir(REPO_ROOT)
        yield
    finally:
        import os

        os.chdir(previous)


def run_streaming(cmd: list[str], *, cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> int:
    """Run a command forwarding stdout/stderr to the terminal."""
    import os

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    with subprocess.Popen(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=env,
    ) as proc:
        return int(proc.wait())


def fetch_fabrary_deck(url_or_slug: str, out_file: Path) -> int:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    source = url_or_slug.strip()
    if not source.startswith("http"):
        source = f"https://fabrary.net/decks/{source}"
    return run_streaming(
        [
            _python(),
            str(SCRIPTS_DECK / "fetch_fabrary_deck.py"),
            source,
            "--out",
            str(out_file),
            "--pretty",
        ]
    )


def _resolve_deck_stem(name: str, assets_path: str) -> str:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from flesh_and_blood_rlbridge.talishar_deck_assets import resolve_talishar_deck_stem

    return resolve_talishar_deck_stem(assets_path, name)


def cpp_build_inputs_for_spec(
    spec: ExperimentSpec,
    env: EnvironmentSettings,
) -> tuple[str, str, Path | None, Path | None]:
    """Resolve C++ engine deck stems and optional FaBrary JSON paths."""
    p1_json: Path | None = None
    p2_json: Path | None = None
    if spec.p1_fixed_deck:
        p1_json = Path(spec.p1_fixed_deck)
    elif spec.p1_starting_deck:
        p1_json = Path(spec.p1_starting_deck)
    if spec.p2_fixed_deck:
        p2_json = Path(spec.p2_fixed_deck)
    elif spec.p2_starting_deck:
        p2_json = Path(spec.p2_starting_deck)

    if spec.opponent_mode == "preset":
        deck2_key = spec.opponent_deck
    elif spec.opponent_mode == "mirror":
        deck2_key = spec.hero_id
        p2_json = p1_json
    else:
        deck2_key = spec.p2_hero_id

    deck1 = _resolve_deck_stem(spec.hero_id, env.assets_path)
    deck2 = _resolve_deck_stem(deck2_key, env.assets_path)
    return (
        deck1,
        deck2,
        p1_json if p1_json and p1_json.is_file() else None,
        p2_json if p2_json and p2_json.is_file() else None,
    )


def build_cpp_engine(
    deck1: str,
    deck2: str,
    env: EnvironmentSettings,
    *,
    deck1_json: Path | None = None,
    deck2_json: Path | None = None,
) -> int:
    if str(SCRIPTS_TRAINING) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_TRAINING))
    from cpp_engine_matchup import build_cpp_engine as _build  # noqa: PLC0415

    return _build(
        deck1,
        deck2,
        assets_path=env.assets_path,
        talishar_url=env.talishar_url,
        deck1_json=deck1_json,
        deck2_json=deck2_json,
    )


def build_cpp_engine_for_spec(spec: ExperimentSpec, env: EnvironmentSettings) -> int:
    deck1, deck2, deck1_json, deck2_json = cpp_build_inputs_for_spec(spec, env)
    return build_cpp_engine(
        deck1,
        deck2,
        env,
        deck1_json=deck1_json,
        deck2_json=deck2_json,
    )


def run_full_pipeline(
    spec: ExperimentSpec,
    env: EnvironmentSettings,
    *,
    cpp_engine_dir: str | None = None,
) -> int:
    env.apply_to_environ()
    out_dir = spec.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [_python(), str(SCRIPTS_TRAINING / "train_full_pipeline.py")]
    cmd.extend(spec.pipeline_argv())
    cmd.extend(["--talishar-url", env.talishar_url])
    cmd.extend(["--talishar-fe-url", env.talishar_fe_url])
    cmd.extend(["--assets-path", env.assets_path])
    if cpp_engine_dir:
        cmd.extend(["--cpp-engine-dir", cpp_engine_dir])
    return run_streaming(cmd)


def run_live_talishar_play(spec: LivePlaySpec, env: EnvironmentSettings) -> int:
    env.apply_to_environ()
    cmd = [
        _python(),
        str(SCRIPTS_EVAL / "talishar_live_play.py"),
        "--results-dir",
        spec.results_dir,
        "--games",
        str(spec.games),
        "--max-steps",
        str(spec.max_steps),
        "--step-delay-ms",
        str(spec.step_delay_ms),
        "--talishar-url",
        env.talishar_url,
        "--talishar-fe-url",
        env.talishar_fe_url,
        "--assets-path",
        env.assets_path,
    ]
    if spec.candidate_id:
        cmd.extend(["--candidate-id", spec.candidate_id])
    if spec.seed is not None:
        cmd.extend(["--seed", str(spec.seed)])
    if spec.human_vs_agent:
        cmd.append("--human-vs-agent")
        cmd.extend(["--human-deck", spec.human_deck])
        if not spec.enable_action_coach:
            cmd.append("--no-action-coach")
        cmd.extend(["--coach-rollouts-per-action", str(spec.coach_rollouts_per_action)])
    return run_streaming(cmd)


def run_card_db_rescan(*, legality_scope: str = "all", dry_run: bool = False) -> int:
    """Refresh ``cards.json`` from the official FAB Card Vault API."""
    import os

    if not CARDS_DB_UPDATE_SCRIPT.is_file():
        raise FileNotFoundError(f"Card DB updater not found: {CARDS_DB_UPDATE_SCRIPT}")
    if not CARDS_DB_PATH.is_file():
        raise FileNotFoundError(f"cards.json not found: {CARDS_DB_PATH}")
    if not FABRARY_DECKS_PATH.is_file():
        raise FileNotFoundError(f"fabrary_decks.json not found: {FABRARY_DECKS_PATH}")

    src_path = str(REPO_ROOT / "src")
    existing_py_path = os.environ.get("PYTHONPATH", "")
    pythonpath = f"{src_path}{os.pathsep}{existing_py_path}" if existing_py_path else src_path

    cmd = [
        _python(),
        "-m",
        "flesh_and_blood_rlbridge.card_db.update_cards_db_from_fabtcg",
        "--cards",
        str(CARDS_DB_PATH),
        "--decks",
        str(FABRARY_DECKS_PATH),
        "--legality-scope",
        legality_scope,
    ]
    if dry_run:
        cmd.append("--dry-run")
    rc = run_streaming(cmd, extra_env={"PYTHONPATH": pythonpath})
    if rc == 0 and not dry_run:
        norm_rc, _summary = normalize_card_db_for_talishar()
        if norm_rc != 0:
            return norm_rc
    return rc


def normalize_card_db_for_talishar(*, dry_run: bool = False) -> tuple[int, dict[str, Any]]:
    """Rewrite ``cards.json`` IDs to match Talishar's generated card dictionary."""
    import os

    src_path = str(REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from flesh_and_blood_rlbridge.card_db.talishar_card_ids import (  # noqa: PLC0415
        normalize_cards_json_file,
    )

    summary = normalize_cards_json_file(CARDS_DB_PATH, dry_run=dry_run)
    print(
        "  Card ID normalization: "
        f"{summary['before']} -> {summary['after']} cards "
        f"({summary['remapped']} remapped, {summary['dropped']} dropped)"
    )
    return 0, summary


def run_eval_dashboard(spec: EvalSpec, env: EnvironmentSettings) -> int:
    env.apply_to_environ()
    cmd = [
        _python(),
        str(SCRIPTS_EVAL / "eval_phase3_checkpoint.py"),
        "--results-dir",
        spec.results_dir,
        "--assets-path",
        env.assets_path,
        "--talishar-url",
        env.talishar_url,
        "--episodes",
        str(spec.episodes),
        "--parallel-workers",
        str(spec.parallel_workers),
        "--max-steps",
        str(spec.max_steps),
        "--render-max-steps",
        str(spec.max_steps),
        "--poll-seconds",
        str(spec.poll_seconds),
    ]
    if spec.candidate_id:
        cmd.extend(["--candidate-id", spec.candidate_id])
    if spec.watch:
        cmd.append("--watch")
    if spec.render_only:
        cmd.append("--render-only")
    return run_streaming(cmd)


def run_runscript(script_name: str, *script_args: str) -> int:
    script = RUNSCRIPTS_ROOT / script_name
    if not script.is_file():
        raise FileNotFoundError(f"Runscript not found: {script}")
    return run_streaming([_python(), str(script), *script_args])


def discover_cpp_engine_dir(
    deck1: str,
    deck2: str,
    *,
    assets_path: str | None = None,
    deck1_json: Path | None = None,
    deck2_json: Path | None = None,
) -> Path | None:
    if str(SCRIPTS_TRAINING) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_TRAINING))
    from cpp_engine_matchup import discover_cpp_engine_dir as _discover  # noqa: PLC0415

    assets = assets_path or str(REPO_ROOT / "Talishar" / "Assets")
    return _discover(
        deck1,
        deck2,
        assets_path=assets,
        deck1_json=deck1_json,
        deck2_json=deck2_json,
    )


def ensure_cpp_engine_for_spec(
    spec: ExperimentSpec,
    env: EnvironmentSettings,
    *,
    build: bool = True,
) -> str | None:
    """Discover or build the C++ engine for *spec*'s matchup."""
    if str(SCRIPTS_TRAINING) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_TRAINING))
    from cpp_engine_matchup import ensure_cpp_engine_for_matchup  # noqa: PLC0415

    deck1, deck2, deck1_json, deck2_json = cpp_build_inputs_for_spec(spec, env)
    if spec.opponent_mode == "preset":
        deck2_key = spec.opponent_deck
    elif spec.opponent_mode == "mirror":
        deck2_key = spec.hero_id
    else:
        deck2_key = spec.p2_hero_id
    return ensure_cpp_engine_for_matchup(
        spec.hero_id,
        deck2_key,
        assets_path=env.assets_path,
        talishar_url=env.talishar_url,
        deck1_json=deck1_json,
        deck2_json=deck2_json,
        build=build,
    )


def run_matchup_simulation(
    spec: MatchupSimSpec,
    env: EnvironmentSettings,
    *,
    deck1_json: Path,
    deck2_json: Path,
) -> int:
    """Run fixed-deck matchup simulation via ``train_full_pipeline.py``."""
    from runscripts._common import read_deck_meta  # noqa: PLC0415

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    env.apply_to_environ()
    p1_meta = read_deck_meta(deck1_json, spec.game_format)
    p2_meta = read_deck_meta(deck2_json, spec.game_format)
    effective_format = normalize_pipeline_format(p1_meta.fmt or spec.game_format)
    matchup_label = f"{p1_meta.short_name}_vs_{p2_meta.short_name}"
    out_dir = RESULTS_ROOT / "matchup_sims" / matchup_label
    results_json = out_dir / "results.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    cpp_dir: str | None = None
    if spec.build_cpp_engine:
        deck1 = _resolve_deck_stem(p1_meta.hero_id, env.assets_path)
        deck2 = _resolve_deck_stem(p2_meta.hero_id, env.assets_path)
        existing = discover_cpp_engine_dir(
            deck1,
            deck2,
            assets_path=env.assets_path,
            deck1_json=deck1_json,
            deck2_json=deck2_json,
        )
        if existing is not None:
            cpp_dir = str(existing)
        else:
            rc = build_cpp_engine(
                deck1,
                deck2,
                env,
                deck1_json=deck1_json,
                deck2_json=deck2_json,
            )
            if rc == 0:
                found = discover_cpp_engine_dir(
                    deck1,
                    deck2,
                    assets_path=env.assets_path,
                    deck1_json=deck1_json,
                    deck2_json=deck2_json,
                )
                cpp_dir = str(found) if found else None

    cmd = [
        _python(),
        str(SCRIPTS_TRAINING / "train_full_pipeline.py"),
        "--format",
        effective_format,
        "--hero-id",
        p1_meta.hero_id,
        "--hero-class",
        p1_meta.hero_class,
        "--equipment-header",
        p1_meta.equipment_header,
        "--opponent-mode",
        "dual",
        "--p2-hero-id",
        p2_meta.hero_id,
        "--p2-hero-class",
        p2_meta.hero_class,
        "--p2-equipment-header",
        p2_meta.equipment_header,
        "--opponent-hero-id",
        p2_meta.hero_id,
        "--deckbuild-episodes",
        "0",
        "--play-episodes",
        str(spec.play_episodes),
        "--warmup-episodes",
        str(spec.warmup_episodes),
        "--warmup-baseline-eval-episodes",
        "20",
        "--iterations",
        str(spec.iterations),
        "--final-eval-episodes",
        str(spec.final_eval_episodes),
        "--final-eval-max-steps",
        str(spec.final_eval_max_steps),
        "--gif-fps",
        "2.0",
        "--talishar-url",
        env.talishar_url,
        "--assets-path",
        env.assets_path,
        "--out-dir",
        str(out_dir),
        "--results-json",
        str(results_json),
        "--p1-fixed-deck",
        str(deck1_json),
        "--p2-fixed-deck",
        str(deck2_json),
    ]
    if cpp_dir:
        cmd.extend(["--cpp-engine-dir", cpp_dir])
    if spec.workers is not None:
        cmd.extend(["--workers", str(spec.workers)])
    return run_streaming(cmd)


def run_sideboard_compare(
    spec: SideboardCompareSpec,
    env: EnvironmentSettings,
    *,
    starting_deck: Path,
    candidates_json: Path | None = None,
) -> int:
    """Run sideboard variant comparison via ``train_sideboard_compare.py``."""
    from runscripts._common import read_deck_meta  # noqa: PLC0415

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    env.apply_to_environ()
    deck_meta = read_deck_meta(starting_deck, spec.game_format)
    out_dir = spec.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    cpp_dir: str | None = None
    if spec.build_cpp_engine:
        deck1 = _resolve_deck_stem(deck_meta.hero_id or spec.hero_id, env.assets_path)
        deck2 = _resolve_deck_stem(spec.opponent_deck, env.assets_path)
        existing = discover_cpp_engine_dir(
            deck1,
            deck2,
            assets_path=env.assets_path,
            deck1_json=starting_deck,
        )
        if existing is not None:
            cpp_dir = str(existing)
        else:
            rc = build_cpp_engine(
                deck1,
                deck2,
                env,
                deck1_json=starting_deck,
            )
            if rc == 0:
                found = discover_cpp_engine_dir(
                    deck1,
                    deck2,
                    assets_path=env.assets_path,
                    deck1_json=starting_deck,
                )
                cpp_dir = str(found) if found else None

    cmd = [
        _python(),
        str(SCRIPTS_TRAINING / "train_sideboard_compare.py"),
        "--format",
        normalize_pipeline_format(spec.game_format),
        "--hero-id",
        spec.hero_id or deck_meta.hero_id,
        "--hero-class",
        spec.hero_class or deck_meta.hero_class,
        "--equipment-header",
        spec.equipment_header or deck_meta.equipment_header,
        "--opponent-hero-id",
        spec.opponent_hero_id,
        "--opponent-deck",
        spec.opponent_deck,
        "--starting-deck",
        str(starting_deck),
        "--num-options",
        str(spec.num_options),
        "--max-parallel",
        str(spec.max_parallel),
        "--play-episodes",
        str(spec.play_episodes),
        "--checkpoint-interval-pct",
        str(spec.checkpoint_interval_pct),
        "--checkpoint-eval-episodes",
        str(spec.checkpoint_eval_episodes),
        "--parallel-seeds",
        str(spec.parallel_seeds),
        "--final-eval-episodes",
        str(spec.final_eval_episodes),
        "--final-eval-max-steps",
        str(spec.final_eval_max_steps),
        "--out-dir",
        str(out_dir),
        "--cache-dir",
        str(REPO_ROOT / "results" / "agent_cache"),
        "--talishar-url",
        env.talishar_url,
        "--talishar-fe-url",
        env.talishar_fe_url,
        "--assets-path",
        env.assets_path,
    ]
    if spec.play_checkpoint_interval is not None:
        cmd.extend([
            "--play-checkpoint-interval",
            str(spec.play_checkpoint_interval),
        ])
    if spec.skip_final_eval:
        cmd.append("--skip-final-eval")
    if not spec.no_render_gif:
        cmd.append("--render-gif")
    if candidates_json is not None:
        cmd.extend(["--candidates-json", str(candidates_json)])
    if cpp_dir:
        cmd.extend(["--cpp-engine-dir", cpp_dir])
    if spec.workers is not None:
        cmd.extend(["--workers", str(spec.workers)])

    dashboard_proc = None
    try:
        from runscripts._common import (  # noqa: PLC0415
            start_sideboard_compare_dashboard,
            stop_background_process,
        )

        dashboard_proc = start_sideboard_compare_dashboard(out_dir)
        return run_streaming(cmd)
    finally:
        if dashboard_proc is not None:
            from runscripts._common import stop_background_process  # noqa: PLC0415

            stop_background_process(dashboard_proc)


def run_unified_random_matchups(
    spec: UnifiedRandomMatchupSpec,
    env: EnvironmentSettings,
) -> int:
    """Train unified agent on random fabrary deck matchups."""
    env.apply_to_environ()
    out_dir = spec.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = spec.cache_dir or str(RESULTS_ROOT / "agent_cache")

    cmd = [
        _python(),
        str(SCRIPTS_TRAINING / "train_unified_random_matchups.py"),
        "--format",
        normalize_pipeline_format(spec.game_format),
        "--matchups",
        str(spec.matchups),
        "--episodes",
        str(spec.episodes),
        "--max-steps",
        str(spec.max_steps),
        "--warmup-episodes",
        str(spec.warmup_episodes),
        "--checkpoint-interval-pct",
        str(spec.checkpoint_interval_pct),
        "--checkpoint-eval-episodes",
        str(spec.checkpoint_eval_episodes),
        "--workers",
        str(spec.workers),
        "--run-dir",
        str(out_dir),
        "--cache-dir",
        cache_dir,
        "--talishar-url",
        env.talishar_url,
    ]
    if spec.seed is not None:
        cmd.extend(["--seed", str(spec.seed)])
    if spec.skip_converged:
        cmd.append("--skip-converged")
    else:
        cmd.append("--no-skip-converged")
    if not spec.build_cpp_engine:
        cmd.append("--no-build-cpp-engine")
        cmd.append("--no-require-cpp-engine")
    return run_streaming(cmd)
