"""Subprocess runners for experiments and evaluation."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fab_tui.config import (
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
    MatchupSimSpec,
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


def run_streaming(cmd: list[str], *, cwd: Path | None = None) -> int:
    """Run a command forwarding stdout/stderr to the terminal."""
    with subprocess.Popen(
        cmd,
        cwd=str(cwd or REPO_ROOT),
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
    if spec.watch:
        cmd.append("--watch")
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
