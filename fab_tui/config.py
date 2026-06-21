"""Experiment configuration and path helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SCRIPTS_TRAINING = SCRIPTS_ROOT / "training"
SCRIPTS_EVAL = SCRIPTS_ROOT / "eval"
SCRIPTS_CPP = SCRIPTS_ROOT / "cpp"
SCRIPTS_DECK = SCRIPTS_ROOT / "deck"
RUNSCRIPTS_ROOT = REPO_ROOT / "runscripts"
RESULTS_ROOT = REPO_ROOT / "results"
AGENT_CACHE_DIR = RESULTS_ROOT / "agent_cache"

OpponentMode = Literal["preset", "mirror", "dual"]
FormatChoice = Literal["silver_age", "classic_constructed", "blitz", "upf", "sage"]
PipelineFormat = Literal["silver_age", "classic_constructed", "blitz", "upf"]
Workflow = Literal["full", "draft_only", "play_only", "matchup_sim"]


def normalize_pipeline_format(fmt: str) -> PipelineFormat:
    """Map TUI / deck JSON labels to ``train_full_pipeline.py --format`` choices."""
    token = str(fmt or "").strip().lower().replace(" ", "_")
    if token in {"sage", "silver_age", "silver", "silverage"}:
        return "silver_age"
    if token in {"classic_constructed", "blitz", "upf"}:
        return token  # type: ignore[return-value]
    return "silver_age"


def slugify(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return token or "experiment"


@dataclass
class EnvironmentSettings:
    talishar_url: str = field(
        default_factory=lambda: os.environ.get("TALISHAR_URL", "http://localhost:8080/game")
    )
    talishar_fe_url: str = field(
        default_factory=lambda: os.environ.get("TALISHAR_FE_URL", "http://localhost:5173")
    )
    assets_path: str = field(
        default_factory=lambda: os.environ.get(
            "TALISHAR_ASSETS_PATH", str(REPO_ROOT / "Talishar" / "Assets")
        )
    )
    fabrary_api_key: str = field(
        default_factory=lambda: os.environ.get("FABRARY_API_KEY", "")
    )

    def apply_to_environ(self) -> None:
        os.environ["TALISHAR_URL"] = self.talishar_url
        os.environ["TALISHAR_FE_URL"] = self.talishar_fe_url
        os.environ["TALISHAR_ASSETS_PATH"] = self.assets_path
        if self.fabrary_api_key:
            os.environ["FABRARY_API_KEY"] = self.fabrary_api_key


@dataclass
class ExperimentSpec:
    """User-facing experiment definition mapped to ``train_full_pipeline.py`` flags."""

    name: str
    workflow: Workflow = "full"
    game_format: FormatChoice = "silver_age"
    opponent_mode: OpponentMode = "dual"

    hero_id: str = "aurora"
    hero_class: str = "Runeblade"
    equipment_header: str = (
        "aurora star_fall aether_ironweave spellbound_creepers "
        "aether_crackers crown_of_dichotomy"
    )

    p2_hero_id: str = "briar"
    p2_hero_class: str = "Runeblade"
    p2_equipment_header: str = (
        "briar star_fall aether_ironweave spellbound_creepers "
        "aether_crackers crown_of_dichotomy"
    )
    opponent_deck: str = "Dummy"
    opponent_hero_id: str = ""

    p1_starting_deck: str | None = None
    p2_starting_deck: str | None = None
    p1_fixed_deck: str | None = None
    p2_fixed_deck: str | None = None

    deckbuild_episodes: int = 50
    sideboard_episodes: int = 20
    play_episodes: int = 100
    iterations: int = 3
    num_eval_games: int = 20
    sideboard_eval_games: int = 1000
    final_eval_episodes: int = 50
    final_eval_max_steps: int = 200
    workers: int | None = None

    build_cpp_engine: bool = True
    out_dir: str | None = None
    results_json: str | None = None

    def resolved_out_dir(self) -> Path:
        if self.out_dir:
            return Path(self.out_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return RESULTS_ROOT / "experiments" / f"{slugify(self.name)}_{stamp}"

    def pipeline_argv(self) -> list[str]:
        args = [
            "--format",
            normalize_pipeline_format(self.game_format),
            "--hero-id",
            self.hero_id,
            "--hero-class",
            self.hero_class,
            "--equipment-header",
            self.equipment_header,
            "--opponent-mode",
            self.opponent_mode,
            "--opponent-deck",
            self.opponent_deck,
            "--opponent-hero-id",
            self.opponent_hero_id or self.p2_hero_id,
            "--p2-hero-id",
            self.p2_hero_id,
            "--p2-hero-class",
            self.p2_hero_class,
            "--p2-equipment-header",
            self.p2_equipment_header,
            "--deckbuild-episodes",
            str(self.deckbuild_episodes),
            "--play-episodes",
            str(self.play_episodes),
            "--iterations",
            str(self.iterations),
            "--num-eval-games",
            str(self.num_eval_games),
            "--final-eval-episodes",
            str(self.final_eval_episodes),
            "--final-eval-max-steps",
            str(self.final_eval_max_steps),
            "--out-dir",
            str(self.resolved_out_dir()),
        ]
        if self.results_json:
            args.extend(["--results-json", self.results_json])
        else:
            args.extend(["--results-json", str(self.resolved_out_dir() / "results.json")])
        if self.p1_starting_deck:
            args.extend(["--p1-starting-deck", self.p1_starting_deck])
        if self.p2_starting_deck:
            args.extend(["--p2-starting-deck", self.p2_starting_deck])
        if self.p1_fixed_deck:
            args.extend(["--p1-fixed-deck", self.p1_fixed_deck])
        if self.p2_fixed_deck:
            args.extend(["--p2-fixed-deck", self.p2_fixed_deck])
        if self.workers is not None:
            args.extend(["--workers", str(self.workers)])
        return args

    def apply_workflow_defaults(self) -> None:
        if self.workflow == "draft_only":
            self.play_episodes = 0
            self.final_eval_episodes = 0
        elif self.workflow == "play_only":
            self.deckbuild_episodes = 0
            self.sideboard_episodes = 0


DEFAULT_CHECKPOINT_INTERVAL_PCT = 5.0
DEFAULT_CHECKPOINT_EVAL_EPISODES = 100


@dataclass
class EvalSpec:
    results_dir: str
    episodes: int = 20
    parallel_workers: int = 4
    max_steps: int = 1000
    watch: bool = False
    poll_seconds: int = 30
    candidate_id: str | None = None


@dataclass
class MatchupSimSpec:
    deck1_source: str
    deck2_source: str
    game_format: FormatChoice = "silver_age"
    play_episodes: int = 500
    checkpoint_interval_pct: float = DEFAULT_CHECKPOINT_INTERVAL_PCT
    checkpoint_eval_episodes: int = DEFAULT_CHECKPOINT_EVAL_EPISODES
    play_checkpoint_interval: int | None = None
    final_eval_episodes: int = 50
    final_eval_max_steps: int = 500
    sideboard_episodes: int = 30
    warmup_episodes: int = 50
    iterations: int = 1
    build_cpp_engine: bool = True
    workers: int | None = None


@dataclass
class SideboardCompareSpec:
    """Sideboard variant comparison via ``train_sideboard_compare.py``."""

    starting_deck: str
    opponent_hero_id: str
    opponent_deck: str
    hero_id: str = ""
    hero_class: str = ""
    equipment_header: str = ""
    game_format: FormatChoice = "silver_age"
    num_options: int = 4
    max_parallel: int = 2
    max_swap_variants: int = 2
    max_swaps_per_variant: int = 1
    play_episodes: int = 10000
    checkpoint_interval_pct: float = DEFAULT_CHECKPOINT_INTERVAL_PCT
    checkpoint_eval_episodes: int = DEFAULT_CHECKPOINT_EVAL_EPISODES
    play_checkpoint_interval: int | None = None
    final_eval_episodes: int = 50
    final_eval_max_steps: int = 200
    skip_final_eval: bool = False
    no_render_gif: bool = True
    build_cpp_engine: bool = True
    workers: int | None = None
    out_dir: str | None = None
    candidates_json: str | None = None

    def resolved_out_dir(self) -> Path:
        if self.out_dir:
            return Path(self.out_dir)
        hero = slugify(self.hero_id or "player")
        opp = slugify(self.opponent_hero_id or "opponent")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return RESULTS_ROOT / "sideboard_compare" / f"{hero}_vs_{opp}_{stamp}"
