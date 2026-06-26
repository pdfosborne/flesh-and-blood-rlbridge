"""Experiment configuration and path helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from fab_bridge.paths import configure_import_paths, repo_root
from runtime_defaults import RUNTIME

configure_import_paths()
REPO_ROOT = repo_root()
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SCRIPTS_TRAINING = SCRIPTS_ROOT / "training"
SCRIPTS_EVAL = SCRIPTS_ROOT / "eval"
SCRIPTS_CPP = SCRIPTS_ROOT / "cpp"
SCRIPTS_DECK = SCRIPTS_ROOT / "deck"
RUNSCRIPTS_ROOT = REPO_ROOT / "runscripts"
RESULTS_ROOT = REPO_ROOT / "results"


def _resolve_agent_cache_dir() -> Path:
    override = os.environ.get("FAB_AGENT_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return RESULTS_ROOT / "agent_cache"


AGENT_CACHE_DIR = _resolve_agent_cache_dir()
CARDS_DB_DIR = REPO_ROOT / "src" / "flesh_and_blood_rlbridge" / "card_db"
CARDS_DB_PATH = CARDS_DB_DIR / "cards.json"
FABRARY_DECKS_PATH = CARDS_DB_DIR / "fabrary_decks.json"
CARDS_DB_UPDATE_SCRIPT = CARDS_DB_DIR / "update_cards_db_from_fabtcg.py"

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


_DOCKER_FE_HOSTS = frozenset(
    {"talishar-fe", "web-server", "fab-bridge", "host.docker.internal"}
)
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    token = host.lower().strip("[]")
    return token in _LOOPBACK_HOSTS or token.startswith("127.")


def _loopback_fe_browser_host(*, fe_host: str, page_host: str | None) -> str:
    """Browser host for Talishar-FE when both GUI and FE are on loopback."""
    if page_host and not _is_loopback_host(page_host):
        return page_host
    # Vite on Windows often binds [::1] only — localhost works, 127.0.0.1 may not.
    if _is_loopback_host(fe_host) or fe_host in {"", "localhost"}:
        return "localhost"
    return fe_host or "localhost"


def browser_talishar_fe_url(fe_url: str, *, page_host: str | None = None) -> str:
    """Map docker-internal Talishar-FE URLs to a host the **user's browser** can reach.

    Do not use this for server-side probes or Playwright inside the fab-bridge
    container — pass ``EnvironmentSettings.talishar_fe_url`` (e.g.
    ``http://talishar-fe:5173``) directly instead.
    """
    override = os.environ.get("TALISHAR_FE_BROWSER_URL", "").strip()
    if override:
        return override.rstrip("/")
    from urllib.parse import urlparse

    parsed = urlparse(fe_url)
    host = (parsed.hostname or "").lower()
    port = parsed.port or 5173
    scheme = parsed.scheme or "http"
    if host in _DOCKER_FE_HOSTS:
        browser_host = (page_host or "127.0.0.1").strip() or "127.0.0.1"
    elif _is_loopback_host(host):
        browser_host = _loopback_fe_browser_host(fe_host=host, page_host=page_host)
    else:
        browser_host = host or "localhost"
    if browser_host in {"0.0.0.0", "[::]", "::"}:
        browser_host = "localhost"
    return f"{scheme}://{browser_host}:{port}"


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

    deckbuild_episodes: int = RUNTIME.tui.deckbuild_episodes
    sideboard_episodes: int = RUNTIME.tui.sideboard_episodes
    play_episodes: int = RUNTIME.tui.play_episodes
    iterations: int = RUNTIME.tui.iterations
    num_eval_games: int = RUNTIME.tui.num_eval_games
    sideboard_eval_games: int = RUNTIME.tui.sideboard_eval_games
    final_eval_episodes: int = RUNTIME.tui.final_eval_episodes
    final_eval_max_steps: int = RUNTIME.tui.final_eval_max_steps
    workers: int | None = RUNTIME.play.workers

    build_cpp_engine: bool = True
    out_dir: str | None = None
    results_json: str | None = None

    def resolved_out_dir(self) -> Path:
        if self.out_dir:
            return Path(self.out_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_ROOT / "experiments" / f"{slugify(self.name)}_{stamp}"
        self.out_dir = str(path)
        return path

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


DEFAULT_CHECKPOINT_INTERVAL_PCT = RUNTIME.play.checkpoint_interval_pct
DEFAULT_CHECKPOINT_EVAL_EPISODES = RUNTIME.play.checkpoint_eval_episodes


@dataclass
class EvalSpec:
    results_dir: str
    episodes: int = RUNTIME.tui.eval_episodes
    parallel_workers: int = RUNTIME.tui.eval_parallel_workers
    max_steps: int = RUNTIME.tui.eval_max_steps
    watch: bool = False
    poll_seconds: int = RUNTIME.tui.eval_poll_seconds
    candidate_id: str | None = None
    render_only: bool = False


@dataclass
class LivePlaySpec:
    """Real-time Talishar play with the live frontend."""

    results_dir: str
    candidate_id: str | None = None
    games: int = 1
    max_steps: int = RUNTIME.tui.eval_max_steps
    step_delay_ms: int = 0
    seed: int | None = None
    human_vs_agent: bool = False
    human_deck: str = "opponent"  # trained | opponent
    enable_action_coach: bool = True
    coach_rollouts_per_action: int = 4


@dataclass
class MatchupSimSpec:
    deck1_source: str
    deck2_source: str
    game_format: FormatChoice = "silver_age"
    play_episodes: int = RUNTIME.matchup_sim.play_episodes
    checkpoint_interval_pct: float = DEFAULT_CHECKPOINT_INTERVAL_PCT
    checkpoint_eval_episodes: int = DEFAULT_CHECKPOINT_EVAL_EPISODES
    play_checkpoint_interval: int | None = RUNTIME.play.play_checkpoint_interval
    final_eval_episodes: int = RUNTIME.matchup_sim.final_eval_episodes
    final_eval_max_steps: int = RUNTIME.matchup_sim.final_eval_max_steps
    sideboard_episodes: int = RUNTIME.matchup_sim.sideboard_episodes
    warmup_episodes: int = RUNTIME.matchup_sim.warmup_episodes
    iterations: int = RUNTIME.matchup_sim.iterations
    build_cpp_engine: bool = True
    workers: int | None = RUNTIME.matchup_sim.workers


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
    num_options: int = RUNTIME.sideboard_compare.num_options
    max_parallel: int = RUNTIME.sideboard_compare.max_parallel
    max_swap_variants: int = RUNTIME.sideboard_compare.max_swap_variants
    max_swaps_per_variant: int = RUNTIME.sideboard_compare.max_swaps_per_variant
    play_episodes: int = RUNTIME.sideboard_compare.play_episodes
    checkpoint_interval_pct: float = DEFAULT_CHECKPOINT_INTERVAL_PCT
    checkpoint_eval_episodes: int = DEFAULT_CHECKPOINT_EVAL_EPISODES
    play_checkpoint_interval: int | None = RUNTIME.play.play_checkpoint_interval
    final_eval_episodes: int = RUNTIME.sideboard_compare.final_eval_episodes
    final_eval_max_steps: int = RUNTIME.sideboard_compare.final_eval_max_steps
    parallel_seeds: int = RUNTIME.play.parallel_seeds
    skip_final_eval: bool = False
    no_render_gif: bool = True
    build_cpp_engine: bool = True
    workers: int | None = RUNTIME.play.workers
    out_dir: str | None = None
    candidates_json: str | None = None

    def resolved_out_dir(self) -> Path:
        if self.out_dir:
            return Path(self.out_dir)
        hero = slugify(self.hero_id or "player")
        opp = slugify(self.opponent_hero_id or "opponent")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_ROOT / "sideboard_compare" / f"{hero}_vs_{opp}_{stamp}"
        self.out_dir = str(path)
        return path


@dataclass
class UnifiedRandomMatchupSpec:
    """Random fabrary deck matchups for unified agent training."""

    game_format: PipelineFormat = "silver_age"
    matchups: int = 3
    episodes: int = RUNTIME.dual_matchup.episodes
    max_steps: int = RUNTIME.dual_matchup.max_steps
    warmup_episodes: int = RUNTIME.play.warmup_episodes
    checkpoint_interval_pct: float = DEFAULT_CHECKPOINT_INTERVAL_PCT
    checkpoint_eval_episodes: int = DEFAULT_CHECKPOINT_EVAL_EPISODES
    workers: int = RUNTIME.dual_matchup.workers
    skip_converged: bool = True
    build_cpp_engine: bool = True
    require_cpp_engine: bool = True
    seed: int | None = None
    cache_dir: str | None = None
    out_dir: str | None = None

    def resolved_out_dir(self) -> Path:
        if self.out_dir:
            return Path(self.out_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = (
            RESULTS_ROOT
            / "unified_random_matchups"
            / normalize_pipeline_format(self.game_format)
            / stamp
        )
        self.out_dir = str(path)
        return path
