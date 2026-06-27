"""Central runtime defaults for FaB RL training and runscripts.

Edit ``META`` below to tune parallelism, training budget, checkpoints, eval,
and game-control settings. Workflow sections under ``RUNTIME`` are derived from
``META`` plus a few fixed per-script knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

@dataclass
class MetaGameControls:
    """Stall detection — when to force-skip stuck eval / replay sessions."""

    stall_no_damage_turns: int = 6
    stall_low_hand_turns: int = 3
    stall_max_single_low_hand_turns: int = 5
    stall_min_attack_hand: int = 2


@dataclass
class MetaUnifiedRandomMatchups:
    """Unified random fabrary matchup training (``train_unified_random_matchups.py``)."""

    matchups: int = 10
    episodes: int = 1000
    max_steps: int = 1_000
    warmup_episodes: int = 50
    checkpoint_interval_pct: float = 10.0
    checkpoint_eval_episodes: int = 0  # 0 = min(100, episodes // 100)
    workers: int = 0  # 0 = inherit META.workers
    skip_converged: bool = False
    build_cpp_engine: bool = True
    require_cpp_engine: bool = True
    seed: int | None = None
    cache_dir: str | None = None
    out_dir: str | None = None
    # FaBrary slugs or full URLs; run scripts/deck/add_custom_decks_to_pool.py after editing.
    #  -- Or add using TUI option 
    custom_deck_links: tuple[str, ...] = ()


@dataclass
class MetaRuntime:
    """Shared runtime knobs — edit this block for your machine and training budget."""

    # ── Parallelism ──────────────────────────────────────────────────────────
    workers: int | None = 16  # C++ env workers (None = auto-detect)
    parallel_seeds: int = 4  # independent seeds; best model used for eval
    parallel_seeds_until_first_checkpoint: bool = True
    sideboard_max_parallel: int = 0  # 0 = train all sideboard candidates at once
    eval_parallel_workers: int = 4

    # ── Training budget (episodes) ───────────────────────────────────────────
    play_episodes: int = 100  # Phase 3 play training (all workflows)

    # ── Checkpoints during training ──────────────────────────────────────────
    checkpoint_interval_pct: float = 5.0
    checkpoint_eval_episodes: int = min(100, int(play_episodes/100)) # 1% of play episodes
    play_checkpoint_interval: int | None = int(play_episodes/20) # 5% of play episodes

    # ── Per-episode limits & warmup ──────────────────────────────────────────
    max_play_steps: int = 1_000  # max turns per play-training episode (all workflows)
    warmup_episodes: int = 500  # heuristic-policy episodes before PPO
    warmup_baseline_eval_episodes: int = 0

    # ── Training Checkpoint evaluation (uses cpp engine if possible)  ────────
    eval_episodes: int = 100  # checkpoint / phase-3 eval watcher
    eval_max_steps: int = 1_000

    # ── Final evaluation (after training, uses Talishar engine) ──────────────
    final_eval_episodes: int = 100
    final_eval_max_steps: int = 1_000
    
    # ── Game controls (stall / force-skip) ───────────────────────────────────
    game: MetaGameControls = field(default_factory=MetaGameControls)

    # ── Dashboard / rendering ────────────────────────────────────────────────
    dashboard_poll_seconds: float = 5.0
    eval_poll_seconds: int = 30
    gif_fps: float = 3.0
    gif_fps_matchup_sim: float = 2.0

    # ── Unified random fabrary matchups ───────────────────────────────────────
    unified_random_matchups: MetaUnifiedRandomMatchups = field(
        default_factory=MetaUnifiedRandomMatchups
    )

# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GameControlsDefaults:
    """Force-skip thresholds for stuck games (eval checkpoint watcher, etc.)."""

    stall_no_damage_turns: int
    stall_low_hand_turns: int
    stall_max_single_low_hand_turns: int
    stall_min_attack_hand: int


@dataclass(frozen=True)
class PlayDefaults:
    """Shared Phase 3 play-training knobs."""

    workers: int | None
    parallel_seeds: int
    parallel_seeds_until_first_checkpoint: bool
    checkpoint_interval_pct: float
    checkpoint_eval_episodes: int
    play_checkpoint_interval: int | None
    max_play_steps: int
    warmup_episodes: int
    warmup_baseline_eval_episodes: int
    gif_fps: float


@dataclass(frozen=True)
class SideboardCompareDefaults:
    """``train_sideboard_compare.py`` / ``runscripts/sideboard_compare.py``."""

    game_format: str = "silver_age"
    num_options: int = 4
    max_parallel: int = 0
    play_episodes: int = 0
    final_eval_episodes: int = 0
    final_eval_max_steps: int = 0
    dashboard_poll_seconds: float = 5.0
    min_swap_margin: float = 0.75
    max_swap_variants: int = 2
    max_swaps_per_variant: int = 1


@dataclass(frozen=True)
class FullPipelineDefaults:
    """``train_full_pipeline.py`` orchestrator."""

    deckbuild_episodes: int = 50
    sideboard_episodes: int = 20
    play_episodes: int = 0
    iterations: int = 3
    num_eval_games: int = 3
    num_sideboard_episodes: int = 1
    max_build_steps: int = 200
    max_sideboard_steps: int = 100
    final_eval_episodes: int = 0
    final_eval_max_steps: int = 0


@dataclass(frozen=True)
class MatchupSimDefaults:
    """``runscripts/simulate_deck_matchup.py``."""

    game_format: str = "silver_age"
    play_episodes: int = 0
    max_play_steps: int = 0
    final_eval_episodes: int = 0
    final_eval_max_steps: int = 0
    sideboard_episodes: int = 30
    warmup_episodes: int = 0
    warmup_baseline_eval_episodes: int = 0
    num_eval_games: int = 50
    iterations: int = 1
    gif_fps: float = 2.0
    workers: int | None = None


@dataclass(frozen=True)
class EvalDashboardDefaults:
    """``runscripts/phase3_eval_dashboard.py`` / checkpoint eval watcher."""

    episodes: int = 0
    parallel_workers: int = 0
    max_steps: int = 0
    render_max_steps: int = 500
    poll_seconds: int = 30
    gif_fps: int = 3


@dataclass(frozen=True)
class DualMatchupDefaults:
    """Fabrary deck cross-matchup training (sage / silver age / classic)."""

    episodes: int = 0
    max_steps: int = 0
    workers: int = 1


@dataclass(frozen=True)
class PpoDefaults:
    """PPO hyperparameters shared by dual-agent trainers."""

    hidden_size: int = 64
    n_layers: int = 2
    n_heads: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    n_steps: int = 512
    ppo_epochs: int = 4
    mini_batch: int = 256
    rollout_batch: int = 512


@dataclass(frozen=True)
class TuiDefaults:
    """``fab_tui`` experiment specs (overrides per workflow still apply)."""

    deckbuild_episodes: int = 50
    sideboard_episodes: int = 20
    play_episodes: int = 0
    iterations: int = 3
    num_eval_games: int = 20
    sideboard_eval_games: int = 1000
    final_eval_episodes: int = 0
    final_eval_max_steps: int = 0
    eval_episodes: int = 0
    eval_parallel_workers: int = 0
    eval_max_steps: int = 0
    eval_poll_seconds: int = 30


@dataclass(frozen=True)
class UnifiedRandomMatchupsDefaults:
    """``train_unified_random_matchups.py`` / fab_tui unified matchup wizard."""

    matchups: int
    episodes: int
    max_steps: int
    warmup_episodes: int
    warmup_baseline_eval_episodes: int
    checkpoint_interval_pct: float
    checkpoint_eval_episodes: int
    workers: int
    skip_converged: bool
    build_cpp_engine: bool
    require_cpp_engine: bool
    seed: int | None
    cache_dir: str | None
    out_dir: str | None
    custom_deck_links: tuple[str, ...]


# ─────────────────────────────────────────────────────────────────────────────
META = MetaRuntime()

@dataclass(frozen=True)
class RuntimeDefaults:
    meta: MetaRuntime
    game: GameControlsDefaults
    play: PlayDefaults
    sideboard_compare: SideboardCompareDefaults
    full_pipeline: FullPipelineDefaults
    matchup_sim: MatchupSimDefaults
    eval_dashboard: EvalDashboardDefaults
    dual_matchup: DualMatchupDefaults
    ppo: PpoDefaults
    tui: TuiDefaults
    unified_random_matchups: UnifiedRandomMatchupsDefaults

def _game_controls(meta: MetaRuntime) -> GameControlsDefaults:
    g = meta.game
    return GameControlsDefaults(
        stall_no_damage_turns=g.stall_no_damage_turns,
        stall_low_hand_turns=g.stall_low_hand_turns,
        stall_max_single_low_hand_turns=g.stall_max_single_low_hand_turns,
        stall_min_attack_hand=g.stall_min_attack_hand,
    )


def build_runtime(meta: MetaRuntime) -> RuntimeDefaults:
    """Derive per-script defaults from shared ``MetaRuntime`` settings."""
    dual_workers = 1 if meta.workers is None else meta.workers
    game = _game_controls(meta)
    urm = meta.unified_random_matchups
    urm_checkpoint_eval = urm.checkpoint_eval_episodes
    if urm_checkpoint_eval <= 0:
        urm_checkpoint_eval = min(100, max(1, urm.episodes // 100))
    urm_workers = urm.workers if urm.workers > 0 else dual_workers

    play = PlayDefaults(
        workers=meta.workers,
        parallel_seeds=meta.parallel_seeds,
        parallel_seeds_until_first_checkpoint=meta.parallel_seeds_until_first_checkpoint,
        checkpoint_interval_pct=meta.checkpoint_interval_pct,
        checkpoint_eval_episodes=meta.checkpoint_eval_episodes,
        play_checkpoint_interval=meta.play_checkpoint_interval,
        max_play_steps=meta.max_play_steps,
        warmup_episodes=meta.warmup_episodes,
        warmup_baseline_eval_episodes=meta.warmup_baseline_eval_episodes,
        gif_fps=meta.gif_fps,
    )

    return RuntimeDefaults(
        meta=meta,
        game=game,
        play=play,
        sideboard_compare=SideboardCompareDefaults(
            max_parallel=meta.sideboard_max_parallel,
            play_episodes=meta.play_episodes,
            final_eval_episodes=meta.final_eval_episodes,
            final_eval_max_steps=meta.final_eval_max_steps,
            dashboard_poll_seconds=meta.dashboard_poll_seconds,
        ),
        full_pipeline=FullPipelineDefaults(
            play_episodes=meta.play_episodes,
            final_eval_episodes=meta.final_eval_episodes,
            final_eval_max_steps=meta.final_eval_max_steps,
        ),
        matchup_sim=MatchupSimDefaults(
            play_episodes=meta.play_episodes,
            max_play_steps=meta.max_play_steps,
            final_eval_episodes=meta.final_eval_episodes,
            final_eval_max_steps=meta.final_eval_max_steps,
            warmup_episodes=meta.warmup_episodes,
            warmup_baseline_eval_episodes=meta.warmup_baseline_eval_episodes,
            gif_fps=meta.gif_fps_matchup_sim,
            workers=meta.workers,
        ),
        eval_dashboard=EvalDashboardDefaults(
            episodes=meta.eval_episodes,
            parallel_workers=meta.eval_parallel_workers,
            max_steps=meta.eval_max_steps,
            poll_seconds=meta.eval_poll_seconds,
            gif_fps=int(meta.gif_fps),
        ),
        dual_matchup=DualMatchupDefaults(
            episodes=meta.play_episodes,
            max_steps=meta.max_play_steps,
            workers=dual_workers,
        ),
        ppo=PpoDefaults(),
        tui=TuiDefaults(
            play_episodes=meta.play_episodes,
            final_eval_episodes=meta.final_eval_episodes,
            final_eval_max_steps=meta.final_eval_max_steps,
            eval_episodes=meta.eval_episodes,
            eval_parallel_workers=meta.eval_parallel_workers,
            eval_max_steps=meta.eval_max_steps,
            eval_poll_seconds=meta.eval_poll_seconds,
        ),
        unified_random_matchups=UnifiedRandomMatchupsDefaults(
            matchups=urm.matchups,
            episodes=urm.episodes,
            max_steps=urm.max_steps,
            warmup_episodes=urm.warmup_episodes,
            warmup_baseline_eval_episodes=meta.warmup_baseline_eval_episodes,
            checkpoint_interval_pct=urm.checkpoint_interval_pct,
            checkpoint_eval_episodes=urm_checkpoint_eval,
            workers=urm_workers,
            skip_converged=urm.skip_converged,
            build_cpp_engine=urm.build_cpp_engine,
            require_cpp_engine=urm.require_cpp_engine,
            seed=urm.seed,
            cache_dir=urm.cache_dir,
            out_dir=urm.out_dir,
            custom_deck_links=urm.custom_deck_links,
        ),
    )


RUNTIME = build_runtime(META)

# Flat aliases for argparse defaults and legacy imports.
DEFAULT_PARALLEL_SEEDS = RUNTIME.play.parallel_seeds
DEFAULT_CHECKPOINT_INTERVAL_PCT = RUNTIME.play.checkpoint_interval_pct
DEFAULT_CHECKPOINT_EVAL_EPISODES = RUNTIME.play.checkpoint_eval_episodes
DEFAULT_WARMUP_EPISODES = RUNTIME.play.warmup_episodes
DEFAULT_WARMUP_BASELINE_EVAL_EPISODES = RUNTIME.play.warmup_baseline_eval_episodes

DEFAULT_STALL_NO_DAMAGE_TURNS = RUNTIME.game.stall_no_damage_turns
DEFAULT_STALL_LOW_HAND_TURNS = RUNTIME.game.stall_low_hand_turns
DEFAULT_STALL_MAX_SINGLE_LOW_HAND_TURNS = RUNTIME.game.stall_max_single_low_hand_turns
DEFAULT_STALL_MIN_ATTACK_HAND = RUNTIME.game.stall_min_attack_hand

DEFAULT_N_EPISODES = RUNTIME.dual_matchup.episodes
DEFAULT_HIDDEN_SIZE = RUNTIME.ppo.hidden_size
DEFAULT_N_LAYERS = RUNTIME.ppo.n_layers
DEFAULT_N_HEADS = RUNTIME.ppo.n_heads
DEFAULT_LR = RUNTIME.ppo.lr
DEFAULT_GAMMA = RUNTIME.ppo.gamma
DEFAULT_LAM = RUNTIME.ppo.lam
DEFAULT_CLIP_EPS = RUNTIME.ppo.clip_eps
DEFAULT_N_STEPS = RUNTIME.ppo.n_steps
DEFAULT_PPO_EPOCHS = RUNTIME.ppo.ppo_epochs
DEFAULT_MINI_BATCH = RUNTIME.ppo.mini_batch
DEFAULT_PPO_ROLLOUT_BATCH = RUNTIME.ppo.rollout_batch

_UR = RUNTIME.unified_random_matchups
DEFAULT_UNIFIED_MATCHUPS = _UR.matchups
DEFAULT_UNIFIED_EPISODES = _UR.episodes
DEFAULT_UNIFIED_MAX_STEPS = _UR.max_steps
DEFAULT_UNIFIED_WARMUP_EPISODES = _UR.warmup_episodes
DEFAULT_UNIFIED_CHECKPOINT_INTERVAL_PCT = _UR.checkpoint_interval_pct
DEFAULT_UNIFIED_CHECKPOINT_EVAL_EPISODES = _UR.checkpoint_eval_episodes
DEFAULT_UNIFIED_WORKERS = _UR.workers


def apply_meta(**overrides: object) -> RuntimeDefaults:
    """Rebuild ``RUNTIME`` after in-place ``META`` edits (mainly for tests)."""
    global RUNTIME, META  # noqa: PLW0603
    global DEFAULT_PARALLEL_SEEDS, DEFAULT_CHECKPOINT_INTERVAL_PCT  # noqa: PLW0603
    global DEFAULT_CHECKPOINT_EVAL_EPISODES, DEFAULT_WARMUP_EPISODES  # noqa: PLW0603
    global DEFAULT_WARMUP_BASELINE_EVAL_EPISODES, DEFAULT_N_EPISODES  # noqa: PLW0603
    global DEFAULT_STALL_NO_DAMAGE_TURNS, DEFAULT_STALL_LOW_HAND_TURNS  # noqa: PLW0603
    global DEFAULT_STALL_MAX_SINGLE_LOW_HAND_TURNS, DEFAULT_STALL_MIN_ATTACK_HAND  # noqa: PLW0603
    global DEFAULT_HIDDEN_SIZE, DEFAULT_N_LAYERS, DEFAULT_N_HEADS, DEFAULT_LR, DEFAULT_GAMMA, DEFAULT_LAM  # noqa: PLW0603
    global DEFAULT_CLIP_EPS, DEFAULT_N_STEPS, DEFAULT_PPO_EPOCHS  # noqa: PLW0603
    global DEFAULT_MINI_BATCH, DEFAULT_PPO_ROLLOUT_BATCH  # noqa: PLW0603
    global DEFAULT_UNIFIED_MATCHUPS, DEFAULT_UNIFIED_EPISODES  # noqa: PLW0603
    global DEFAULT_UNIFIED_MAX_STEPS, DEFAULT_UNIFIED_WARMUP_EPISODES  # noqa: PLW0603
    global DEFAULT_UNIFIED_CHECKPOINT_INTERVAL_PCT  # noqa: PLW0603
    global DEFAULT_UNIFIED_CHECKPOINT_EVAL_EPISODES, DEFAULT_UNIFIED_WORKERS  # noqa: PLW0603

    META = replace(META, **overrides)
    RUNTIME = build_runtime(META)

    DEFAULT_PARALLEL_SEEDS = RUNTIME.play.parallel_seeds
    DEFAULT_CHECKPOINT_INTERVAL_PCT = RUNTIME.play.checkpoint_interval_pct
    DEFAULT_CHECKPOINT_EVAL_EPISODES = RUNTIME.play.checkpoint_eval_episodes
    DEFAULT_WARMUP_EPISODES = RUNTIME.play.warmup_episodes
    DEFAULT_WARMUP_BASELINE_EVAL_EPISODES = RUNTIME.play.warmup_baseline_eval_episodes
    DEFAULT_STALL_NO_DAMAGE_TURNS = RUNTIME.game.stall_no_damage_turns
    DEFAULT_STALL_LOW_HAND_TURNS = RUNTIME.game.stall_low_hand_turns
    DEFAULT_STALL_MAX_SINGLE_LOW_HAND_TURNS = RUNTIME.game.stall_max_single_low_hand_turns
    DEFAULT_STALL_MIN_ATTACK_HAND = RUNTIME.game.stall_min_attack_hand
    DEFAULT_N_EPISODES = RUNTIME.dual_matchup.episodes
    DEFAULT_HIDDEN_SIZE = RUNTIME.ppo.hidden_size
    DEFAULT_N_LAYERS = RUNTIME.ppo.n_layers
    DEFAULT_N_HEADS = RUNTIME.ppo.n_heads
    DEFAULT_LR = RUNTIME.ppo.lr
    DEFAULT_GAMMA = RUNTIME.ppo.gamma
    DEFAULT_LAM = RUNTIME.ppo.lam
    DEFAULT_CLIP_EPS = RUNTIME.ppo.clip_eps
    DEFAULT_N_STEPS = RUNTIME.ppo.n_steps
    DEFAULT_PPO_EPOCHS = RUNTIME.ppo.ppo_epochs
    DEFAULT_MINI_BATCH = RUNTIME.ppo.mini_batch
    DEFAULT_PPO_ROLLOUT_BATCH = RUNTIME.ppo.rollout_batch
    _ur = RUNTIME.unified_random_matchups
    DEFAULT_UNIFIED_MATCHUPS = _ur.matchups
    DEFAULT_UNIFIED_EPISODES = _ur.episodes
    DEFAULT_UNIFIED_MAX_STEPS = _ur.max_steps
    DEFAULT_UNIFIED_WARMUP_EPISODES = _ur.warmup_episodes
    DEFAULT_UNIFIED_CHECKPOINT_INTERVAL_PCT = _ur.checkpoint_interval_pct
    DEFAULT_UNIFIED_CHECKPOINT_EVAL_EPISODES = _ur.checkpoint_eval_episodes
    DEFAULT_UNIFIED_WORKERS = _ur.workers
    return RUNTIME
