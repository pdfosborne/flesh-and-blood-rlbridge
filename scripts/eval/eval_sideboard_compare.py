#!/usr/bin/env python3
"""Evaluate sideboard candidates with the unified agent — no PPO training.

Runs C++ fast eval (default 1000 games) then Talishar HTTP eval (default 10 games)
for each candidate deck vs a fixed opponent.

Usage:
    python scripts/eval/eval_sideboard_compare.py \\
        --starting-deck deck.json \\
        --opponent-deck KayoSAGEPrecon \\
        --candidates-json candidates_manifest.json \\
        --out-dir results/sideboard_compare/my_run
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

REPO_ROOT = _bootstrap.configure_paths()
_RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()
if _RL_SRC.is_dir() and str(_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_RL_SRC))

from agent_cache import AgentCacheStore  # noqa: E402
from cpp_engine_matchup import (  # noqa: E402
    ensure_cpp_engine_for_matchup,
    resolve_cpp_eval_deck_stems,
)
from flesh_and_blood_rlbridge.opponent_deck import normalize_talishar_asset_name  # noqa: E402
from flesh_and_blood_rlbridge.player_observation import (  # noqa: E402
    ACTION_CAPACITY,
    PLAYER_OBS_DIM,
    PLAYER_OBS_SCHEMA_VERSION,
)
from rl_agents.ppo import PPOAgent, UNIFIED_AGENT_WEIGHT_VERSION  # noqa: E402
from runtime_defaults import RUNTIME  # noqa: E402
from train_dual_agent_common import Matchup, make_env, _env_supports_fast_training  # noqa: E402
from train_pipeline_common import (  # noqa: E402
    DEFAULT_EQUIPMENT_HEADER,
    DEFAULT_FORMAT,
    DEFAULT_HERO_CLASS,
    DEFAULT_HERO_ID,
    DEFAULT_OPPONENT_DECK,
    DEFAULT_OPPONENT_HERO,
    PhaseAgents,
    SideboardCandidate,
    _write_deck_file,
    load_sideboard_candidates_from_json,
    resolve_assets_path,
    save_deck_state,
)
from train_play import (  # noqa: E402
    LOGIC_POLICY,
    evaluate_fixed_matchup,
    evaluate_policy_matchup,
)
from train_sideboard_compare import (  # noqa: E402
    _attach_final_eval_deltas,
    _baseline_final_eval_win_rate,
    _print_final_eval_comparison,
    _rank_sideboard_results,
    resolve_max_parallel,
)

DEFAULT_CPP_EVAL_EPISODES = 1000
DEFAULT_TALISHAR_EVAL_EPISODES = 10
DEFAULT_MAX_STEPS = RUNTIME.sideboard_compare.final_eval_max_steps or RUNTIME.play.max_play_steps

CPP_EVAL_VARIANTS: tuple[tuple[str, str, str, str], ...] = (
    ("logic_vs_logic", "logic", "logic", "C++ logic vs logic"),
    ("agent_vs_logic", "agent", "logic", "C++ agent vs logic"),
    ("logic_vs_agent", "logic", "agent", "C++ logic vs agent"),
    ("agent_vs_agent", "agent", "agent", "C++ agent vs agent"),
)


def _resolve_cpp_seat_policy(kind: str, agent: PPOAgent) -> Any:
    if kind == "logic":
        return LOGIC_POLICY
    return agent


def _load_unified_agent(
    cache_dir: Path,
    game_format: str,
    *,
    probe_env: Any,
) -> PPOAgent:
    store = AgentCacheStore(
        cache_dir,
        game_format,
        obs_schema_version=PLAYER_OBS_SCHEMA_VERSION,
    )
    if _env_supports_fast_training(probe_env):
        obs_dim = int(probe_env.fast_reset()["obs_vec"].shape[0])
        n_actions = int(probe_env.fast_action_capacity())
    else:
        reset = probe_env.reset()
        obs = reset.observation if hasattr(reset, "observation") else reset
        agent_probe = PPOAgent()
        obs_dim = agent_probe._obs_to_vec(obs).shape[0]
        n_actions = ACTION_CAPACITY

    weights_path = (
        store.cache_root / f"unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.json"
    )
    agent = store.load_required(
        obs_dim=obs_dim,
        n_actions=n_actions,
        mask_actions=True,
    )
    print(
        f"  Unified agent policy loaded from cache: {weights_path} "
        f"(obs_dim={obs_dim}, n_actions={n_actions})"
    )
    return agent


def _write_candidate_deck_json(candidate_dir: Path, game_deck: dict[str, int]) -> Path:
    """Write a minimal deck JSON for C++ engine generation."""
    path = candidate_dir / "game_deck.json"
    path.write_text(json.dumps({"deck": game_deck}, indent=2), encoding="utf-8")
    return path


def _precon_play_deck(assets_path: str, precon_stem: str) -> dict[str, int]:
    asset = Path(assets_path) / f"{precon_stem}.txt"
    if not asset.is_file():
        return {}
    lines = asset.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return {}
    counts: dict[str, int] = {}
    for card_id in lines[1].split():
        if card_id:
            counts[card_id] = counts.get(card_id, 0) + 1
    return counts


def _deck_signature(deck: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(k), int(v)) for k, v in deck.items() if int(v) > 0))


def _needs_custom_cpp_engine(
    assets_path: str,
    precon_stem: str,
    game_deck: dict[str, int],
    *,
    swaps: tuple[tuple[str, str], ...],
) -> bool:
    if swaps:
        return True
    return _deck_signature(game_deck) != _deck_signature(
        _precon_play_deck(assets_path, precon_stem)
    )


def _warn_cpp_eval_skipped(candidate_id: str, *, cpp_eval_episodes: int) -> None:
    if cpp_eval_episodes <= 0:
        return
    print(
        f"  NOTE: Skipping {cpp_eval_episodes} C++ eval game(s) for {candidate_id} "
        f"(no compiled engine) — Talishar HTTP eval only"
    )


def _refresh_dashboard(out_dir: Path) -> None:
    try:
        from sideboard_compare_dashboard import write_sideboard_compare_dashboard  # noqa: E402

        write_sideboard_compare_dashboard(out_dir, auto_refresh_seconds=5.0)
    except Exception:
        pass


def _write_eval_live(
    path: Path,
    *,
    phase: str,
    completed: int,
    target: int,
    wins: int = 0,
    losses: int = 0,
    draws: int = 0,
    timeouts: int = 0,
    runtime_backend: str = "",
    variant: str = "",
    p1_policy: str = "",
    p2_policy: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "episodes_completed": completed,
        "target_episodes": target,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "runtime_backend": runtime_backend,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if variant:
        payload["variant"] = variant
    if p1_policy:
        payload["p1_policy"] = p1_policy
    if p2_policy:
        payload["p2_policy"] = p2_policy
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_partial_candidate_result(
    candidate_dir: Path,
    *,
    candidate: SideboardCandidate,
    args: argparse.Namespace,
    play_win_rate: float | None = None,
    cpp_metrics: dict[str, Any] | None = None,
    cpp_eval_variants: dict[str, Any] | None = None,
    final_eval_win_rate: float | None = None,
    talishar_metrics: dict[str, Any] | None = None,
) -> None:
    result: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "label": candidate.label,
        "swaps": [list(pair) for pair in candidate.swaps],
        "guide_margin": candidate.guide_margin,
        "game_deck_size": sum(candidate.game_deck.values()),
        "eval_only": True,
        "progress_pct": 50.0 if play_win_rate is not None and final_eval_win_rate is None else 100.0,
        "out_dir": str(candidate_dir),
    }
    if play_win_rate is not None:
        result["play_win_rate"] = float(play_win_rate)
        result["cpp_eval_win_rate"] = float(play_win_rate)
    if cpp_metrics:
        result["cpp_eval"] = cpp_metrics
    if cpp_eval_variants:
        result["cpp_eval_variants"] = cpp_eval_variants
    if final_eval_win_rate is not None:
        result["final_eval_win_rate"] = float(final_eval_win_rate)
        result["final_eval"] = {
            "win_rate": float(final_eval_win_rate),
            "wins": int((talishar_metrics or {}).get("p1_wins", 0)),
            "losses": int((talishar_metrics or {}).get("losses", 0)),
            "draws": int((talishar_metrics or {}).get("draws", 0)),
            "episodes": args.talishar_eval_episodes,
            "runtime_backend": (talishar_metrics or {}).get("runtime_backend", "HTTP Talishar"),
        }
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "candidate_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


def _eval_candidate(
    candidate: SideboardCandidate,
    *,
    out_dir: Path,
    opponent_hero_id: str,
    args: argparse.Namespace,
    assets_path: str,
    unified_agent: PPOAgent,
) -> dict[str, Any]:
    from agent_cache import clone_agent_weights  # noqa: PLC0415

    agent = PPOAgent()
    clone_agent_weights(unified_agent, agent)
    candidate_dir = out_dir / "candidates" / candidate.candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    equipment_header = candidate.equipment_header or args.equipment_header

    deck_name = f"rl_eval_{candidate.candidate_id}_{uuid.uuid4().hex[:8]}"
    _write_deck_file(
        candidate.game_deck,
        equipment_header,
        deck_name,
        assets_path,
    )
    opp_name = normalize_talishar_asset_name(args.opponent_deck, assets_path)
    deck_json_path = _write_candidate_deck_json(candidate_dir, candidate.game_deck)
    cpp_deck1, cpp_deck2 = resolve_cpp_eval_deck_stems(
        assets_path,
        args.hero_id,
        opp_name,
    )
    # Reuse a Talishar precon engine when the list matches; otherwise build/find by JSON hash.
    deck_json_for_cpp = (
        deck_json_path
        if _needs_custom_cpp_engine(
            assets_path,
            cpp_deck1,
            candidate.game_deck,
            swaps=candidate.swaps,
        )
        else None
    )

    cpp_dir = ensure_cpp_engine_for_matchup(
        cpp_deck1,
        cpp_deck2,
        assets_path=assets_path,
        talishar_url=args.talishar_url,
        deck1_json=deck_json_for_cpp,
        build=not args.no_build_cpp_engine,
    )
    if cpp_dir is None:
        if args.require_cpp_engine and not args.no_require_cpp_engine:
            raise RuntimeError(
                f"C++ engine required for {candidate.candidate_id} but unavailable"
            )
        _warn_cpp_eval_skipped(
            candidate.candidate_id,
            cpp_eval_episodes=args.cpp_eval_episodes,
        )

    matchup = Matchup(
        name=f"{candidate.candidate_id}-vs-{opp_name}",
        p1_deck=deck_name,
        p2_deck=opp_name,
        description=f"{candidate.label} eval vs {opp_name}",
        tags=[candidate.candidate_id, args.format],
        p1_hero=args.hero_id.replace("_", "-"),
        p2_hero=args.opponent_hero_id.replace("_", "-"),
        cpp_engine_deck1=cpp_deck1,
        cpp_engine_deck2=cpp_deck2,
        cpp_engine_dir=cpp_dir,
    )

    eval_seed = 0 if args.seed is None else args.seed

    print(
        f"\n  [{candidate.candidate_id}] {candidate.label}\n"
        f"    C++ eval: {len(CPP_EVAL_VARIANTS)} policy matchups × "
        f"{args.cpp_eval_episodes if cpp_dir else 0} ep  |  "
        f"Talishar final eval: {args.talishar_eval_episodes} ep"
    )

    eval_live_path = candidate_dir / "eval_live.json"
    final_live_path = candidate_dir / "final_eval" / "final_eval_live.json"
    cpp_eval_dir = candidate_dir / "cpp_eval"
    _write_eval_live(
        eval_live_path,
        phase="cpp_checkpoint",
        completed=0,
        target=args.cpp_eval_episodes if cpp_dir else 0,
        runtime_backend="C++ engine",
    )
    _write_partial_candidate_result(candidate_dir, candidate=candidate, args=args)
    _refresh_dashboard(out_dir)

    cpp_variant_metrics: dict[str, Any] = {}
    cpp_metrics: dict[str, Any] = {}
    if cpp_dir and args.cpp_eval_episodes > 0:
        cpp_eval_dir.mkdir(parents=True, exist_ok=True)
        for variant_index, (variant_key, p1_kind, p2_kind, variant_label) in enumerate(
            CPP_EVAL_VARIANTS
        ):
            p1_policy = _resolve_cpp_seat_policy(p1_kind, agent)
            p2_policy = _resolve_cpp_seat_policy(p2_kind, agent)
            variant_seed = eval_seed + variant_index * 50_000
            print(f"  Running {variant_label} ({args.cpp_eval_episodes} games)…")
            _write_eval_live(
                eval_live_path,
                phase="cpp_checkpoint",
                completed=0,
                target=args.cpp_eval_episodes,
                runtime_backend="C++ engine",
                variant=variant_key,
                p1_policy=p1_kind,
                p2_policy=p2_kind,
            )
            _refresh_dashboard(out_dir)
            metrics = evaluate_policy_matchup(
                matchup,
                p1_policy,
                p2_policy,
                base_url=args.talishar_url,
                game_format=args.format,
                max_steps=args.max_eval_steps,
                episodes=args.cpp_eval_episodes,
                seed=variant_seed,
                backend="cpp",
                eval_label=variant_label,
                live_progress_path=eval_live_path,
                p1_deck_card_ids=set(candidate.game_deck.keys()),
            )
            cpp_variant_metrics[variant_key] = metrics
            (cpp_eval_dir / f"{variant_key}.json").write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )
            wr = float(metrics.get("p1_win_rate", 0.0))
            print(
                f"  {variant_label} done: {wr * 100:.1f}% "
                f"({metrics.get('p1_wins', 0)}W/"
                f"{metrics.get('losses', 0)}L/"
                f"{metrics.get('draws', 0)}D)"
            )
            _write_partial_candidate_result(
                candidate_dir,
                candidate=candidate,
                args=args,
                play_win_rate=float(
                    cpp_variant_metrics.get("agent_vs_agent", {}).get("p1_win_rate", wr)
                )
                if "agent_vs_agent" in cpp_variant_metrics
                else None,
                cpp_metrics=cpp_variant_metrics.get("agent_vs_agent"),
                cpp_eval_variants=cpp_variant_metrics,
            )
            _refresh_dashboard(out_dir)

        cpp_metrics = cpp_variant_metrics.get("agent_vs_agent", {})
        cpp_wr = float(cpp_metrics.get("p1_win_rate", 0.0)) if cpp_metrics else 0.0
        _write_eval_live(
            eval_live_path,
            phase="cpp_checkpoint",
            completed=args.cpp_eval_episodes,
            target=args.cpp_eval_episodes,
            wins=int(cpp_metrics.get("p1_wins", 0)),
            losses=int(cpp_metrics.get("losses", 0)),
            draws=int(cpp_metrics.get("draws", 0)),
            timeouts=int(cpp_metrics.get("timeouts", 0)),
            runtime_backend=str(cpp_metrics.get("runtime_backend", "C++ engine")),
            variant="agent_vs_agent",
            p1_policy="agent",
            p2_policy="agent",
        )
        _write_partial_candidate_result(
            candidate_dir,
            candidate=candidate,
            args=args,
            play_win_rate=cpp_wr,
            cpp_metrics=cpp_metrics,
            cpp_eval_variants=cpp_variant_metrics,
        )
        _refresh_dashboard(out_dir)
        print(
            f"  C++ agent vs agent done: {cpp_wr * 100:.1f}% "
            f"({cpp_metrics.get('p1_wins', 0)}W/"
            f"{cpp_metrics.get('losses', 0)}L/"
            f"{cpp_metrics.get('draws', 0)}D/"
            f"{cpp_metrics.get('timeouts', 0)}T/"
            f"{cpp_metrics.get('errors', 0)}E)"
        )
        if int(cpp_metrics.get("errors", 0)) > 0:
            print(
                f"  WARNING: C++ agent vs agent had "
                f"{cpp_metrics.get('errors', 0)} classification error(s)"
            )

    talishar_metrics: dict[str, Any] = {}
    if args.talishar_eval_episodes > 0:
        print(
            f"  Running Talishar final eval ({args.talishar_eval_episodes} games)… "
            f"(Talishar server required: {args.talishar_url})"
        )
        _write_eval_live(
            final_live_path,
            phase="final_eval",
            completed=0,
            target=args.talishar_eval_episodes,
            runtime_backend="HTTP Talishar",
        )
        _refresh_dashboard(out_dir)
        talishar_metrics = evaluate_fixed_matchup(
            matchup,
            agent,
            base_url=args.talishar_url,
            game_format=args.format,
            max_steps=args.max_eval_steps,
            episodes=args.talishar_eval_episodes,
            seed=eval_seed + 10_000,
            backend="http",
            eval_label="Talishar final eval",
            live_progress_path=final_live_path,
            p1_deck_card_ids=set(candidate.game_deck.keys()),
        )
        eval_dir = candidate_dir / "final_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        damage_breakdown = talishar_metrics.get("damage_breakdown")
        (eval_dir / "p1_final_eval.json").write_text(
            json.dumps(
                {
                    "eval": {
                        "win_rate": talishar_metrics.get("p1_win_rate", 0.0),
                        "wins": talishar_metrics.get("p1_wins", 0),
                        "losses": talishar_metrics.get("losses", 0),
                        "draws": talishar_metrics.get("draws", 0),
                        "episodes": args.talishar_eval_episodes,
                        "runtime_backend": talishar_metrics.get(
                            "runtime_backend", "HTTP Talishar"
                        ),
                    },
                    "analysis": (
                        {"damage_breakdown": damage_breakdown}
                        if damage_breakdown
                        else None
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tal_wr_done = float(talishar_metrics.get("p1_win_rate", 0.0))
        print(
            f"  Talishar final eval done: {tal_wr_done * 100:.1f}% "
            f"({talishar_metrics.get('p1_wins', 0)}W/"
            f"{talishar_metrics.get('losses', 0)}L/"
            f"{talishar_metrics.get('draws', 0)}D)"
        )

    cpp_wr = float(cpp_metrics.get("p1_win_rate", 0.0)) if cpp_metrics else None
    tal_wr = float(talishar_metrics.get("p1_win_rate", 0.0)) if talishar_metrics else None
    play_wr = float(cpp_wr if cpp_wr is not None else tal_wr or 0.0)

    result = {
        "candidate_id": candidate.candidate_id,
        "label": candidate.label,
        "swaps": [list(pair) for pair in candidate.swaps],
        "guide_margin": candidate.guide_margin,
        "game_deck_size": sum(candidate.game_deck.values()),
        "play_win_rate": play_wr,
        "cpp_eval_win_rate": cpp_wr,
        "cpp_eval": cpp_metrics,
        "cpp_eval_variants": cpp_variant_metrics or None,
        "final_eval_win_rate": tal_wr,
        "final_eval": (
            {
                "win_rate": tal_wr,
                "wins": int(talishar_metrics.get("p1_wins", 0)),
                "losses": int(talishar_metrics.get("losses", 0)),
                "draws": int(talishar_metrics.get("draws", 0)),
                "episodes": args.talishar_eval_episodes,
                "runtime_backend": talishar_metrics.get(
                    "runtime_backend", "HTTP Talishar"
                ),
                "damage_breakdown": talishar_metrics.get("damage_breakdown"),
            }
            if talishar_metrics
            else None
        ),
        "eval_only": True,
        "progress_pct": 100.0,
        "out_dir": str(candidate_dir),
    }
    (candidate_dir / "candidate_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        f"    → C++ {cpp_wr * 100:.1f}%  |  Talishar {tal_wr * 100:.1f}%"
        if cpp_wr is not None and tal_wr is not None
        else (
            f"    → Talishar {tal_wr * 100:.1f}%"
            if tal_wr is not None
            else "    → done"
        )
    )
    _refresh_dashboard(out_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate sideboard candidates with unified agent (no training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--format", default=DEFAULT_FORMAT,
        choices=["silver_age", "classic_constructed", "blitz", "upf"])
    parser.add_argument("--hero-id", default=DEFAULT_HERO_ID)
    parser.add_argument("--hero-class", default=DEFAULT_HERO_CLASS)
    parser.add_argument("--equipment-header", default=DEFAULT_EQUIPMENT_HEADER)
    parser.add_argument("--opponent-deck", default=DEFAULT_OPPONENT_DECK)
    parser.add_argument("--opponent-hero-id", default=DEFAULT_OPPONENT_HERO)
    parser.add_argument("--starting-deck", required=True)
    parser.add_argument("--candidates-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "results" / "agent_cache"))
    parser.add_argument("--cpp-eval-episodes", type=int, default=DEFAULT_CPP_EVAL_EPISODES)
    parser.add_argument("--talishar-eval-episodes", type=int, default=DEFAULT_TALISHAR_EVAL_EPISODES)
    parser.add_argument("--max-eval-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-parallel", type=int, default=RUNTIME.sideboard_compare.max_parallel)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--talishar-url", default=None)
    parser.add_argument("--assets-path", default=None)
    parser.add_argument("--no-build-cpp-engine", action="store_true")
    parser.add_argument(
        "--require-cpp-engine",
        action="store_true",
        help="Fail when the C++ engine cannot be built (default: HTTP Talishar fallback).",
    )
    parser.add_argument(
        "--no-require-cpp-engine",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    import os

    args.talishar_url = args.talishar_url or os.environ.get(
        "TALISHAR_URL", "http://localhost:8080/game"
    )
    assets_path = resolve_assets_path(args.assets_path)
    args.opponent_deck = normalize_talishar_asset_name(args.opponent_deck, assets_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    starting = Path(args.starting_deck)
    card_pool_data = json.loads(starting.read_text(encoding="utf-8"))
    game_deck = {str(k): int(v) for k, v in card_pool_data.get("deck", {}).items()}
    card_pool = {
        str(k): int(v)
        for k, v in (card_pool_data.get("sideboard") or card_pool_data.get("card_pool") or game_deck).items()
    }
    min_size = 40 if args.format in {"silver_age", "blitz", "sage"} else 60

    candidates, card_pool = load_sideboard_candidates_from_json(
        args.candidates_json,
        card_pool=card_pool,
        min_deck_size=min_size,
    )
    if not candidates:
        print("ERROR: no candidates in manifest")
        return 1

    max_parallel = resolve_max_parallel(args.max_parallel, len(candidates))

    probe_deck = f"rl_eval_probe_{uuid.uuid4().hex[:8]}"
    _write_deck_file(game_deck, args.equipment_header, probe_deck, assets_path)
    opp_name = normalize_talishar_asset_name(args.opponent_deck, assets_path)
    cpp_deck1, cpp_deck2 = resolve_cpp_eval_deck_stems(
        assets_path,
        args.hero_id,
        opp_name,
    )
    probe_cpp = ensure_cpp_engine_for_matchup(
        cpp_deck1,
        cpp_deck2,
        assets_path=assets_path,
        talishar_url=args.talishar_url,
        deck1_json=(
            starting
            if _needs_custom_cpp_engine(
                assets_path,
                cpp_deck1,
                game_deck,
                swaps=(),
            )
            else None
        ),
        build=not args.no_build_cpp_engine,
    )
    probe_matchup = Matchup(
        name="probe",
        p1_deck=probe_deck,
        p2_deck=opp_name,
        description="probe",
        cpp_engine_deck1=cpp_deck1,
        cpp_engine_deck2=cpp_deck2,
        cpp_engine_dir=probe_cpp,
    )
    probe_env = make_env(
        probe_matchup,
        base_url=args.talishar_url,
        game_format=args.format,
        max_turns=args.max_eval_steps,
    )
    try:
        unified_agent = _load_unified_agent(
            Path(args.cache_dir),
            args.format,
            probe_env=probe_env,
        )
    finally:
        probe_env.close()

    manifest_path = out_dir / "candidates_manifest.json"
    existing_gui_run_id: Optional[str] = None
    if manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_gui_run_id = prior.get("gui_run_id")
        except (OSError, json.JSONDecodeError):
            pass

    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "eval_only": True,
        "format": args.format,
        "hero_id": args.hero_id,
        "hero_class": args.hero_class,
        "equipment_header": args.equipment_header,
        "opponent_hero_id": args.opponent_hero_id,
        "opponent_deck": args.opponent_deck,
        "play_episodes": 0,
        "checkpoint_eval_episodes": args.cpp_eval_episodes,
        "cpp_eval_episodes": args.cpp_eval_episodes,
        "cpp_eval_variant_count": len(CPP_EVAL_VARIANTS),
        "talishar_eval_episodes": args.talishar_eval_episodes,
        "final_eval_episodes": args.talishar_eval_episodes,
        "final_eval_max_steps": args.max_eval_steps,
        "skip_final_eval": args.talishar_eval_episodes <= 0,
        "max_parallel": max_parallel,
        "candidates": [asdict(c) for c in candidates],
    }
    if probe_cpp:
        manifest["cpp_engine_dir"] = probe_cpp
    if existing_gui_run_id:
        manifest["gui_run_id"] = existing_gui_run_id
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _refresh_dashboard(out_dir)

    print(
        f"\n{'='*62}\n"
        f"  Sideboard Compare — Unified Agent Eval (no training)\n"
        f"  Hero: {args.hero_id}  vs  {args.opponent_hero_id} ({args.opponent_deck})\n"
        f"  Candidates: {len(candidates)}  |  parallel={max_parallel}\n"
        f"  C++ eval: {args.cpp_eval_episodes} ep/candidate  |  "
        f"Talishar eval: {args.talishar_eval_episodes} ep/candidate\n"
        f"{'='*62}"
    )

    results: list[dict[str, Any]] = []
    for batch_start in range(0, len(candidates), max_parallel):
        batch = candidates[batch_start : batch_start + max_parallel]
        if len(batch) == 1:
            results.append(
                _eval_candidate(
                    batch[0],
                    out_dir=out_dir,
                    opponent_hero_id=args.opponent_hero_id,
                    args=args,
                    assets_path=assets_path,
                    unified_agent=unified_agent,
                )
            )
            continue

        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {
                pool.submit(
                    _eval_candidate,
                    candidate,
                    out_dir=out_dir,
                    opponent_hero_id=args.opponent_hero_id,
                    args=args,
                    assets_path=assets_path,
                    unified_agent=unified_agent,
                ): candidate.candidate_id
                for candidate in batch
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"  ERROR: candidate {cid} failed: {exc!r}")

    if not results:
        print("ERROR: all candidate evaluations failed")
        return 1

    _attach_final_eval_deltas(results)
    results = _rank_sideboard_results(results)
    baseline_wr = _baseline_final_eval_win_rate(results)
    winner = results[0]

    winner_deck = next(
        c.game_deck for c in candidates if c.candidate_id == winner["candidate_id"]
    )
    winner_agents = PhaseAgents(
        player="p1",
        equipment_header=args.equipment_header,
        card_pool=dict(card_pool),
        active_decks={args.opponent_hero_id: dict(winner_deck)},
        last_play_win_rate=float(winner.get("final_eval_win_rate") or winner["play_win_rate"]),
        win_rates=[float(winner.get("play_win_rate", 0.0))],
        sideboard_converged=True,
    )
    save_deck_state(
        out_dir,
        p1=winner_agents,
        p2=None,
        game_format=args.format,
        opponent_mode="preset",
    )

    deck_name = f"rl_sb_compare_{winner['candidate_id']}"
    _write_deck_file(
        winner_agents.active_decks[args.opponent_hero_id],
        args.equipment_header,
        deck_name,
        assets_path,
    )

    summary = {
        "winner": winner,
        "ranking": results,
        "baseline_final_eval_win_rate": baseline_wr,
        "deck_state": str(out_dir / "deck_state.json"),
        "winning_deck_asset": deck_name,
        "eval_only": True,
    }
    summary_path = out_dir / "sideboard_compare_results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*62}\n  Evaluation complete\n{'='*62}")
    print("\n  Ranking (C++ eval win rate):")
    for rank, row in enumerate(
        sorted(results, key=lambda r: (-float(r.get("play_win_rate", 0.0)), r["candidate_id"])),
        start=1,
    ):
        cpp = row.get("cpp_eval_win_rate", row.get("play_win_rate"))
        print(
            f"    {rank}. {row['candidate_id']:>12}  "
            f"C++={float(cpp or 0.0):.1%}  {row['label']}"
        )
    _print_final_eval_comparison(results, baseline_wr=baseline_wr)
    print(f"\n  Results → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
