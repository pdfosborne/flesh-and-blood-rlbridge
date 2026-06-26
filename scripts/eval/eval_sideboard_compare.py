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
from cpp_engine_matchup import ensure_cpp_engine_for_matchup  # noqa: E402
from flesh_and_blood_rlbridge.opponent_deck import normalize_talishar_asset_name  # noqa: E402
from flesh_and_blood_rlbridge.player_observation import (  # noqa: E402
    ACTION_CAPACITY,
    PLAYER_OBS_DIM,
    PLAYER_OBS_SCHEMA_VERSION,
)
from rl_agents.ppo import PPOAgent  # noqa: E402
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
from train_play import evaluate_fixed_matchup  # noqa: E402
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
        store.cache_root / f"unified_agent_v{PLAYER_OBS_SCHEMA_VERSION}.json"
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

    cpp_dir = ensure_cpp_engine_for_matchup(
        deck_name,
        opp_name,
        assets_path=assets_path,
        talishar_url=args.talishar_url,
        build=not args.no_build_cpp_engine,
    )
    if cpp_dir is None and not args.no_require_cpp_engine:
        raise RuntimeError(
            f"C++ engine required for {candidate.candidate_id} but unavailable"
        )

    matchup = Matchup(
        name=f"{candidate.candidate_id}-vs-{opp_name}",
        p1_deck=deck_name,
        p2_deck=opp_name,
        description=f"{candidate.label} eval vs {opp_name}",
        tags=[candidate.candidate_id, args.format],
        p1_hero=args.hero_id.replace("_", "-"),
        p2_hero=args.opponent_hero_id.replace("_", "-"),
        cpp_engine_dir=cpp_dir,
    )

    print(
        f"\n  [{candidate.candidate_id}] {candidate.label}\n"
        f"    C++ eval: {args.cpp_eval_episodes} ep  |  "
        f"Talishar eval: {args.talishar_eval_episodes} ep"
    )

    cpp_metrics: dict[str, Any] = {}
    if args.cpp_eval_episodes > 0:
        cpp_metrics = evaluate_fixed_matchup(
            matchup,
            agent,
            base_url=args.talishar_url,
            game_format=args.format,
            max_steps=args.max_eval_steps,
            episodes=args.cpp_eval_episodes,
            seed=args.seed,
            backend="cpp" if cpp_dir else "auto",
        )

    talishar_metrics: dict[str, Any] = {}
    if args.talishar_eval_episodes > 0:
        talishar_metrics = evaluate_fixed_matchup(
            matchup,
            agent,
            base_url=args.talishar_url,
            game_format=args.format,
            max_steps=args.max_eval_steps,
            episodes=args.talishar_eval_episodes,
            seed=(args.seed + 10_000) if args.seed is not None else None,
            backend="http",
        )
        eval_dir = candidate_dir / "final_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
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
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    cpp_wr = float(cpp_metrics.get("p1_win_rate", 0.0)) if cpp_metrics else None
    tal_wr = float(talishar_metrics.get("p1_win_rate", 0.0)) if talishar_metrics else None

    result = {
        "candidate_id": candidate.candidate_id,
        "label": candidate.label,
        "swaps": [list(pair) for pair in candidate.swaps],
        "guide_margin": candidate.guide_margin,
        "game_deck_size": sum(candidate.game_deck.values()),
        "play_win_rate": float(cpp_wr or 0.0),
        "cpp_eval_win_rate": cpp_wr,
        "cpp_eval": cpp_metrics,
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
        else f"    → done"
    )
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
    parser.add_argument("--no-require-cpp-engine", action="store_true")
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
    probe_cpp = ensure_cpp_engine_for_matchup(
        probe_deck,
        opp_name,
        assets_path=assets_path,
        talishar_url=args.talishar_url,
        build=not args.no_build_cpp_engine,
    )
    probe_matchup = Matchup(
        name="probe",
        p1_deck=probe_deck,
        p2_deck=opp_name,
        description="probe",
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
        "cpp_eval_episodes": args.cpp_eval_episodes,
        "talishar_eval_episodes": args.talishar_eval_episodes,
        "final_eval_episodes": args.talishar_eval_episodes,
        "final_eval_max_steps": args.max_eval_steps,
        "skip_final_eval": args.talishar_eval_episodes <= 0,
        "max_parallel": max_parallel,
        "candidates": [asdict(c) for c in candidates],
    }
    if existing_gui_run_id:
        manifest["gui_run_id"] = existing_gui_run_id
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
