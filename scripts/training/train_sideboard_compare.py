#!/usr/bin/env python3
"""Compare sideboard variants by training play agents against a fixed opponent.

Given a starting deck and registered card pool, this script:

1. Builds sideboard candidates using :class:`SideboardGuidePolicy` heuristics
   (baseline deck, full guide sideboard, and ranked 1-for-1 swaps).
2. Trains a play agent for each candidate (``--max-parallel`` at a time).
3. Runs a greedy **final eval** per candidate and reports win-rate deltas vs baseline.
4. Picks the deck with the best final eval win rate vs the configured opponent.

Typical usage::

    python scripts/training/train_sideboard_compare.py \\
        --starting-deck path/to/aurora.json \\
        --opponent-hero-id briar \\
        --opponent-deck BriarSAGEPrecon \\
        --num-options 4 \\
        --max-parallel 2 \\
        --play-episodes 500 \\
        --out-dir results/sideboard_compare_aurora_vs_briar
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
if str(_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_RL_SRC))

from flesh_and_blood_rlbridge.opponent_deck import normalize_talishar_asset_name  # noqa: E402

from cpp_engine_matchup import ensure_cpp_engine_for_matchup  # noqa: E402
from train_pipeline_common import (  # noqa: E402
    DEFAULT_AGENT_CACHE_DIR,
    DEFAULT_EQUIPMENT_HEADER,
    DEFAULT_FORMAT,
    DEFAULT_HERO_CLASS,
    DEFAULT_HERO_ID,
    DEFAULT_OPPONENT_DECK,
    DEFAULT_OPPONENT_HERO,
    OUT_DIR,
    PhaseAgents,
    SideboardCandidate,
    _load_starting_deck,
    _write_deck_file,
    ensure_pool_metadata,
    generate_sideboard_candidates,
    load_deck_and_pool_from_json,
    load_sideboard_candidates_from_json,
    min_deck_size_for_format,
    resolve_assets_path,
    save_deck_state,
)
from train_play import (  # noqa: E402
    DEFAULT_CHECKPOINT_EVAL_EPISODES,
    DEFAULT_CHECKPOINT_INTERVAL_PCT,
    DEFAULT_PARALLEL_SEEDS,
    DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
    DEFAULT_WARMUP_EPISODES,
    auto_detect_workers,
    resolve_play_checkpoint_interval,
    run_final_evaluation,
    run_phase3_play,
)


def _baseline_final_eval_win_rate(results: list[dict[str, Any]]) -> Optional[float]:
    for row in results:
        if row.get("candidate_id") == "baseline":
            rate = row.get("final_eval_win_rate")
            if rate is not None:
                return float(rate)
    for row in results:
        rate = row.get("final_eval_win_rate")
        if rate is not None:
            return float(rate)
    return None


def _attach_final_eval_deltas(results: list[dict[str, Any]]) -> None:
    baseline_wr = _baseline_final_eval_win_rate(results)
    for row in results:
        fe_wr = row.get("final_eval_win_rate")
        if fe_wr is None or baseline_wr is None:
            continue
        row["final_eval_delta_vs_baseline"] = float(fe_wr) - baseline_wr


def _rank_sideboard_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
        fe = row.get("final_eval_win_rate")
        train = float(row.get("play_win_rate", 0.0))
        primary = float(fe) if fe is not None else train
        return (-primary, -train, str(row.get("candidate_id", "")))

    return sorted(results, key=sort_key)


def _print_final_eval_comparison(
    results: list[dict[str, Any]],
    *,
    baseline_wr: Optional[float],
) -> None:
    has_final = any(row.get("final_eval_win_rate") is not None for row in results)
    if not has_final:
        return

    print(f"\n{'='*62}\n  Final eval comparison\n{'='*62}")
    print(
        f"  {'Candidate':<14}  {'Train':>7}  {'Final':>7}  {'Δ vs base':>10}  Label"
    )
    print(f"  {'-'*14}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*20}")
    for row in results:
        train = float(row.get("play_win_rate", 0.0))
        final = row.get("final_eval_win_rate")
        delta = row.get("final_eval_delta_vs_baseline")
        final_txt = f"{float(final):.1%}" if final is not None else "n/a"
        if delta is not None:
            delta_txt = f"{float(delta):+.1%}"
        elif baseline_wr is not None and final is not None:
            delta_txt = f"{float(final) - baseline_wr:+.1%}"
        else:
            delta_txt = "n/a"
        marker = " ←" if row.get("candidate_id") == results[0].get("candidate_id") else ""
        print(
            f"  {row.get('candidate_id', ''):<14}  "
            f"{train:>6.1%}  {final_txt:>7}  {delta_txt:>10}  "
            f"{row.get('label', '')}{marker}"
        )


def _train_candidate(
    candidate: SideboardCandidate,
    *,
    out_dir: Path,
    card_pool: dict[str, int],
    opponent_hero_id: str,
    args: argparse.Namespace,
    assets_path: str,
    play_workers: int,
) -> dict[str, Any]:
    """Train play on one sideboard candidate; return result summary."""
    candidate_dir = out_dir / "candidates" / candidate.candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)

    agents = PhaseAgents(
        player="p1",
        equipment_header=args.equipment_header,
        card_pool=dict(card_pool),
        active_decks={opponent_hero_id: dict(candidate.game_deck)},
    )

    print(
        f"\n  [{candidate.candidate_id}] {candidate.label}\n"
        f"    deck size={sum(candidate.game_deck.values())}  "
        f"out={candidate_dir}"
    )

    p1_wr, _ = run_phase3_play(
        agents,
        None,
        opponent_mode="preset",
        game_format=args.format,
        p1_hero_id=args.hero_id,
        p2_hero_id=args.p2_hero_id,
        p1_equipment_header=args.equipment_header,
        p2_equipment_header=args.p2_equipment_header,
        p1_opponent_hero_id=opponent_hero_id,
        p1_opponent_deck_name=args.opponent_deck,
        n_episodes=args.play_episodes,
        max_play_steps=args.max_play_steps,
        warmup_episodes=min(
            args.warmup_episodes,
            max(1, math.ceil(args.play_episodes / 10)),
            args.play_episodes,
        ),
        warmup_baseline_eval_episodes=args.warmup_baseline_eval_episodes,
        n_workers=play_workers,
        assets_path=assets_path,
        base_url=args.talishar_url,
        out_dir=candidate_dir,
        cache_dir=Path(args.cache_dir),
        seed=args.seed,
        cpp_engine_dir=args.cpp_engine_dir,
        checkpoint_interval=args.play_checkpoint_interval,
        checkpoint_interval_pct=args.checkpoint_interval_pct,
        checkpoint_eval_episodes=args.checkpoint_eval_episodes,
        parallel_batch_size=args.play_batch_size or play_workers,
        parallel_seeds=args.parallel_seeds,
    )

    checkpoint_history_path = candidate_dir / "checkpoint_eval_history.json"
    checkpoint_eval_history: list[dict[str, Any]] = []
    if checkpoint_history_path.is_file():
        try:
            checkpoint_eval_history = json.loads(
                checkpoint_history_path.read_text(encoding="utf-8")
            )
        except Exception:
            checkpoint_eval_history = []

    result = {
        "candidate_id": candidate.candidate_id,
        "label": candidate.label,
        "swaps": [list(pair) for pair in candidate.swaps],
        "guide_margin": candidate.guide_margin,
        "game_deck_size": sum(candidate.game_deck.values()),
        "play_win_rate": float(p1_wr),
        "checkpoint_eval_history": checkpoint_eval_history,
        "out_dir": str(candidate_dir),
    }
    if checkpoint_eval_history:
        result["latest_checkpoint_eval_win_rate"] = float(
            checkpoint_eval_history[-1].get("p1_win_rate", 0.0)
        )
    (candidate_dir / "candidate_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    cache_hit_path = candidate_dir / "play_cache_hit.json"
    if cache_hit_path.is_file():
        try:
            cache_hit = json.loads(cache_hit_path.read_text(encoding="utf-8"))
            result["training_skipped"] = bool(cache_hit.get("skipped_training"))
            result["cache_hit"] = cache_hit
            (candidate_dir / "candidate_result.json").write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    return result, agents


def _final_eval_candidate(
    agents: PhaseAgents,
    candidate: SideboardCandidate,
    *,
    opponent_hero_id: str,
    args: argparse.Namespace,
    assets_path: str,
    candidate_dir: Path,
) -> dict[str, Any]:
    """Run greedy final eval for a trained candidate."""
    eval_dir = candidate_dir / "final_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n  [{candidate.candidate_id}] Final eval "
        f"({args.final_eval_episodes} games)…"
    )
    summary = run_final_evaluation(
        agents,
        None,
        hero_id=args.hero_id,
        equipment_header=args.equipment_header,
        game_format=args.format,
        opponent_deck_name=args.opponent_deck,
        opponent_hero_id=opponent_hero_id,
        opponent_mode="preset",
        num_eval_episodes=args.final_eval_episodes,
        max_steps=args.final_eval_max_steps,
        assets_path=assets_path,
        base_url=args.talishar_url,
        fe_url=args.talishar_fe_url,
        out_dir=eval_dir,
        render_gif=not args.no_render_gif,
        gif_fps=args.gif_fps,
    )
    if summary.get("skipped"):
        return {"skipped": True, "reason": summary.get("reason", "unknown")}

    eval_block = summary.get("eval") or {}
    return {
        "win_rate": float(eval_block.get("win_rate", 0.0)),
        "wins": int(eval_block.get("wins", 0)),
        "losses": int(eval_block.get("losses", 0)),
        "draws": int(eval_block.get("draws", 0)),
        "episodes": int(eval_block.get("episodes", args.final_eval_episodes)),
        "json": str(eval_dir / f"{agents.player}_final_eval.json"),
    }


def _train_and_eval_candidate(
    candidate: SideboardCandidate,
    *,
    out_dir: Path,
    card_pool: dict[str, int],
    opponent_hero_id: str,
    args: argparse.Namespace,
    assets_path: str,
    play_workers: int,
) -> dict[str, Any]:
    result, agents = _train_candidate(
        candidate,
        out_dir=out_dir,
        card_pool=card_pool,
        opponent_hero_id=opponent_hero_id,
        args=args,
        assets_path=assets_path,
        play_workers=play_workers,
    )
    if args.skip_final_eval:
        return result

    candidate_dir = out_dir / "candidates" / candidate.candidate_id
    final_eval = _final_eval_candidate(
        agents,
        candidate,
        opponent_hero_id=opponent_hero_id,
        args=args,
        assets_path=assets_path,
        candidate_dir=candidate_dir,
    )
    if not final_eval.get("skipped"):
        result["final_eval_win_rate"] = final_eval["win_rate"]
        result["final_eval"] = final_eval
        (candidate_dir / "candidate_result.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sideboard variants via parallel play training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--format", default=DEFAULT_FORMAT,
        choices=["silver_age", "classic_constructed", "blitz", "upf"])
    parser.add_argument("--hero-id", default=DEFAULT_HERO_ID)
    parser.add_argument("--hero-class", default=DEFAULT_HERO_CLASS)
    parser.add_argument("--equipment-header", default=DEFAULT_EQUIPMENT_HEADER)
    parser.add_argument("--opponent-deck", default=DEFAULT_OPPONENT_DECK)
    parser.add_argument("--opponent-hero-id", default=DEFAULT_OPPONENT_HERO)
    parser.add_argument("--p2-hero-id", default="dorinthea_ironsong")
    parser.add_argument("--p2-equipment-header",
        default="dorinthea_ironsong dori_equipment_sword dori_equipment_sword "
                "helm_of_avarice gauntlet_of_might ironrot_legs valor_boots")

    parser.add_argument("--starting-deck", required=True,
        help="JSON deck export (deck + optional sideboard inventory)")
    parser.add_argument("--card-pool", default=None,
        help="Optional JSON with full registered pool (overrides sideboard key)")
    parser.add_argument("--num-options", type=int, default=4,
        help="Total sideboard variants to compare (including baseline)")
    parser.add_argument("--max-parallel", type=int, default=2,
        help="How many candidates to train/evaluate concurrently")
    parser.add_argument("--no-baseline", action="store_true",
        help="Skip the unmodified starting deck candidate")
    parser.add_argument("--no-guide-full", action="store_true",
        help="Skip the full SideboardGuidePolicy deck candidate")
    parser.add_argument("--min-swap-margin", type=float, default=0.75,
        help="Minimum guide score margin for ranked swap candidates")
    parser.add_argument("--candidates-json", default=None,
        help="Pre-built candidate manifest from the TUI sideboard picker")

    parser.add_argument("--play-episodes", type=int, default=500)
    parser.add_argument("--play-checkpoint-interval", type=int, default=None,
        help="Fixed checkpoint interval in episodes (default: 10%% of --play-episodes)")
    parser.add_argument("--checkpoint-interval-pct", type=float,
        default=DEFAULT_CHECKPOINT_INTERVAL_PCT,
        help="Checkpoint every N%% of play episodes when interval is unset")
    parser.add_argument("--checkpoint-eval-episodes", type=int,
        default=DEFAULT_CHECKPOINT_EVAL_EPISODES,
        help="C++ eval games vs opponent agent (greedy) at each checkpoint (0=off)")
    parser.add_argument("--max-play-steps", type=int, default=200)
    parser.add_argument("--warmup-episodes", type=int, default=DEFAULT_WARMUP_EPISODES)
    parser.add_argument("--warmup-baseline-eval-episodes", type=int,
        default=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES)
    parser.add_argument("--workers", type=int, default=None,
        help="Parallel C++ sessions per candidate (auto-detected when omitted)")
    parser.add_argument("--play-batch-size", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_AGENT_CACHE_DIR),
        help="Shared PPO + episode cache root (reused across sideboard runs)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--parallel-seeds",
        type=int,
        default=DEFAULT_PARALLEL_SEEDS,
        help=(
            "Independent RNG seeds per candidate; training win%% is averaged "
            f"and best P1/P2 agents are used for eval (default: "
            f"{DEFAULT_PARALLEL_SEEDS}; use 1 to disable)"
        ),
    )

    parser.add_argument("--out-dir", default=str(OUT_DIR / "sideboard_compare"))
    parser.add_argument("--talishar-url",
        default=os.environ.get("TALISHAR_URL", "http://localhost:8080/game"))
    parser.add_argument("--cpp-engine-dir", default=None)
    parser.add_argument("--assets-path", default=os.environ.get("TALISHAR_ASSETS_PATH", ""))
    parser.add_argument(
        "--no-build-cpp-engine",
        action="store_true",
        help="Do not auto-build a C++ engine when none is cached",
    )
    parser.add_argument("--final-eval-episodes", type=int, default=50,
        help="Greedy final eval games per candidate after training")
    parser.add_argument("--final-eval-max-steps", type=int, default=200)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--talishar-fe-url",
        default=os.environ.get("TALISHAR_FE_URL", "http://localhost:5173"))
    parser.add_argument("--no-render-gif", action="store_true",
        help="Skip GIF render during final eval")
    parser.add_argument("--gif-fps", type=float, default=3.0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_path = resolve_assets_path(args.assets_path or None)
    args.opponent_deck = normalize_talishar_asset_name(args.opponent_deck, assets_path)
    min_size = min_deck_size_for_format(args.format)

    game_deck, card_pool = load_deck_and_pool_from_json(
        args.starting_deck,
        card_pool_path=args.card_pool,
    )
    if not game_deck:
        game_deck = _load_starting_deck(args.starting_deck) or {}
    if not card_pool:
        card_pool = dict(game_deck)

    if sum(game_deck.values()) < min_size:
        print(
            f"ERROR: starting deck has {sum(game_deck.values())} cards; "
            f"need at least {min_size} for {args.format}"
        )
        raise SystemExit(1)
    if sum(card_pool.values()) < sum(game_deck.values()):
        print("ERROR: card pool is smaller than the starting deck")
        raise SystemExit(1)

    pool_agents = PhaseAgents(player="p1", card_pool=dict(card_pool))
    ensure_pool_metadata(
        pool_agents,
        hero_id=args.hero_id,
        hero_class=args.hero_class,
        game_format=args.format,
    )

    if not args.no_build_cpp_engine:
        args.cpp_engine_dir = ensure_cpp_engine_for_matchup(
            args.hero_id,
            args.opponent_deck,
            assets_path=assets_path,
            talishar_url=args.talishar_url,
            deck1_json=Path(args.starting_deck),
            build=True,
        ) or args.cpp_engine_dir

    total_workers = args.workers
    if total_workers is None:
        total_workers = auto_detect_workers(
            hero_id=args.hero_id,
            p2_hero_id=args.opponent_hero_id,
            cpp_engine_dir=args.cpp_engine_dir,
            assets_path=assets_path,
        )
    max_parallel = max(1, args.max_parallel)
    workers_per_candidate = max(1, total_workers // max_parallel)
    resolved_ckpt_interval = resolve_play_checkpoint_interval(
        args.play_episodes,
        checkpoint_interval=args.play_checkpoint_interval,
        checkpoint_interval_pct=args.checkpoint_interval_pct,
    )

    if args.candidates_json:
        candidates, card_pool = load_sideboard_candidates_from_json(
            args.candidates_json,
            card_pool=card_pool,
            min_deck_size=min_size,
        )
    else:
        candidates = generate_sideboard_candidates(
            card_pool,
            game_deck,
            args.opponent_hero_id,
            pool_agents.pool_by_id,
            hero_id=args.hero_id,
            game_format=args.format,
            num_options=max(1, args.num_options),
            include_baseline=not args.no_baseline,
            include_guide_full=not args.no_guide_full,
            min_swap_margin=args.min_swap_margin,
        )
    if not candidates:
        print("ERROR: no sideboard candidates generated")
        raise SystemExit(1)

    print(
        f"\n{'='*62}\n"
        f"  Sideboard Compare — Play Training\n"
        f"  Hero: {args.hero_id}  vs  {args.opponent_hero_id} ({args.opponent_deck})\n"
        f"  Candidates: {len(candidates)}  |  parallel={max_parallel}  "
        f"|  workers/candidate={workers_per_candidate}\n"
        f"  Play episodes/candidate: {args.play_episodes}\n"
        f"  Checkpoints: every {resolved_ckpt_interval} ep  |  "
        f"eval {args.checkpoint_eval_episodes} games @ opponent agent (greedy)\n"
        f"  Agent cache : {args.cache_dir}\n"
        f"{'='*62}"
    )
    for candidate in candidates:
        print(f"    • {candidate.candidate_id}: {candidate.label}")

    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "format": args.format,
        "hero_id": args.hero_id,
        "opponent_hero_id": args.opponent_hero_id,
        "opponent_deck": args.opponent_deck,
        "num_options": args.num_options,
        "max_parallel": max_parallel,
        "play_episodes": args.play_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "skip_final_eval": bool(args.skip_final_eval),
        "checkpoint_interval": resolved_ckpt_interval,
        "checkpoint_eval_episodes": args.checkpoint_eval_episodes,
        "parallel_seeds": args.parallel_seeds,
        "cpp_engine_dir": args.cpp_engine_dir,
        "candidates": [asdict(c) for c in candidates],
    }
    (out_dir / "candidates_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    for batch_start in range(0, len(candidates), max_parallel):
        batch = candidates[batch_start: batch_start + max_parallel]
        batch_no = batch_start // max_parallel + 1
        print(
            f"\n{'#'*62}\n"
            f"  BATCH {batch_no} — training {len(batch)} candidate(s) in parallel\n"
            f"{'#'*62}"
        )

        if len(batch) == 1:
            results.append(
                _train_and_eval_candidate(
                    batch[0],
                    out_dir=out_dir,
                    card_pool=card_pool,
                    opponent_hero_id=args.opponent_hero_id,
                    args=args,
                    assets_path=assets_path,
                    play_workers=workers_per_candidate,
                )
            )
            continue

        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {
                pool.submit(
                    _train_and_eval_candidate,
                    candidate,
                    out_dir=out_dir,
                    card_pool=card_pool,
                    opponent_hero_id=args.opponent_hero_id,
                    args=args,
                    assets_path=assets_path,
                    play_workers=workers_per_candidate,
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
        print("ERROR: all candidate trainings failed")
        raise SystemExit(1)

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
        win_rates=[float(winner["play_win_rate"])],
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
    winner_agents.deck_asset_name = deck_name

    summary = {
        "winner": winner,
        "ranking": results,
        "baseline_final_eval_win_rate": baseline_wr,
        "deck_state": str(out_dir / "deck_state.json"),
        "winning_deck_asset": deck_name,
    }
    summary_path = out_dir / "sideboard_compare_results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*62}\n  Sideboard comparison complete\n{'='*62}")
    print("\n  Ranking (training win rate):")
    for rank, row in enumerate(
        sorted(results, key=lambda r: (-float(r["play_win_rate"]), r["candidate_id"])),
        start=1,
    ):
        marker = "  ← best train" if row["candidate_id"] == winner["candidate_id"] and winner.get("final_eval_win_rate") is None else ""
        print(
            f"    {rank}. {row['candidate_id']:>12}  "
            f"{row['play_win_rate']:.1%}  {row['label']}{marker}"
        )
    _print_final_eval_comparison(results, baseline_wr=baseline_wr)
    if winner.get("final_eval_win_rate") is not None:
        print(
            f"\n  Winner (final eval): {winner['candidate_id']}  "
            f"{float(winner['final_eval_win_rate']):.1%}"
        )
    print(f"\n  Results → {summary_path}")
    print(f"  Winning deck asset → {deck_name}.txt")
    print(f"  Deck state → {out_dir / 'deck_state.json'}")


if __name__ == "__main__":
    main()
