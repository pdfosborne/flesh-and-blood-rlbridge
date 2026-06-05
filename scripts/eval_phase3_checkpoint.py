#!/usr/bin/env python3
"""Evaluate saved phase-3 play checkpoints with a live terminal dashboard.

The script auto-discovers the latest checkpoint package produced by
scripts/train_full_pipeline.py, reconstructs the saved matchup from the
checkpoint metadata, runs evaluation episodes, and optionally renders one
optimal-policy rollout GIF.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()

for path in (SRC_DIR, RL_SRC, Path(__file__).resolve().parent):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flesh_and_blood_rlbridge import TalisharEngineEnvironment  # noqa: E402
from rlbridge.rl_agents.ppo import PPOAgent  # noqa: E402
from train_full_pipeline import _frames_to_gif, _render_game_with_talishar_frontend, _write_deck_file  # noqa: E402


@dataclass
class CheckpointBundle:
    role: str
    checkpoint_dir: Path
    metadata: dict[str, Any]
    weights_path: Path

    @property
    def matchup(self) -> str:
        return str(self.metadata.get("matchup", "unknown_matchup"))

    @property
    def episodes_completed(self) -> int:
        return int(self.metadata.get("episodes_completed", 0) or 0)

    @property
    def game_format(self) -> str:
        return str(self.metadata.get("game_format", "silver_age"))

    @property
    def deck_spec(self) -> dict[str, Any]:
        value = self.metadata.get("deck_spec", {})
        return value if isinstance(value, dict) else {}


def _load_checkpoint(checkpoint_dir: Path, role: str) -> Optional[CheckpointBundle]:
    meta_path = checkpoint_dir / "metadata.json"
    weights_path = checkpoint_dir / "weights" / "agent_weights.json"
    if not meta_path.is_file() or not weights_path.is_file():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return CheckpointBundle(
        role=role,
        checkpoint_dir=checkpoint_dir,
        metadata=metadata,
        weights_path=weights_path,
    )


def _latest_checkpoint(results_dir: Path, role: str) -> Optional[CheckpointBundle]:
    best: Optional[tuple[float, int, CheckpointBundle]] = None
    for meta_path in results_dir.glob(f"p3_*/{role}/episode_*/metadata.json"):
        bundle = _load_checkpoint(meta_path.parent, role)
        if bundle is None:
            continue
        score = (meta_path.stat().st_mtime, bundle.episodes_completed)
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], bundle)
    return best[2] if best is not None else None


def _paired_checkpoint(bundle: CheckpointBundle, role: str) -> Optional[CheckpointBundle]:
    sibling = bundle.checkpoint_dir.parents[1] / role / bundle.checkpoint_dir.name
    return _load_checkpoint(sibling, role)


def _load_agent(weights_path: Path) -> PPOAgent:
    agent = PPOAgent()
    agent.load(str(weights_path))
    return agent


def _deck_cards(bundle: CheckpointBundle) -> dict[str, int]:
    cards = bundle.deck_spec.get("cards", {})
    if not isinstance(cards, dict):
        return {}
    return {str(card_id): int(count) for card_id, count in cards.items()}


def _equipment_header(bundle: CheckpointBundle) -> str:
    return str(bundle.deck_spec.get("equipment_header", "") or "")


def _print_dashboard(
    *,
    bundle: CheckpointBundle,
    opponent_label: str,
    episode: int,
    total_episodes: int,
    wins: int,
    losses: int,
    draws: int,
    last_outcome: str,
    last_steps: int,
    eval_dir: Path,
) -> None:
    total_done = max(1, episode)
    win_rate = wins / total_done
    print("\033[2J\033[H", end="")
    print("=" * 72)
    print("  Phase 3 Eval Dashboard")
    print("=" * 72)
    print(f"  Checkpoint   : {bundle.checkpoint_dir}")
    print(f"  Matchup      : {bundle.matchup}")
    print(f"  Format       : {bundle.game_format}")
    print(f"  Train eps    : {bundle.episodes_completed}/{bundle.metadata.get('target_episodes', '?')}")
    print(f"  Opponent     : {opponent_label}")
    print(f"  Eval dir     : {eval_dir}")
    print("-" * 72)
    print(f"  Episode      : {episode}/{total_episodes}")
    print(f"  Record       : {wins}W  {losses}L  {draws}D")
    print(f"  Win %        : {win_rate * 100:6.2f}%")
    print(f"  Last result  : {last_outcome}  ({last_steps} steps)")
    print("=" * 72)


def _evaluate_checkpoint(
    *,
    p1_bundle: CheckpointBundle,
    p2_bundle: Optional[CheckpointBundle],
    episodes: int,
    max_steps: int,
    base_url: str,
    fe_url: str,
    assets_path: str,
    render_gif: bool,
    gif_fps: float,
    seed: Optional[int],
) -> dict[str, Any]:
    p1_agent = _load_agent(p1_bundle.weights_path)
    p2_agent = _load_agent(p2_bundle.weights_path) if p2_bundle is not None else None

    p1_cards = _deck_cards(p1_bundle)
    if not p1_cards:
        raise RuntimeError(f"Checkpoint missing P1 deck spec: {p1_bundle.checkpoint_dir}")

    eval_dir = p1_bundle.checkpoint_dir / "eval_dashboard"
    eval_dir.mkdir(parents=True, exist_ok=True)

    p1_deck_name = f"eval_p1_{uuid.uuid4().hex[:8]}"
    p1_deck_file = _write_deck_file(p1_cards, _equipment_header(p1_bundle), p1_deck_name, assets_path)

    opponent_mode = str(p1_bundle.metadata.get("opponent_mode", "preset") or "preset")
    p2_deck_file: Optional[Path] = None
    cleanup_files = [p1_deck_file]
    if p2_bundle is not None and _deck_cards(p2_bundle):
        p2_deck_name = f"eval_p2_{uuid.uuid4().hex[:8]}"
        p2_deck_file = _write_deck_file(
            _deck_cards(p2_bundle),
            _equipment_header(p2_bundle),
            p2_deck_name,
            assets_path,
        )
        cleanup_files.append(p2_deck_file)
        opponent_label = f"paired {p2_bundle.role} checkpoint"
    elif opponent_mode == "mirror":
        p2_deck_name = p1_deck_name
        opponent_label = "mirror (same checkpoint)"
        p2_agent = p1_agent
    else:
        p2_deck_name = str(p1_bundle.metadata.get("opponent_deck_name", "Ira") or "Ira")
        opponent_label = f"preset deck {p2_deck_name}"

    wins = 0
    losses = 0
    draws = 0
    episode_log: list[dict[str, Any]] = []
    env = TalisharEngineEnvironment(
        base_url=base_url,
        game_format=p1_bundle.game_format,
        local_deck_name=p1_deck_name,
        opponent_deck_name=p2_deck_name,
        max_turns=max_steps,
        self_play=True,
        render_mode=None,
    )
    try:
        for ep in range(1, episodes + 1):
            ep_seed = (seed + ep) if seed is not None else None
            result = env.reset(seed=ep_seed)
            obs = result.observation
            steps = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
                acting = int(obs_data.get("actingPlayerID", 1) or 1)
                if acting == 1:
                    action = p1_agent.act_greedy(obs)
                elif p2_agent is not None:
                    action = p2_agent.act_greedy(obs)
                else:
                    action = env.sample_action()
                step = env.step(action)
                obs = step.observation
                steps += 1
                terminated = bool(step.terminated)
                truncated = bool(step.truncated)

            obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
            p1_hp = float(obs_data.get("playerHealth", 0) or 0)
            p2_hp = float(obs_data.get("opponentHealth", 0) or 0)
            if p1_hp > p2_hp:
                wins += 1
                outcome = "win"
            elif p2_hp > p1_hp:
                losses += 1
                outcome = "loss"
            else:
                draws += 1
                outcome = "draw"

            episode_log.append(
                {
                    "episode": ep,
                    "outcome": outcome,
                    "steps": steps,
                    "p1_hp": p1_hp,
                    "p2_hp": p2_hp,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            _print_dashboard(
                bundle=p1_bundle,
                opponent_label=opponent_label,
                episode=ep,
                total_episodes=episodes,
                wins=wins,
                losses=losses,
                draws=draws,
                last_outcome=outcome,
                last_steps=steps,
                eval_dir=eval_dir,
            )
    finally:
        env.close()

    gif_path: Optional[Path] = None
    render_dir: Optional[Path] = None
    try:
        if render_gif:
            render_dir = eval_dir / "optimal_policy_frames"
            gif_path = eval_dir / "optimal_policy.gif"
            render_opponent = SimpleNamespace(play=p2_agent)
            frame_paths = _render_game_with_talishar_frontend(
                agents=SimpleNamespace(play=p1_agent),
                opponent_agents=render_opponent,
                opponent_mode="dual",
                base_url=base_url,
                fe_url=fe_url,
                game_format=p1_bundle.game_format,
                deck_name=p1_deck_name,
                opp_name=p2_deck_name,
                max_steps=max_steps,
                render_dir=render_dir,
                player_label=p1_bundle.role,
            )
            if frame_paths:
                _frames_to_gif(frame_paths, gif_path, fps=gif_fps)
            else:
                gif_path = None
    finally:
        for file_path in cleanup_files:
            try:
                if file_path is not None and file_path.exists():
                    file_path.unlink(missing_ok=True)
            except Exception:
                pass

    summary = {
        "checkpoint_dir": str(p1_bundle.checkpoint_dir),
        "matchup": p1_bundle.matchup,
        "episodes_completed": p1_bundle.episodes_completed,
        "eval": {
            "episodes": episodes,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / max(1, episodes),
            "episode_log": episode_log,
        },
        "render": {
            "frames_dir": str(render_dir) if render_dir is not None else None,
            "gif": str(gif_path) if gif_path is not None else None,
        },
    }
    summary_path = eval_dir / "latest_eval.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation summary -> {summary_path}")
    if gif_path is not None:
        print(f"Saved optimal-policy GIF -> {gif_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate latest phase-3 checkpoint with a live terminal dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "full_pipeline"))
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Specific p1 checkpoint directory to evaluate.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--watch", action="store_true",
                        help="Keep watching results/ and re-evaluate when a new checkpoint appears.")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--no-render-gif", action="store_true")
    parser.add_argument("--gif-fps", type=float, default=3.0)
    parser.add_argument("--talishar-url", default=os.environ.get("TALISHAR_URL", "http://localhost:8080/game"))
    parser.add_argument("--talishar-fe-url", default=os.environ.get("TALISHAR_FE_URL", "http://localhost:5173"))
    parser.add_argument("--assets-path", default=os.environ.get("TALISHAR_ASSETS_PATH", ""))
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        raise SystemExit(f"results directory not found: {results_dir}")
    if not args.assets_path:
        raise SystemExit("TALISHAR_ASSETS_PATH or --assets-path is required")

    last_seen: Optional[Path] = None
    while True:
        if args.checkpoint_dir:
            p1_bundle = _load_checkpoint(Path(args.checkpoint_dir).expanduser().resolve(), "p1")
        else:
            p1_bundle = _latest_checkpoint(results_dir, "p1")

        if p1_bundle is None:
            print(f"Waiting for phase-3 p1 checkpoints under {results_dir}...")
        elif p1_bundle.checkpoint_dir != last_seen:
            p2_bundle = _paired_checkpoint(p1_bundle, "p2")
            _evaluate_checkpoint(
                p1_bundle=p1_bundle,
                p2_bundle=p2_bundle,
                episodes=args.episodes,
                max_steps=args.max_steps,
                base_url=args.talishar_url,
                fe_url=args.talishar_fe_url,
                assets_path=args.assets_path,
                render_gif=not args.no_render_gif,
                gif_fps=args.gif_fps,
                seed=args.seed,
            )
            last_seen = p1_bundle.checkpoint_dir

        if not args.watch:
            break
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()