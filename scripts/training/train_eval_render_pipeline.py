#!/usr/bin/env python3
"""Run full Talishar pipeline: train -> evaluate -> optimal-policy render.

This script orchestrates existing training scripts, then evaluates the resulting
P1/P2 policies head-to-head, then renders one rollout using the better policy on
both sides and saves each state as an image.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import textwrap
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()
for p in (SRC_DIR, REPO_ROOT, RL_SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from flesh_and_blood_rlbridge.talishar_engine_environment import (  # noqa: E402
    TalisharEngineEnvironment,
    parse_acting_player_id,
)
from rl_agents.ppo import PPOAgent  # noqa: E402

from play_outcome_stats import (  # noqa: E402
    absolute_p1_p2_deck_from_env,
    absolute_p1_p2_deck_from_obs,
    absolute_p1_p2_hp_from_env,
    absolute_p1_p2_hp_from_obs,
    classify_p1_episode_outcome,
)


def _load_agent(weights_path: Path) -> PPOAgent:
    agent = PPOAgent()
    agent.load(str(weights_path))
    return agent


def _run_training(args: argparse.Namespace) -> None:
    script_map = {
        "sage-precons": REPO_ROOT / "scripts" / "training" / "train_sage_precons.py",
        "silver-age": REPO_ROOT / "scripts" / "training" / "train_silver_age_decks.py",
        "classic-constructed": REPO_ROOT / "scripts" / "training" / "train_classic_constructed_decks.py",
    }
    script_path = script_map[args.trainer]
    cmd = [
        sys.executable,
        str(script_path),
        "--episodes",
        str(args.episodes),
        "--matchup",
        args.matchup,
        "--max-steps",
        str(args.max_steps),
        "--format",
        args.format,
        "--out-dir",
        str(args.out_dir),
        "--cache-dir",
        str(args.cache_dir),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if bool(getattr(args, "show_frontend_training", False)):
        cmd.append("--show-frontend")
    frontend_url = getattr(args, "frontend_url", None)
    if bool(getattr(args, "show_frontend_training", False)) and not frontend_url:
        frontend_url = os.environ.get("TALISHAR_FE_URL") or "http://localhost:5173"
    if frontend_url:
        cmd.extend(["--frontend-url", str(frontend_url)])
    workers = int(getattr(args, "workers", 1))
    if workers > 1:
        cmd.extend(["--workers", str(workers)])

    print("\n[1/3] Training agents...")
    print(
        f"  Progress: running {args.episodes} episodes for matchup '{args.matchup}' "
        f"(trainer={args.trainer}, max_steps={args.max_steps})"
    )

    started = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
    try:
        heartbeat_every = 60.0
        next_heartbeat = heartbeat_every
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            elapsed = time.monotonic() - started
            if elapsed >= next_heartbeat:
                print(
                    f"  [train-progress] elapsed={elapsed:.0f}s "
                    "status=running (waiting for next episode completion log)"
                )
                next_heartbeat += heartbeat_every
            time.sleep(1.0)
    except Exception:
        if proc.poll() is None:
            proc.kill()
        raise

    if proc.returncode != 0:
        raise RuntimeError(f"Training script failed with exit code {proc.returncode}")

    elapsed_total = time.monotonic() - started
    print(f"  [train-progress] completed in {elapsed_total:.1f}s")


def _discover_agents(out_dir: Path, matchup: str) -> tuple[Path, Path, dict[str, Any]]:
    matchup_dir = out_dir / matchup
    if not matchup_dir.is_dir():
        raise RuntimeError(f"Matchup output directory not found: {matchup_dir}")

    newest_by_role: dict[str, tuple[float, Path, dict[str, Any]]] = {}
    for package_dir in matchup_dir.glob("ppo_*"):
        meta_path = package_dir / "metadata.json"
        weights_path = package_dir / "weights" / "agent_weights.json"
        if not meta_path.is_file() or not weights_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        role = str(meta.get("role", "")).strip().lower()
        if role not in {"p1", "p2"}:
            continue
        ts = package_dir.stat().st_mtime
        best = newest_by_role.get(role)
        if best is None or ts > best[0]:
            newest_by_role[role] = (ts, weights_path, meta)

    if "p1" not in newest_by_role or "p2" not in newest_by_role:
        raise RuntimeError(
            f"Could not find both p1 and p2 trained packages under {matchup_dir}"
        )

    p1_weights = newest_by_role["p1"][1]
    p2_weights = newest_by_role["p2"][1]
    meta = {
        "p1": newest_by_role["p1"][2],
        "p2": newest_by_role["p2"][2],
    }
    return p1_weights, p2_weights, meta


def _frames_to_gif(frame_paths: list[Path], gif_path: Path, fps: float = 4.0) -> None:
    """Assemble a list of PNG frame paths into an animated GIF."""
    from PIL import Image

    if not frame_paths:
        return
    frames: list[Image.Image] = []
    for p in frame_paths:
        try:
            frames.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    if not frames:
        return
    duration_ms = max(1, int(1000.0 / fps))
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
    )
    print(f"  Saved eval animation GIF ({len(frames)} frames) → {gif_path}")


def _eval_agents(
    base_url: str,
    p1_weights: Path,
    p2_weights: Path,
    p1_deck: str,
    p2_deck: str,
    game_format: str,
    episodes: int,
    max_steps: int,
    seed: int | None,
    *,
    show_frontend: bool = False,
    live_image_path: Path | None = None,
    gif_path: Path | None = None,
) -> dict[str, Any]:
    print("\n[2/3] Evaluating trained agents...")
    print(f"  Progress: {episodes} evaluation episodes (max_steps={max_steps})")
    p1_agent = _load_agent(p1_weights)
    p2_agent = _load_agent(p2_weights)

    need_render = show_frontend or gif_path is not None
    render_mode = "rgb_array" if need_render else None
    env = TalisharEngineEnvironment(
        base_url=base_url,
        local_deck_name=p1_deck,
        opponent_deck_name=p2_deck,
        game_format=game_format,
        self_play=True,
        max_turns=max_steps,
        render_mode=render_mode,
    )
    if show_frontend and live_image_path is not None:
        live_image_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Live eval image path → {live_image_path}")

    # Temporary directory for GIF frames (one per step across all episodes)
    gif_frame_paths: list[Path] = []
    gif_frames_dir: Path | None = None
    if gif_path is not None:
        import tempfile
        gif_frames_dir = gif_path.parent / "_gif_frames_tmp"
        gif_frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Eval GIF will be saved → {gif_path}")

    p1_wins = 0
    p2_wins = 0
    draws = 0
    timeouts = 0
    episodes_log: list[dict[str, Any]] = []
    try:
        for ep in range(1, episodes + 1):
            ep_seed = (seed + ep) if seed is not None else None
            reset_out = env.reset(seed=ep_seed)
            obs = reset_out.observation
            total_reward = 0.0
            terminated = False
            truncated = False
            steps = 0

            for step_no in range(1, max_steps + 1):
                acting = parse_acting_player_id(env, obs)
                agent = p1_agent if acting == 1 else p2_agent
                action = agent.act_greedy(obs)
                step = env.step(action)
                obs = step.observation
                total_reward += float(step.reward)
                terminated = bool(step.terminated)
                truncated = bool(step.truncated)
                steps = step_no
                if show_frontend and live_image_path is not None:
                    _save_state_image(env, obs, live_image_path)
                if gif_frames_dir is not None:
                    frame_path = gif_frames_dir / f"ep{ep:04d}_step{step_no:04d}.png"
                    _save_state_image(env, obs, frame_path)
                    gif_frame_paths.append(frame_path)
                if terminated or truncated:
                    break

            if not terminated and not truncated and steps >= max_steps:
                truncated = True

            if isinstance(obs, str):
                try:
                    obs_dict = json.loads(obs)
                except json.JSONDecodeError:
                    obs_dict = {}
            else:
                obs_dict = obs if isinstance(obs, dict) else {}

            p1_hp, p2_hp = absolute_p1_p2_hp_from_env(env)
            p1_deck, p2_deck = absolute_p1_p2_deck_from_env(env)
            if p1_hp is None or p2_hp is None:
                p1_hp_f, p2_hp_f = absolute_p1_p2_hp_from_obs(obs_dict)
                p1_hp = int(p1_hp_f) if p1_hp_f is not None else None
                p2_hp = int(p2_hp_f) if p2_hp_f is not None else None
            if p1_deck is None or p2_deck is None:
                p1_deck, p2_deck = absolute_p1_p2_deck_from_obs(obs_dict)
            outcome = classify_p1_episode_outcome(
                p1_hp=p1_hp,
                p2_hp=p2_hp,
                p1_deck=p1_deck,
                p2_deck=p2_deck,
                terminated=terminated,
                truncated=truncated,
            )
            if outcome == "win":
                p1_wins += 1
                winner = "p1"
            elif outcome == "loss":
                p2_wins += 1
                winner = "p2"
            elif outcome == "draw":
                draws += 1
                winner = "draw"
            else:
                timeouts += 1
                winner = "timeout"

            episodes_log.append(
                {
                    "episode": ep,
                    "steps": steps,
                    "terminated": terminated,
                    "truncated": truncated,
                    "total_reward": total_reward,
                    "winner": winner,
                }
            )
            pct = (ep / max(1, episodes)) * 100.0
            print(
                f"  [eval {ep:3d}/{episodes:3d} | {pct:6.2f}%] "
                f"winner={winner:<4} reward={total_reward:+.3f} steps={steps:3d}"
            )
    finally:
        env.close()

    # Build the GIF from collected frames
    if gif_path is not None and gif_frame_paths:
        _frames_to_gif(gif_frame_paths, gif_path)
        # Clean up temporary frame images
        import shutil
        if gif_frames_dir is not None and gif_frames_dir.is_dir():
            shutil.rmtree(gif_frames_dir, ignore_errors=True)

    total = max(1, episodes)
    summary = {
        "episodes": episodes,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "timeouts": timeouts,
        "p1_win_rate": p1_wins / total,
        "p2_win_rate": p2_wins / total,
        "timeout_rate": timeouts / total,
        "episodes_log": episodes_log,
        "eval_gif": str(gif_path) if gif_path is not None else None,
    }
    print(
        "  Eval summary: "
        f"p1={p1_wins}/{episodes}, p2={p2_wins}/{episodes}, "
        f"draws={draws}/{episodes}, timeouts={timeouts}/{episodes}"
    )
    return summary


def _state_text(obs: Any) -> str:
    if isinstance(obs, str):
        try:
            parsed = json.loads(obs)
        except json.JSONDecodeError:
            parsed = {"raw": obs}
    elif isinstance(obs, dict):
        parsed = obs
    else:
        parsed = {"raw": repr(obs)}

    lines: list[str] = ["Talishar State Snapshot"]
    lines.append(f"turnNo: {parsed.get('turnNo', '?')}")
    lines.append(f"turnPhase: {parsed.get('turnPhase', '?')}")
    lines.append(f"actingPlayerID: {parsed.get('actingPlayerID', '?')}")
    lines.append(f"playerHealth: {parsed.get('playerHealth', '?')}")
    lines.append(f"opponentHealth: {parsed.get('opponentHealth', '?')}")
    lines.append(f"legalActions: {len(parsed.get('legalActions', []) or [])}")
    prompt = parsed.get("prompt", "")
    if prompt:
        lines.append(f"prompt: {prompt}")

    raw_json = json.dumps(parsed, indent=2, ensure_ascii=False)
    lines.append("")
    lines.append("Raw observation JSON:")
    for ln in raw_json.splitlines():
        lines.extend(textwrap.wrap(ln, width=120) or [""])
    return "\n".join(lines)


def _save_rgb_frame(env: TalisharEngineEnvironment, out_path: Path) -> bool:
    rr = env.render()
    b64 = getattr(rr, "data", None)
    if not b64:
        return False
    out_path.write_bytes(base64.b64decode(b64))
    return True


def _save_text_fallback_image(obs: Any, out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    text = _state_text(obs)
    lines = text.splitlines()
    font = ImageFont.load_default()

    line_height = 14
    width = 1600
    height = max(300, 20 + line_height * (len(lines) + 2))
    img = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(img)
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=(235, 235, 235), font=font)
        y += line_height
    out_path.write_bytes(b"")
    img.save(out_path)


def _save_state_image(env: TalisharEngineEnvironment, obs: Any, out_path: Path) -> bool:
    if _save_rgb_frame(env, out_path):
        return True
    _save_text_fallback_image(obs, out_path)
    return True


def _render_optimal_policy(
    base_url: str,
    weights_path: Path,
    p1_deck: str,
    p2_deck: str,
    game_format: str,
    max_steps: int,
    seed: int | None,
    out_dir: Path,
) -> dict[str, Any]:
    print("\n[3/3] Rendering rollout with optimal policy...")
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = _load_agent(weights_path)
    env = TalisharEngineEnvironment(
        base_url=base_url,
        local_deck_name=p1_deck,
        opponent_deck_name=p2_deck,
        game_format=game_format,
        self_play=True,
        max_turns=max_steps,
        render_mode="rgb_array",
    )

    saved = 0
    terminated = False
    truncated = False
    try:
        reset = env.reset(seed=seed)
        obs = reset.observation
        if _save_state_image(env, obs, out_dir / "state_0000_reset.png"):
            saved += 1

        for step_no in range(1, max_steps + 1):
            action = agent.act_greedy(obs)
            step = env.step(action)
            obs = step.observation
            terminated = bool(step.terminated)
            truncated = bool(step.truncated)

            acting = parse_acting_player_id(env, obs)
            frame_name = f"state_{step_no:04d}_actingP{acting}.png"
            if _save_state_image(env, obs, out_dir / frame_name):
                saved += 1
            if terminated or truncated:
                break
    finally:
        env.close()

    summary = {
        "weights": str(weights_path),
        "frames_saved": saved,
        "terminated": terminated,
        "truncated": truncated,
    }
    print(f"  Saved {saved} state images to {out_dir}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full training -> evaluation -> optimal-policy render pipeline."
    )
    parser.add_argument(
        "--trainer",
        choices=["sage-precons", "silver-age", "classic-constructed"],
        default="sage-precons",
        help="Which existing training script family to run.",
    )
    parser.add_argument("--matchup", required=True, help="Matchup name to train/eval/render.")
    parser.add_argument("--episodes", type=int, default=300, help="Training episodes.")
    parser.add_argument("--max-steps", type=int, default=100, help="Training max steps.")
    parser.add_argument(
        "--show-frontend-training",
        action="store_true",
        help="Write a live training-state image during training stage (no browser tabs).",
    )
    parser.add_argument(
        "--show-frontend-eval",
        action="store_true",
        help="Write a live eval-state image during evaluation stage (overwrites same file each step).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel game sessions for training (default: 1). Set to 2-4 for speedup.",
    )
    parser.add_argument(
        "--frontend-url",
        default=None,
        help="Talishar FE URL override for training stage.",
    )
    parser.add_argument("--format", default="sage", help="Talishar game format.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "sage_precon_agents",
        help="Output root for trained agents.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "results" / "agent_cache",
        help="Cache directory for training tiers.",
    )
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes.")
    parser.add_argument("--eval-max-steps", type=int, default=100, help="Evaluation max steps.")
    parser.add_argument(
        "--render-max-steps",
        type=int,
        default=100,
        help="Max steps for the rendered rollout.",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=None,
        help="Directory to save per-state render images.",
    )
    parser.add_argument(
        "--eval-gif",
        type=Path,
        default=None,
        help=(
            "Path to save the optimal-policy animation GIF recorded during the eval step. "
            "Defaults to <out-dir>/<matchup>/eval_optimal_policy.gif when not provided."
        ),
    )
    args = parser.parse_args()

    base_url = os.environ.get("TALISHAR_URL", "http://localhost:8080/game")

    _run_training(args)
    p1_weights, p2_weights, meta = _discover_agents(args.out_dir, args.matchup)
    p1_deck = str(meta["p1"].get("p1_deck", ""))
    p2_deck = str(meta["p1"].get("p2_deck", ""))
    if not p1_deck or not p2_deck:
        raise RuntimeError("Could not infer p1/p2 deck names from training metadata.")

    eval_gif_path = args.eval_gif
    if eval_gif_path is None:
        eval_gif_path = args.out_dir / args.matchup / "eval_optimal_policy.gif"

    eval_summary = _eval_agents(
        base_url=base_url,
        p1_weights=p1_weights,
        p2_weights=p2_weights,
        p1_deck=p1_deck,
        p2_deck=p2_deck,
        game_format=args.format,
        episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        seed=args.seed,
        show_frontend=bool(getattr(args, "show_frontend_eval", False)),
        live_image_path=(
            args.out_dir / args.matchup / "eval_live_state.png"
            if getattr(args, "show_frontend_eval", False)
            else None
        ),
        gif_path=eval_gif_path,
    )

    optimal_role = "p1" if eval_summary["p1_wins"] >= eval_summary["p2_wins"] else "p2"
    optimal_weights = p1_weights if optimal_role == "p1" else p2_weights
    render_dir = args.render_dir
    if render_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        render_dir = args.out_dir / args.matchup / "optimal_policy_render" / ts

    render_summary = _render_optimal_policy(
        base_url=base_url,
        weights_path=optimal_weights,
        p1_deck=p1_deck,
        p2_deck=p2_deck,
        game_format=args.format,
        max_steps=args.render_max_steps,
        seed=args.seed,
        out_dir=render_dir,
    )

    pipeline_summary = {
        "trainer": args.trainer,
        "matchup": args.matchup,
        "base_url": base_url,
        "training": {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "out_dir": str(args.out_dir),
            "cache_dir": str(args.cache_dir),
            "p1_weights": str(p1_weights),
            "p2_weights": str(p2_weights),
        },
        "evaluation": eval_summary,
        "optimal_policy": {
            "role": optimal_role,
            "weights": str(optimal_weights),
        },
        "render": {
            "dir": str(render_dir),
            **render_summary,
        },
    }
    summary_path = args.out_dir / args.matchup / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(pipeline_summary, indent=2), encoding="utf-8")

    print("\nPipeline complete.")
    print(f"  Optimal policy : {optimal_role} ({optimal_weights})")
    print(f"  Eval GIF       : {eval_gif_path}")
    print(f"  Render dir     : {render_dir}")
    print(f"  Summary        : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
