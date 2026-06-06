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
from train_full_pipeline import (  # noqa: E402
    _ensure_playwright,
    _frames_to_gif,
    _save_state_image,
    _write_deck_file,
)


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

    @property
    def p1_hero(self) -> str:
        return str(self.metadata.get("p1_hero", "") or "")

    @property
    def p2_hero(self) -> str:
        return str(self.metadata.get("p2_hero", "") or "")

    @property
    def own_hero(self) -> str:
        """Hero ID for the side this checkpoint represents."""
        return self.p1_hero if self.role == "p1" else self.p2_hero

    @property
    def opponent_hero(self) -> str:
        """Hero ID of the opposing side."""
        return self.p2_hero if self.role == "p1" else self.p1_hero

    @property
    def raw_opponent_deck_name(self) -> str:
        """Raw ``opponent_deck_name`` stored in metadata (may be a UUID or preset name)."""
        return str(self.metadata.get("opponent_deck_name", "") or "")

    @property
    def preset_opponent_deck_name(self) -> str:
        """Cleaned preset deck name for P2, or ``""`` if the stored value is a
        UUID-based training artefact (``rl_p3_*``) or otherwise unusable."""
        name = self.raw_opponent_deck_name
        if not name or name.startswith("rl_p3_") or name in ("unknown", "Ira"):
            return ""
        return name


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
    """Find the paired *role* checkpoint for the same training run as *bundle*.

    Tries the exact same episode directory first (both roles save together at
    every checkpoint interval).  If that is missing, returns the *latest*
    checkpoint for *role* found anywhere in the same run directory so that a
    slightly-ahead or slightly-behind P2 is still discovered.
    """
    # Exact episode match: .../p3_xxx/p2/episode_010000/
    sibling = bundle.checkpoint_dir.parents[1] / role / bundle.checkpoint_dir.name
    result = _load_checkpoint(sibling, role)
    if result is not None:
        return result
    # Latest available checkpoint for *role* within the same run
    role_dir = bundle.checkpoint_dir.parents[1] / role
    if not role_dir.is_dir():
        return None
    best: Optional[tuple[float, int, CheckpointBundle]] = None
    for meta_path in role_dir.glob("episode_*/metadata.json"):
        candidate = _load_checkpoint(meta_path.parent, role)
        if candidate is None:
            continue
        score = (meta_path.stat().st_mtime, candidate.episodes_completed)
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], candidate)
    return best[2] if best is not None else None


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
    header = str(bundle.deck_spec.get("equipment_header", "") or "")
    # The hero ID must be the first token on the equipment header line so
    # Talishar knows which hero card to load.  Older checkpoints may store
    # the header without the hero ID prefix (e.g. just the equipment pieces).
    # Recover it from the p1_hero / p2_hero fields saved in metadata.
    hero_key = f"{bundle.role}_hero"           # "p1_hero" or "p2_hero"
    hero_id = (
        str(bundle.metadata.get(hero_key, "") or "")
        .replace("-", "_")                      # API stores dashes, header uses underscores
        .strip()
    )
    if hero_id and not header.startswith(hero_id):
        header = (hero_id + " " + header).strip()
    return header


def _print_dashboard(
    *,
    bundle: CheckpointBundle,
    opponent_label: str,
    episode: int,
    total_episodes: int,
    wins: int,
    losses: int,
    timeouts: int,
    last_outcome: str,
    last_steps: int,
    eval_dir: Path,
) -> None:
    total_done = max(1, episode)
    decided = max(1, total_done - timeouts)   # exclude timeouts from win-rate denominator
    win_rate_all = wins / total_done
    win_rate_dec = wins / decided
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
    # Show win-rate chart path if it already exists for this run.
    chart_path = bundle.checkpoint_dir.parent / "winrate_chart.png"
    if chart_path.is_file():
        print(f"  Win-rate chart: {chart_path}")
    print("-" * 72)
    print(f"  Episode      : {episode}/{total_episodes}")
    print(f"  Record       : {wins}W  {losses}L  {timeouts}T")
    print(f"  Win %        : {win_rate_all * 100:6.2f}%  "
          f"(excl. timeouts: {win_rate_dec * 100:.1f}%)")
    print(f"  Last result  : {last_outcome}  ({last_steps} steps)")
    print("=" * 72)


def _run_render_episode(
    *,
    p1_agent: Any,
    p2_agent: Any,
    base_url: str,
    fe_url: str,
    game_format: str,
    p1_deck_name: str,
    p2_deck_name: str,
    max_steps: int,
    render_dir: Path,
    player_label: str,
) -> tuple[list[Path], str]:
    """Run one Playwright render episode using render_mode='rgb_array'.

    Returns ``(frame_paths, outcome)`` where outcome is one of
    ``"win"``, ``"loss"``, ``"draw"``, or ``"timeout"``.
    """
    _ensure_playwright()
    render_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    outcome = "timeout"

    env = TalisharEngineEnvironment(
        base_url=base_url,
        frontend_url=fe_url,
        game_format=game_format,
        local_deck_name=p1_deck_name,
        opponent_deck_name=p2_deck_name,
        max_turns=max_steps,
        self_play=True,
        render_mode="rgb_array",
        use_cpp_engine=False,
    )
    try:
        result = env.reset()
        obs = result.observation

        frame_path = render_dir / "frame_0000_reset.png"
        if _save_state_image(env, obs, frame_path):
            frame_paths.append(frame_path)

        step_no = 0
        terminated = False
        truncated = False
        while not (terminated or truncated) and step_no < max_steps:
            step_no += 1
            obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
            acting = int(obs_data.get("actingPlayerID", 1) or 1)

            if acting == 1:
                action = (p1_agent.act_greedy(obs) if p1_agent and hasattr(p1_agent, "act_greedy")
                          else env.sample_action())
            else:
                action = (p2_agent.act_greedy(obs) if p2_agent and hasattr(p2_agent, "act_greedy")
                          else env.sample_action())

            step = env.step(action)
            obs = step.observation
            terminated = bool(step.terminated)
            truncated = bool(step.truncated)

            # Only capture P1's turn frames — the board is always shown from P1's perspective.
            if acting == 1:
                fpath = render_dir / f"frame_{step_no:04d}.png"
                if _save_state_image(env, obs, fpath):
                    frame_paths.append(fpath)

        if terminated:
            obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
            p1_hp = float(obs_data.get("playerHealth", 0) or 0)
            p2_hp = float(obs_data.get("opponentHealth", 0) or 0)
            if p1_hp > p2_hp:
                outcome = "win"
            elif p2_hp > p1_hp:
                outcome = "loss"
            else:
                outcome = "draw"
        else:
            outcome = "timeout"

    except Exception as exc:
        print(f"  [{player_label}] Render error: {exc}")
    finally:
        env.close()

    print(f"  [{player_label}] Render episode: {outcome}  ({len(frame_paths)} frames) → {render_dir}")
    return frame_paths, outcome


# ---------------------------------------------------------------------------
# Win-rate history + chart helpers
# ---------------------------------------------------------------------------

def _append_to_history(summary: dict[str, Any], history_path: Path) -> list[dict[str, Any]]:
    """Append this checkpoint's eval result to the persistent history file.

    Stored at the p1 run-level dir so all episode checkpoints in the same
    training run contribute to a single growing curve.
    """
    import datetime  # noqa: PLC0415

    history: list[dict[str, Any]] = []
    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            history = []

    ev = summary.get("eval", {})
    entry: dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "episodes_completed": summary.get("episodes_completed", 0),
        "checkpoint_dir": summary.get("checkpoint_dir", ""),
        "eval_episodes": ev.get("episodes", 0),
        "wins": ev.get("wins", 0),
        "losses": ev.get("losses", 0),
        "timeouts": ev.get("timeouts", 0),
        "win_rate": ev.get("win_rate", 0.0),
        "win_rate_decided": ev.get("win_rate_decided", 0.0),
    }

    # Overwrite if the same checkpoint is re-evaluated.
    history = [h for h in history if h.get("checkpoint_dir") != entry["checkpoint_dir"]]
    history.append(entry)
    history.sort(key=lambda h: int(h.get("episodes_completed", 0)))

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def _update_winrate_chart(
    history: list[dict[str, Any]],
    chart_path: Path,
    matchup: str = "",
) -> bool:
    """Regenerate the win-rate line chart from history. Returns True on success."""
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return False

    if not history:
        return False

    x = [int(h.get("episodes_completed", 0)) for h in history]
    y_all = [float(h.get("win_rate", 0.0)) * 100 for h in history]
    y_dec = [float(h.get("win_rate_decided", 0.0)) * 100 for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_all, marker="o", linewidth=2, label="Win % (all episodes)")
    ax.plot(x, y_dec, marker="s", linewidth=2, linestyle="--", label="Win % (excl. timeouts)")
    ax.axhline(50, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    ax.set_xlabel("Training episodes completed")
    y_axis_title = matchup.split(" vs ")[0] + " Win %"
    ax.set_ylabel(y_axis_title)
    ax.set_ylim(0, 100)
    title = f"Win Rate Tracker"
    if matchup:
        title += f"  ·  {matchup}"
    #title += f"  ·  {len(history)} checkpoint(s)"
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Annotate each data point with its win %.
    for xi, yi in zip(x, y_all):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(chart_path), dpi=120)
    plt.close(fig)
    return True


def _resolve_p2_preset_deck_name(
    p1_bundle: CheckpointBundle,
    p2_bundle: Optional[CheckpointBundle],
) -> str:
    """Return the best Talishar Assets preset deck name for P2.

    Used when P2 has no ``deck_spec.cards`` (preset or mirror training mode) or
    when no P2 checkpoint was found at all.

    Priority
    --------
    1. ``p1_bundle.preset_opponent_deck_name`` — the name stored during training
       (skipped when it is empty or a UUID artefact like ``rl_p3_p2_*``).
    2. ``p2_bundle.own_hero`` converted to ``{Hero}SAGEPrecon`` (e.g.
       ``"dorinthea"`` → ``"DorintheaSAGEPrecon"``).
    3. ``p1_bundle.p2_hero`` (the opponent hero recorded in P1's matchup
       metadata) converted the same way.
    4. ``"Ira"`` as a last resort.
    """
    # 1. Stored preset name from P1 metadata
    name = p1_bundle.preset_opponent_deck_name
    if name:
        return name
    # 2-3. Derive from hero ID
    hero = ""
    if p2_bundle is not None:
        hero = p2_bundle.own_hero
    if not hero:
        hero = p1_bundle.p2_hero
    if hero:
        # "dorinthea" / "dorinthea-young" → "Dorinthea" → "DorintheaSAGEPrecon"
        hero_clean = hero.replace("-", "_").split("_")[0].strip().capitalize()
        if hero_clean:
            return f"{hero_clean}SAGEPrecon"
    return "Ira"


def _prefer_non_pass_action(obs_data: dict[str, Any], fallback_action: Any) -> Any:
    """Choose a non-pass legal action index when available.

    The observation contains legal actions as ``[{index, label, zone}, ...]``.
    During chooser phases, greedy policies can repeatedly pick pass/confirm,
    which stalls progression. This helper picks the first clearly non-pass
    option when one exists.
    """
    legal = obs_data.get("legalActions")
    if not isinstance(legal, list) or len(legal) <= 1:
        return fallback_action

    non_pass: list[int] = []
    for entry in legal:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        if not isinstance(idx, int):
            continue
        label = str(entry.get("label", "") or "").strip().lower()
        if any(tok in label for tok in (
            "pass", "decline", "cancel", "skip", "no ", "undo",
        )):
            continue
        non_pass.append(idx)

    if not non_pass:
        return fallback_action
    return non_pass[0]


def _run_eval_episode_batch(
    *,
    episode_numbers: list[int],
    seed: Optional[int],
    base_url: str,
    game_format: str,
    p1_deck_name: str,
    p2_deck_name: str,
    max_steps: int,
    p1_weights_path: Path,
    p2_weights_path: Optional[Path],
    mirror_opponent: bool,
    stall_no_damage_turns: int,
    stall_low_hand_turns: int,
    stall_max_single_low_hand_turns: int,
    stall_min_attack_hand: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Evaluate a batch of episodes in one worker thread.

    One Talishar environment is reused per worker to reduce reset overhead.
    """
    if not episode_numbers:
        return []

    p1_agent = _load_agent(p1_weights_path)
    if mirror_opponent:
        p2_agent: Any = p1_agent
    elif p2_weights_path is not None:
        p2_agent = _load_agent(p2_weights_path)
    else:
        p2_agent = None

    env = TalisharEngineEnvironment(
        base_url=base_url,
        frontend_url=None,
        game_format=game_format,
        local_deck_name=p1_deck_name,
        opponent_deck_name=p2_deck_name,
        max_turns=max_steps,
        self_play=True,
        render_mode=None,
        use_cpp_engine=False,
        verbose=verbose,
    )
    logs: list[dict[str, Any]] = []
    _ANTI_STALL_STREAK = 5
    try:
        for ep in episode_numbers:
            ep_seed = (seed + ep) if seed is not None else None
            try:
                result = env.reset(seed=ep_seed)
            except Exception:
                logs.append(
                    {
                        "episode": ep,
                        "outcome": "error",
                        "steps": 0,
                        "p1_hp": 0.0,
                        "p2_hp": 0.0,
                        "terminated": False,
                        "truncated": True,
                        "stall_early_stop": False,
                    }
                )
                continue

            obs = result.observation
            steps = 0
            terminated = False
            truncated = False
            last_info: dict[str, Any] = {}
            stall_triggered = False

            init_obs = json.loads(obs) if isinstance(obs, str) else (obs or {})
            init_turn_no = int(init_obs.get("turnNo", 0) or 0)
            init_total_hp = float(init_obs.get("playerHealth", 0) or 0) + float(
                init_obs.get("opponentHealth", 0) or 0
            )
            turns_without_damage = 0
            turn_start_total_hp = init_total_hp
            last_seen_turn_no = init_turn_no
            low_hand_turn_streak: dict[int, int] = {1: 0, 2: 0}
            seen_main_phase_turns: set[tuple[int, int]] = set()

            while not (terminated or truncated):
                obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
                acting = int(obs_data.get("actingPlayerID", 1) or 1)
                phase = str(obs_data.get("turnPhase", "") or "")
                phase_norm = phase.strip().lower()
                turn_no = int(obs_data.get("turnNo", 0) or 0)

                total_hp = float(obs_data.get("playerHealth", 0) or 0) + float(
                    obs_data.get("opponentHealth", 0) or 0
                )
                if turn_no != last_seen_turn_no:
                    if total_hp >= turn_start_total_hp:
                        turns_without_damage += 1
                    else:
                        turns_without_damage = 0
                    turn_start_total_hp = total_hp
                    last_seen_turn_no = turn_no

                if phase_norm == "m":
                    marker = (acting, turn_no)
                    if marker not in seen_main_phase_turns:
                        seen_main_phase_turns.add(marker)
                        hand_size = int(obs_data.get("playerHandSize", 0) or 0)
                        if hand_size < stall_min_attack_hand:
                            low_hand_turn_streak[acting] = low_hand_turn_streak.get(acting, 0) + 1
                        else:
                            low_hand_turn_streak[acting] = 0

                both_low_streak = (
                    low_hand_turn_streak.get(1, 0) >= stall_low_hand_turns
                    and low_hand_turn_streak.get(2, 0) >= stall_low_hand_turns
                )
                one_sided_low_streak = (
                    max(low_hand_turn_streak.get(1, 0), low_hand_turn_streak.get(2, 0))
                    >= stall_max_single_low_hand_turns
                )
                if turns_without_damage >= stall_no_damage_turns and (
                    both_low_streak or one_sided_low_streak
                ):
                    stall_triggered = True
                    truncated = True
                    break

                repeat_streak = int(last_info.get("repeat_streak", 0))
                if repeat_streak >= _ANTI_STALL_STREAK:
                    action = _prefer_non_pass_action(obs_data, env.sample_action())
                elif acting == 1:
                    action = p1_agent.act_greedy(obs)
                elif p2_agent is not None:
                    action = p2_agent.act_greedy(obs)
                else:
                    action = env.sample_action()

                if phase in {
                    "MULTICHOOSEHAND",
                    "CHOOSEHAND",
                    "CHOOSEMULTIZONE",
                    "MAYCHOOSEMULTIZONE",
                    "CHOOSEOPTION",
                }:
                    action = _prefer_non_pass_action(obs_data, action)

                step = env.step(action)
                obs = step.observation
                last_info = step.info or {}
                steps += 1
                terminated = bool(step.terminated)
                truncated = bool(step.truncated)

            obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
            p1_hp = float(obs_data.get("playerHealth", 0) or 0)
            p2_hp = float(obs_data.get("opponentHealth", 0) or 0)
            if p1_hp <= 0 and p2_hp > 0:
                outcome = "loss"
            elif p2_hp <= 0 and p1_hp > 0:
                outcome = "win"
            else:
                outcome = "stall_timeout" if stall_triggered else "timeout"

            logs.append(
                {
                    "episode": ep,
                    "outcome": outcome,
                    "steps": steps,
                    "p1_hp": p1_hp,
                    "p2_hp": p2_hp,
                    "terminated": terminated,
                    "truncated": truncated,
                    "stall_early_stop": stall_triggered,
                }
            )
    finally:
        env.close()

    return logs


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
    render_max_steps: int,
    gif_fps: float,
    seed: Optional[int],
    stall_no_damage_turns: int,
    stall_low_hand_turns: int,
    stall_max_single_low_hand_turns: int,
    stall_min_attack_hand: int,
    parallel_workers: int,
    verbose: bool = False,
) -> dict[str, Any]:
    print(f"\n{'='*72}")
    print(f"  Evaluating checkpoint  : {p1_bundle.checkpoint_dir.name}")
    print(f"  Matchup                : {p1_bundle.matchup}")
    print(f"  Training episodes done : {p1_bundle.episodes_completed}")
    print(f"  Eval episodes          : {episodes}  |  max steps: {max_steps}")
    print(
        "  Stall guard            : "
        f"no-dmg-turns>={stall_no_damage_turns}, "
        f"low-hand<{stall_min_attack_hand} for "
        f"{stall_low_hand_turns}/{stall_max_single_low_hand_turns} turns"
    )
    if render_gif:
        print(f"  Render replay          : enabled (single episode, max steps: {render_max_steps})")
    print(f"  Talishar URL           : {base_url}")
    print(f"{'='*72}")

    print("  Loading agent weights...", flush=True)
    p1_agent = _load_agent(p1_bundle.weights_path)
    p2_agent = _load_agent(p2_bundle.weights_path) if p2_bundle is not None else None
    if p2_bundle is not None:
        print(f"  P1 weights : {p1_bundle.weights_path}")
        print(f"  P2 weights : {p2_bundle.weights_path}")
    else:
        print(f"  P1 weights : {p1_bundle.weights_path}")
        print("  P2 weights : (none found — will use heuristic)")

    p1_cards = _deck_cards(p1_bundle)
    if not p1_cards:
        raise RuntimeError(f"Checkpoint missing P1 deck spec: {p1_bundle.checkpoint_dir}")

    eval_dir = p1_bundle.checkpoint_dir.parent / "eval_dashboard"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("  Writing deck files...", flush=True)
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
        opponent_label = f"trained P2 agent — {p2_bundle.role} checkpoint ({p2_bundle.episodes_completed} eps)"
    elif opponent_mode == "mirror":
        p2_deck_name = p1_deck_name
        opponent_label = "mirror (same checkpoint)"
        p2_agent = p1_agent
    else:
        # Preset mode or dual mode where P2 deck cards weren't stored.
        # Resolve to a valid Talishar Assets deck name from hero metadata.
        p2_deck_name = _resolve_p2_preset_deck_name(p1_bundle, p2_bundle)
        if p2_agent is not None:
            opponent_label = (
                f"preset deck {p2_deck_name} "
                f"+ trained P2 agent ({p2_bundle.episodes_completed} eps)"  # type: ignore[union-attr]
            )
        else:
            opponent_label = f"preset deck {p2_deck_name} (P2 default heuristic)"

    print(f"  P1 deck      : {p1_deck_name}")
    print(f"  P2 deck      : {p2_deck_name}")
    print(f"  Opponent     : {opponent_label}")
    print(f"  Eval dir     : {eval_dir}", flush=True)

    wins = 0
    losses = 0
    timeouts = 0
    stall_timeouts = 0
    episode_log: list[dict[str, Any]] = []

    # Keep evaluation episodes fast: no frame rendering/capture during batch eval.
    env_render_mode = None
    env_fe_url = None

    # ── Pre-flight: verify Talishar is reachable before creating the env ──────
    # Eval always uses HTTP Talishar (not the C++ engine) so the full game
    # flow is exercised and the frontend render works.  Check connectivity
    # immediately with a short timeout so we fail fast with a clear message
    # instead of silently hanging for 90 s at env.reset().
    print(f"  Backend      : HTTP Talishar  ({base_url})", flush=True)
    import requests as _requests
    try:
        _requests.get(base_url.rstrip("/") + "/", timeout=5.0)
    except Exception as _conn_exc:
        raise RuntimeError(
            f"\n  Cannot reach Talishar at {base_url}\n"
            f"  Error: {_conn_exc}\n"
            "  Start the server first:  ./start_talishar.ps1\n"
            "  Or set TALISHAR_URL / --talishar-url to the correct address."
        ) from _conn_exc

    workers = max(1, min(int(parallel_workers), episodes))
    print(
        f"  Starting {episodes} evaluation episode(s) "
        f"with {workers} worker(s)...",
        flush=True,
    )
    all_logs: list[dict[str, Any]] = []
    mirror_opponent = opponent_mode == "mirror"
    p2_weights_path = p2_bundle.weights_path if p2_bundle is not None else None

    if workers == 1:
        all_logs = _run_eval_episode_batch(
            episode_numbers=list(range(1, episodes + 1)),
            seed=seed,
            base_url=base_url,
            game_format=p1_bundle.game_format,
            p1_deck_name=p1_deck_name,
            p2_deck_name=p2_deck_name,
            max_steps=max_steps,
            p1_weights_path=p1_bundle.weights_path,
            p2_weights_path=p2_weights_path,
            mirror_opponent=mirror_opponent,
            stall_no_damage_turns=stall_no_damage_turns,
            stall_low_hand_turns=stall_low_hand_turns,
            stall_max_single_low_hand_turns=stall_max_single_low_hand_turns,
            stall_min_attack_hand=stall_min_attack_hand,
            verbose=verbose,
        )
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

        batches: list[list[int]] = [[] for _ in range(workers)]
        for i, ep in enumerate(range(1, episodes + 1)):
            batches[i % workers].append(ep)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_eval_episode_batch,
                    episode_numbers=batch,
                    seed=seed,
                    base_url=base_url,
                    game_format=p1_bundle.game_format,
                    p1_deck_name=p1_deck_name,
                    p2_deck_name=p2_deck_name,
                    max_steps=max_steps,
                    p1_weights_path=p1_bundle.weights_path,
                    p2_weights_path=p2_weights_path,
                    mirror_opponent=mirror_opponent,
                    stall_no_damage_turns=stall_no_damage_turns,
                    stall_low_hand_turns=stall_low_hand_turns,
                    stall_max_single_low_hand_turns=stall_max_single_low_hand_turns,
                    stall_min_attack_hand=stall_min_attack_hand,
                    verbose=verbose,
                )
                for batch in batches
                if batch
            ]
            for future in as_completed(futures):
                all_logs.extend(future.result())

    all_logs.sort(key=lambda x: int(x.get("episode", 0)))
    for i, rec in enumerate(all_logs, start=1):
        outcome = str(rec.get("outcome", "timeout"))
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            timeouts += 1
            if outcome == "stall_timeout":
                stall_timeouts += 1
        episode_log.append(rec)
        _print_dashboard(
            bundle=p1_bundle,
            opponent_label=opponent_label,
            episode=i,
            total_episodes=episodes,
            wins=wins,
            losses=losses,
            timeouts=timeouts,
            last_outcome=outcome,
            last_steps=int(rec.get("steps", 0) or 0),
            eval_dir=eval_dir,
        )

    gif_path: Optional[Path] = None
    render_dir: Optional[Path] = None
    render_outcome: Optional[str] = None
    try:
        if render_gif:
            gif_path = eval_dir / "optimal_policy.gif"
            print("  [render] Evaluation complete; running a single dedicated replay episode…")
            render_dir = eval_dir / "optimal_policy_frames"
            frame_paths, render_outcome = _run_render_episode(
                p1_agent=p1_agent,
                p2_agent=p2_agent,
                base_url=base_url,
                fe_url=fe_url,
                game_format=p1_bundle.game_format,
                p1_deck_name=p1_deck_name,
                p2_deck_name=p2_deck_name,
                max_steps=render_max_steps,
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
            "timeouts": timeouts,
            "stall_timeouts": stall_timeouts,
            "win_rate": wins / max(1, episodes),
            "win_rate_decided": wins / max(1, episodes - timeouts),
            "episode_log": episode_log,
        },
        "render": {
            "frames_dir": str(render_dir) if render_dir is not None else None,
            "gif": str(gif_path) if gif_path is not None else None,
            "outcome": render_outcome,
            "max_steps": render_max_steps if render_gif else None,
        },
    }
    summary_path = eval_dir / "latest_eval.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation summary -> {summary_path}")
    if gif_path is not None:
        print(f"Saved optimal-policy GIF -> {gif_path}")

    # Update persistent win-rate history and chart for this training run.
    # Store at the p1 run-level dir (.../p3_xxx/p1/) so every episode
    # checkpoint in the same run contributes to a single growing curve.
    history_dir = p1_bundle.checkpoint_dir.parent  # .../p3_xxx/p1/
    history_path = history_dir / "eval_history.json"
    chart_path = history_dir / "winrate_chart.png"
    history = _append_to_history(summary, history_path)
    chart_matchup = p1_bundle.p1_hero + " vs " + p1_bundle.p2_hero
    chart_ok = _update_winrate_chart(history, chart_path, matchup=chart_matchup)
    if chart_ok:
        print(f"Updated win-rate chart   -> {chart_path}  ({len(history)} checkpoint(s))")
    else:
        print(f"Updated eval history     -> {history_path}  ({len(history)} checkpoint(s))")

    summary["history"] = {
        "history_path": str(history_path),
        "chart_path": str(chart_path) if chart_ok else None,
        "checkpoints_in_history": len(history),
    }
    # Re-write summary with history paths included.
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

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
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of worker threads used to evaluate episodes in parallel.",
    )
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--watch", action="store_true",
                        help="Keep watching results/ and re-evaluate when a new checkpoint appears.")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--no-render-gif", action="store_true")
    parser.add_argument(
        "--render-max-steps",
        type=int,
        default=20,
        help="Max steps for the single post-eval render replay episode.",
    )
    parser.add_argument("--gif-fps", type=float, default=3.0)
    parser.add_argument(
        "--stall-no-damage-turns",
        type=int,
        default=6,
        help="Early-end eval episode after this many consecutive no-damage turns when low-hand stall pattern is also present.",
    )
    parser.add_argument(
        "--stall-low-hand-turns",
        type=int,
        default=3,
        help="Require both players to have this many consecutive low-hand main phases for one stall condition.",
    )
    parser.add_argument(
        "--stall-max-single-low-hand-turns",
        type=int,
        default=5,
        help="Alternative stall condition: either player reaches this many consecutive low-hand main phases.",
    )
    parser.add_argument(
        "--stall-min-attack-hand",
        type=int,
        default=2,
        help="Hand size threshold considered too low to mount attacks in main phase.",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose Talishar environment debug logs.")
    parser.add_argument("--talishar-url", default=os.environ.get("TALISHAR_URL", "http://localhost:8080/game"))
    parser.add_argument("--talishar-fe-url", default=os.environ.get("TALISHAR_FE_URL", "http://localhost:5173"))
    parser.add_argument("--assets-path", default=os.environ.get("TALISHAR_ASSETS_PATH", ""))
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        raise SystemExit(f"results directory not found: {results_dir}")
    if not args.assets_path:
        raise SystemExit("TALISHAR_ASSETS_PATH or --assets-path is required")

    print("=" * 72)
    print("  Phase 3 Eval Dashboard — starting up")
    print("=" * 72)
    print(f"  Watching      : {results_dir}")
    print(f"  Talishar URL  : {args.talishar_url}")
    print(f"  Assets path   : {args.assets_path}")
    print(
        f"  Episodes      : {args.episodes}  |  workers: {max(1, args.parallel_workers)}"
        f"  |  max steps: {args.max_steps}"
    )
    print(f"  Watch mode    : {'yes' if args.watch else 'no'}  "
          f"(poll every {args.poll_seconds}s)")
    print(f"  Render GIF    : {'yes' if not args.no_render_gif else 'no'}")
    if not args.no_render_gif:
        print(f"  Render steps  : {args.render_max_steps}")
    print(
        "  Stall guard   : "
        f"{args.stall_no_damage_turns} no-dmg turns, "
        f"< {args.stall_min_attack_hand} cards for "
        f"{args.stall_low_hand_turns}/{args.stall_max_single_low_hand_turns} turns"
    )
    print("=" * 72, flush=True)

    last_seen: Optional[Path] = None
    poll_count = 0
    while True:
        if args.checkpoint_dir:
            p1_bundle = _load_checkpoint(Path(args.checkpoint_dir).expanduser().resolve(), "p1")
        else:
            p1_bundle = _latest_checkpoint(results_dir, "p1")

        if p1_bundle is None:
            print(f"  [watch] No checkpoints found under {results_dir}  "
                  f"(poll #{poll_count + 1})", flush=True)
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
                render_max_steps=args.render_max_steps,
                gif_fps=args.gif_fps,
                seed=args.seed,
                stall_no_damage_turns=args.stall_no_damage_turns,
                stall_low_hand_turns=args.stall_low_hand_turns,
                stall_max_single_low_hand_turns=args.stall_max_single_low_hand_turns,
                stall_min_attack_hand=args.stall_min_attack_hand,
                parallel_workers=args.parallel_workers,
                verbose=args.verbose,
            )
            last_seen = p1_bundle.checkpoint_dir
        else:
            print(f"  [watch] checkpoint already evaluated "
                  f"({p1_bundle.checkpoint_dir.name})  "
                  f"— waiting {args.poll_seconds:.0f}s for new checkpoint "
                  f"(poll #{poll_count + 1})", flush=True)

        poll_count += 1
        if not args.watch:
            break
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()