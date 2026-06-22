#!/usr/bin/env python3
"""Play Talishar games with a trained agent and the live Talishar frontend.

Loads the latest (or specified) phase-3 checkpoint, opens the Talishar FE in
your browser on each game, and steps the agent greedily while you watch.
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

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

REPO_ROOT = _bootstrap.configure_paths()
RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()
if str(RL_SRC) not in sys.path:
    sys.path.insert(0, str(RL_SRC))

from flesh_and_blood_rlbridge import TalisharEngineEnvironment  # noqa: E402
from flesh_and_blood_rlbridge.live_action_advisor import LiveActionCoach  # noqa: E402
from flesh_and_blood_rlbridge.talishar_engine_environment import (  # noqa: E402
    parse_acting_player_id,
)
from scripts.eval.eval_phase3_checkpoint import (  # noqa: E402
    CheckpointBundle,
    _deck_cards,
    _equipment_header,
    _latest_checkpoint,
    _load_agent,
    _load_checkpoint,
    _paired_checkpoint,
    _resolve_p2_preset_deck_name,
    is_sideboard_compare_dir,
)
from scripts.training.play_outcome_stats import (  # noqa: E402
    absolute_p1_p2_hp_from_env,
    absolute_p1_p2_hp_from_obs,
    classify_p1_episode_outcome,
)
from scripts.training.train_play import _ensure_playwright  # noqa: E402
from scripts.training.train_pipeline_common import _write_deck_file  # noqa: E402


@dataclass
class LivePlayContext:
    """Resolved checkpoint, decks, and agents for a live Talishar session."""

    p1_bundle: CheckpointBundle
    p2_bundle: Optional[CheckpointBundle]
    p1_agent: Any
    p2_agent: Any
    p1_deck_name: str
    p2_deck_name: str
    game_format: str
    opponent_label: str
    cleanup_files: list[Path]
    human_vs_agent: bool = False
    human_deck: str = "opponent"  # "trained" | "opponent"
    human_player_id: int = 1
    agent_player_id: int = 2
    trained_deck_label: str = ""
    opponent_deck_label: str = ""


def deck_labels_from_bundle(bundle: CheckpointBundle) -> tuple[str, str]:
    """Return display labels for the trained and opponent decks."""
    trained = str(
        bundle.metadata.get("p1_hero")
        or bundle.metadata.get("hero_id")
        or ""
    ).replace("_", " ").strip()
    if not trained and "-vs-" in bundle.matchup:
        trained = bundle.matchup.split("-vs-", 1)[0].replace("_", " ").strip()
    if not trained:
        trained = "trained deck"

    opponent = str(bundle.metadata.get("opponent_deck_name") or "").strip()
    if not opponent and "-vs-" in bundle.matchup:
        opponent = bundle.matchup.split("-vs-", 1)[1].replace("_", " ").strip()
    if not opponent:
        opponent = "opponent deck"
    return trained, opponent


def resolve_checkpoint_bundles(
    results_dir: Path,
    *,
    candidate_id: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
) -> tuple[CheckpointBundle, Optional[CheckpointBundle]]:
    if checkpoint_dir is not None:
        p1_bundle = _load_checkpoint(checkpoint_dir.expanduser().resolve(), "p1")
        if p1_bundle is None:
            raise FileNotFoundError(f"P1 checkpoint not found: {checkpoint_dir}")
        p2_bundle = _paired_checkpoint(p1_bundle, "p2")
        return p1_bundle, p2_bundle

    p1_bundle = _latest_checkpoint(results_dir, "p1", candidate_id=candidate_id)
    if p1_bundle is None:
        scope = results_dir
        if candidate_id and is_sideboard_compare_dir(results_dir):
            scope = results_dir / "candidates" / candidate_id
        raise FileNotFoundError(f"No P1 checkpoints found under {scope}")

    return p1_bundle, _paired_checkpoint(p1_bundle, "p2")


def prepare_live_play_context(
    results_dir: Path,
    *,
    candidate_id: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    assets_path: str,
) -> LivePlayContext:
    p1_bundle, p2_bundle = resolve_checkpoint_bundles(
        results_dir,
        candidate_id=candidate_id,
        checkpoint_dir=checkpoint_dir,
    )

    p1_cards = _deck_cards(p1_bundle)
    if not p1_cards:
        raise RuntimeError(f"Checkpoint missing P1 deck spec: {p1_bundle.checkpoint_dir}")

    p1_deck_name = f"live_p1_{uuid.uuid4().hex[:8]}"
    p1_deck_file = _write_deck_file(
        p1_cards,
        _equipment_header(p1_bundle),
        p1_deck_name,
        assets_path,
    )
    cleanup_files = [p1_deck_file]

    opponent_mode = str(p1_bundle.metadata.get("opponent_mode", "preset") or "preset")
    p2_agent: Any = None
    if p2_bundle is not None and _deck_cards(p2_bundle):
        p2_deck_name = f"live_p2_{uuid.uuid4().hex[:8]}"
        p2_deck_file = _write_deck_file(
            _deck_cards(p2_bundle),
            _equipment_header(p2_bundle),
            p2_deck_name,
            assets_path,
        )
        cleanup_files.append(p2_deck_file)
        opponent_label = (
            f"trained P2 agent — {p2_bundle.role} checkpoint "
            f"({p2_bundle.episodes_completed} eps)"
        )
    elif opponent_mode == "mirror":
        p2_deck_name = p1_deck_name
        opponent_label = "mirror (same checkpoint)"
    else:
        p2_deck_name = _resolve_p2_preset_deck_name(p1_bundle, p2_bundle)
        if p2_bundle is not None:
            opponent_label = (
                f"preset deck {p2_deck_name} "
                f"+ trained P2 agent ({p2_bundle.episodes_completed} eps)"
            )
        else:
            opponent_label = f"preset deck {p2_deck_name} (P2 default heuristic)"

    p1_agent = _load_agent(p1_bundle.weights_path)
    if opponent_mode == "mirror":
        p2_agent = p1_agent
    elif p2_bundle is not None:
        p2_agent = _load_agent(p2_bundle.weights_path)
    else:
        p2_agent = None

    trained_label, opponent_label_name = deck_labels_from_bundle(p1_bundle)
    if opponent_mode == "mirror":
        opponent_label_name = trained_label

    return LivePlayContext(
        p1_bundle=p1_bundle,
        p2_bundle=p2_bundle,
        p1_agent=p1_agent,
        p2_agent=p2_agent,
        p1_deck_name=p1_deck_name,
        p2_deck_name=p2_deck_name,
        game_format=p1_bundle.game_format,
        opponent_label=opponent_label,
        cleanup_files=cleanup_files,
        trained_deck_label=trained_label,
        opponent_deck_label=opponent_label_name,
    )


def _outcome_from_human_perspective(outcome: str, human_player_id: int) -> str:
    """Map a P1-relative outcome to the human player's seat."""
    if human_player_id == 2:
        if outcome == "win":
            return "loss"
        if outcome == "loss":
            return "win"
    return outcome


def configure_human_vs_agent(
    ctx: LivePlayContext,
    *,
    human_deck: str = "opponent",
) -> LivePlayContext:
    """Assign player seats/agents from the human's chosen deck side."""
    if human_deck not in ("trained", "opponent"):
        raise ValueError("human_deck must be 'trained' or 'opponent'")

    trained_agent = ctx.p1_agent
    trained_label = ctx.trained_deck_label or "trained deck"
    opponent_label = ctx.opponent_deck_label or "opponent deck"

    if human_deck == "trained":
        human_player_id = 1
        agent_player_id = 2
        p1_agent = None
        p2_agent = trained_agent
        human_deck_label = trained_label
        agent_deck_label = opponent_label
    else:
        human_player_id = 2
        agent_player_id = 1
        p1_agent = trained_agent
        p2_agent = None
        human_deck_label = opponent_label
        agent_deck_label = trained_label

    return LivePlayContext(
        p1_bundle=ctx.p1_bundle,
        p2_bundle=ctx.p2_bundle,
        p1_agent=p1_agent,
        p2_agent=p2_agent,
        p1_deck_name=ctx.p1_deck_name,
        p2_deck_name=ctx.p2_deck_name,
        game_format=ctx.game_format,
        opponent_label=(
            f"You ({human_deck_label}) vs trained agent ({agent_deck_label})"
        ),
        cleanup_files=ctx.cleanup_files,
        human_vs_agent=True,
        human_deck=human_deck,
        human_player_id=human_player_id,
        agent_player_id=agent_player_id,
        trained_deck_label=ctx.trained_deck_label,
        opponent_deck_label=ctx.opponent_deck_label,
    )


def _verify_talishar_reachable(base_url: str) -> None:
    import requests

    try:
        requests.get(base_url.rstrip("/") + "/", timeout=5.0)
    except Exception as exc:
        raise RuntimeError(
            f"\n  Cannot reach Talishar at {base_url}\n"
            f"  Error: {exc}\n"
            "  Start the server first:  ./start_talishar.ps1\n"
            "  Or set TALISHAR_URL / --talishar-url to the correct address."
        ) from exc


def _verify_frontend_reachable(fe_url: str) -> None:
    import requests

    try:
        requests.get(fe_url.rstrip("/") + "/", timeout=5.0)
    except Exception as exc:
        raise RuntimeError(
            f"\n  Cannot reach Talishar frontend at {fe_url}\n"
            f"  Error: {exc}\n"
            "  Start the FE dev server:  .\\start_talishar.ps1\n"
            "  (or run `npm run dev` in Talishar-FE)"
        ) from exc


def _overlay_state_key(state: dict[str, Any]) -> str:
    turn_phase = state.get("turnPhase", {})
    if isinstance(turn_phase, dict):
        phase = str(turn_phase.get("turnPhase", "") or "")
    else:
        phase = str(turn_phase or "")
    hand_size = len(state.get("playerHand", []) or [])
    prompt = state.get("playerPrompt", {})
    popup = state.get("playerInputPopUp", {})
    popup_active = bool(isinstance(popup, dict) and popup.get("active"))
    return "|".join(
        [
            str(state.get("lastUpdate", "")),
            str(state.get("turnNo", "")),
            phase,
            str(hand_size),
            str(popup_active),
            str(len(prompt.get("buttons", [])) if isinstance(prompt, dict) else 0),
        ]
    )


def _trained_agent_from_context(ctx: LivePlayContext) -> Any:
    if ctx.agent_player_id == 2:
        return ctx.p2_agent
    return ctx.p1_agent


def _refresh_human_action_overlay(
    env: TalisharEngineEnvironment,
    ctx: LivePlayContext,
    action_coach: Optional[LiveActionCoach],
    state: dict[str, Any],
) -> None:
    if action_coach is None:
        return
    state_key = _overlay_state_key(state)
    if state_key == env._last_action_overlay_key:
        return
    env._last_state = state
    env._acting_player_id = ctx.human_player_id
    legal = env._legal_actions(state)
    obs = env._encode_observation(state, legal)
    hints = action_coach.build_hints(
        env,
        obs,
        human_player_id=ctx.human_player_id,
        legal_actions=legal,
    )
    env.update_frontend_action_overlay(
        hints,
        state_key=state_key,
    )


def _pick_action(
    env: TalisharEngineEnvironment,
    obs: Any,
    *,
    acting: int,
    p1_agent: Any,
    p2_agent: Any,
) -> str:
    agent = p1_agent if acting == 1 else p2_agent
    if agent is not None and hasattr(agent, "act_greedy"):
        return str(agent.act_greedy(obs))
    if agent is not None and hasattr(agent, "act"):
        return str(agent.act(obs))
    return env.sample_action()


def run_live_game(
    env: TalisharEngineEnvironment,
    ctx: LivePlayContext,
    *,
    max_steps: int,
    seed: Optional[int] = None,
    step_delay_ms: int = 0,
    episode_no: int = 1,
    action_coach: Optional[LiveActionCoach] = None,
) -> dict[str, Any]:
    result = env.reset(seed=seed)
    obs = result.observation
    fe_url = env._frontend_game_url() or ""

    if ctx.human_vs_agent:
        human_deck_label = (
            ctx.trained_deck_label
            if ctx.human_deck == "trained"
            else ctx.opponent_deck_label
        )
        print(
            f"\n  Game {episode_no}: you are playing {human_deck_label} "
            f"(P{ctx.human_player_id}) — click actions in the Talishar window",
            flush=True,
        )
    elif getattr(env, "_render_mode", None) == "human":
        render_result = env.render()
        if render_result.mode == "human" and render_result.text:
            fe_url = render_result.text
        print(f"\n  Game {episode_no}: browser opened — watch Talishar FE", flush=True)
    else:
        print(
            f"\n  Game {episode_no}: Chromium window opened — watch Talishar FE",
            flush=True,
        )

    if fe_url:
        print(f"  Frontend URL : {fe_url}", flush=True)

    steps = 0
    terminated = False
    truncated = False
    human_prompted = False

    while not (terminated or truncated) and steps < max_steps:
        acting = parse_acting_player_id(env, obs)

        if ctx.human_vs_agent and acting == ctx.human_player_id:
            if not human_prompted:
                print(
                    f"  Your turn (P{ctx.human_player_id}) — "
                    "click actions in the Talishar window",
                    flush=True,
                )
                if action_coach is not None:
                    print(
                        "  Agent coach overlay enabled (policy + C++ win estimates).",
                        flush=True,
                    )
                human_prompted = True
            if isinstance(env._last_state, dict) and env._last_state:
                _refresh_human_action_overlay(env, ctx, action_coach, env._last_state)

            def _on_waiting(state: dict[str, Any]) -> None:
                _refresh_human_action_overlay(env, ctx, action_coach, state)

            obs = env.wait_for_human_player(
                ctx.human_player_id,
                on_waiting=_on_waiting if action_coach is not None else None,
            )
            env.clear_frontend_action_overlay()
            human_prompted = False
            acting = parse_acting_player_id(env, obs)
            terminated = env._is_game_over(env._last_state)
            steps += 1
            if terminated:
                break
            if acting == ctx.human_player_id:
                continue

        action = _pick_action(
            env,
            obs,
            acting=acting,
            p1_agent=ctx.p1_agent,
            p2_agent=ctx.p2_agent,
        )
        step = env.step(action)
        obs = step.observation
        terminated = bool(step.terminated)
        truncated = bool(step.truncated)
        steps += 1

        if step_delay_ms > 0:
            time.sleep(step_delay_ms / 1000.0)

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
    if p1_hp is None or p2_hp is None:
        p1_hp_f, p2_hp_f = absolute_p1_p2_hp_from_obs(obs_dict)
        p1_hp = int(p1_hp_f) if p1_hp_f is not None else None
        p2_hp = int(p2_hp_f) if p2_hp_f is not None else None

    outcome = classify_p1_episode_outcome(
        p1_hp=p1_hp,
        p2_hp=p2_hp,
        terminated=terminated,
        truncated=truncated,
    )
    if ctx.human_vs_agent:
        outcome = _outcome_from_human_perspective(outcome, ctx.human_player_id)
        result_label = outcome.upper()
        if outcome == "win":
            result_label = "YOU WIN"
        elif outcome == "loss":
            result_label = "YOU LOSE"
    else:
        result_label = outcome.upper()
    print(
        f"  Game {episode_no} finished: {result_label} "
        f"({steps} steps, P1 HP={p1_hp}, P2 HP={p2_hp})",
        flush=True,
    )
    return {
        "episode": episode_no,
        "outcome": outcome,
        "steps": steps,
        "p1_hp": p1_hp,
        "p2_hp": p2_hp,
        "frontend_url": fe_url,
        "terminated": terminated,
        "truncated": truncated,
    }


def run_live_play_session(
    results_dir: Path,
    *,
    candidate_id: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    games: int = 1,
    max_steps: int = 60,
    seed: Optional[int] = None,
    step_delay_ms: int = 0,
    human_vs_agent: bool = False,
    human_deck: str = "opponent",
    enable_action_coach: bool = True,
    coach_rollouts_per_action: int = 4,
    coach_max_rollout_steps: int = 60,
    cpp_engine_dir: Optional[str] = None,
    base_url: str,
    fe_url: str,
    assets_path: str,
) -> dict[str, Any]:
    if games < 1:
        raise ValueError("games must be >= 1")
    if not assets_path:
        raise ValueError("TALISHAR_ASSETS_PATH or --assets-path is required")

    _ensure_playwright()
    _verify_talishar_reachable(base_url)
    _verify_frontend_reachable(fe_url)
    ctx = prepare_live_play_context(
        results_dir,
        candidate_id=candidate_id,
        checkpoint_dir=checkpoint_dir,
        assets_path=assets_path,
    )
    if human_vs_agent:
        ctx = configure_human_vs_agent(ctx, human_deck=human_deck)

    mode_label = "Human vs trained agent" if ctx.human_vs_agent else "Real-time Talishar play"
    print(f"\n{'=' * 72}")
    print(f"  {mode_label}")
    print(f"{'=' * 72}")
    print(f"  Checkpoint   : {ctx.p1_bundle.checkpoint_dir.name}")
    print(f"  Matchup      : {ctx.p1_bundle.matchup}")
    print(f"  Train eps    : {ctx.p1_bundle.episodes_completed}")
    print(f"  Setup        : {ctx.opponent_label}")
    print(f"  Games        : {games}  |  max steps: {max_steps}")
    print(f"  Talishar URL : {base_url}")
    print(f"  Frontend URL : {fe_url}")
    print(f"{'=' * 72}\n", flush=True)

    episode_logs: list[dict[str, Any]] = []
    action_coach: Optional[LiveActionCoach] = None
    if ctx.human_vs_agent and enable_action_coach:
        trained_agent = _trained_agent_from_context(ctx)
        if trained_agent is not None:
            action_coach = LiveActionCoach.try_create(
                coach_agent=trained_agent,
                opponent_agent=trained_agent,
                base_url=base_url,
                game_format=ctx.game_format,
                deck1=ctx.p1_deck_name,
                deck2=ctx.p2_deck_name,
                rollouts_per_action=coach_rollouts_per_action,
                max_rollout_steps=coach_max_rollout_steps,
                cpp_engine_dir=cpp_engine_dir,
            )
            if action_coach.cpp_env is None:
                print(
                    "  Agent coach: policy overlay only "
                    "(no C++ engine found for win-rate estimates).",
                    flush=True,
                )
            else:
                print(
                    f"  Agent coach: policy + C++ rollouts "
                    f"({coach_rollouts_per_action} per action).",
                    flush=True,
                )

    env = TalisharEngineEnvironment(
        base_url=base_url,
        frontend_url=fe_url,
        game_format=ctx.game_format,
        local_deck_name=ctx.p1_deck_name,
        opponent_deck_name=ctx.p2_deck_name,
        max_turns=max_steps,
        self_play=True,
        render_mode="rgb_array",
        playwright_headless=False,
        frontend_player_id=ctx.human_player_id if ctx.human_vs_agent else None,
        enable_frontend_card_hover=ctx.human_vs_agent,
        use_cpp_engine=False,
        enable_combat_tracker=False,
    )
    try:
        for ep in range(1, games + 1):
            ep_seed = (seed + ep) if seed is not None else None
            episode_logs.append(
                run_live_game(
                    env,
                    ctx,
                    max_steps=max_steps,
                    seed=ep_seed,
                    step_delay_ms=step_delay_ms,
                    episode_no=ep,
                    action_coach=action_coach,
                )
            )
    finally:
        env.close()
        if action_coach is not None:
            action_coach.close()
        for path in ctx.cleanup_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    wins = sum(1 for row in episode_logs if row.get("outcome") == "win")
    losses = sum(1 for row in episode_logs if row.get("outcome") == "loss")
    draws = sum(1 for row in episode_logs if row.get("outcome") == "draw")
    timeouts = sum(1 for row in episode_logs if row.get("outcome") == "timeout")

    summary = {
        "checkpoint_dir": str(ctx.p1_bundle.checkpoint_dir),
        "matchup": ctx.p1_bundle.matchup,
        "games": games,
        "human_vs_agent": ctx.human_vs_agent,
        "record": {"wins": wins, "losses": losses, "draws": draws, "timeouts": timeouts},
        "episodes": episode_logs,
    }
    if ctx.human_vs_agent:
        print(
            f"\n  Your record: {wins}W  {losses}L  {draws}D  {timeouts}T",
            flush=True,
        )
    else:
        print(
            f"\n  Session record: {wins}W  {losses}L  {draws}D  {timeouts}T",
            flush=True,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Talishar with a trained agent and the live frontend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "full_pipeline"))
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Sideboard compare candidate (default: latest checkpoint across candidates).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Specific P1 checkpoint directory (overrides --results-dir discovery).",
    )
    parser.add_argument("--games", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--step-delay-ms",
        type=int,
        default=0,
        help="Optional pause between agent steps (milliseconds) for easier viewing.",
    )
    parser.add_argument(
        "--human-vs-agent",
        action="store_true",
        help=(
            "You play on the Talishar board with the training opponent deck; "
            "the trained agent plays automatically with its deck."
        ),
    )
    parser.add_argument(
        "--human-deck",
        choices=("trained", "opponent"),
        default="opponent",
        help="Which deck you play when using --human-vs-agent.",
    )
    parser.add_argument(
        "--no-action-coach",
        action="store_true",
        help="Disable the frontend agent coach overlay in human-vs-agent mode.",
    )
    parser.add_argument(
        "--coach-rollouts-per-action",
        type=int,
        default=4,
        help="C++ rollout games per legal action for win-rate estimates.",
    )
    parser.add_argument(
        "--coach-max-rollout-steps",
        type=int,
        default=60,
        help="Max steps per C++ rollout used by the action coach.",
    )
    parser.add_argument(
        "--cpp-engine-dir",
        default=None,
        help="Optional compiled C++ engine directory for coach win-rate estimates.",
    )
    parser.add_argument(
        "--talishar-url",
        default=os.environ.get("TALISHAR_URL", "http://localhost:8080/game"),
    )
    parser.add_argument(
        "--talishar-fe-url",
        default=os.environ.get("TALISHAR_FE_URL", "http://localhost:5173"),
    )
    parser.add_argument("--assets-path", default=os.environ.get("TALISHAR_ASSETS_PATH", ""))
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        raise SystemExit(f"results directory not found: {results_dir}")

    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else None
    )

    try:
        run_live_play_session(
            results_dir,
            candidate_id=args.candidate_id,
            checkpoint_dir=checkpoint_dir,
            games=args.games,
            max_steps=args.max_steps,
            seed=args.seed,
            step_delay_ms=args.step_delay_ms,
            human_vs_agent=args.human_vs_agent,
            human_deck=args.human_deck,
            enable_action_coach=not args.no_action_coach,
            coach_rollouts_per_action=args.coach_rollouts_per_action,
            coach_max_rollout_steps=args.coach_max_rollout_steps,
            cpp_engine_dir=args.cpp_engine_dir,
            base_url=args.talishar_url,
            fe_url=args.talishar_fe_url,
            assets_path=args.assets_path,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
