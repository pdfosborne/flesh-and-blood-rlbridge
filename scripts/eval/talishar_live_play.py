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
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

REPO_ROOT = _bootstrap.configure_paths()
RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()
if str(RL_SRC) not in sys.path:
    sys.path.insert(0, str(RL_SRC))

from flesh_and_blood_rlbridge import TalisharEngineEnvironment  # noqa: E402
from flesh_and_blood_rlbridge.live_action_advisor import LiveActionCoach  # noqa: E402
from flesh_and_blood_rlbridge.live_play_cancel import LivePlayCancelled  # noqa: E402
from flesh_and_blood_rlbridge.talishar_engine_environment import (  # noqa: E402
    parse_acting_player_id,
)
from eval_phase3_checkpoint import (  # noqa: E402
    CheckpointBundle,
    _deck_cards,
    _equipment_header,
    _latest_checkpoint,
    _load_agent,
    _load_checkpoint,
    _paired_checkpoint,
    _resolve_p2_preset_deck_name,
    deck_labels_from_checkpoints,
    is_sideboard_compare_dir,
)
from play_outcome_stats import (  # noqa: E402
    absolute_p1_p2_deck_from_env,
    absolute_p1_p2_deck_from_obs,
    absolute_p1_p2_hp_from_env,
    absolute_p1_p2_hp_from_obs,
    classify_p1_episode_outcome,
)
from train_play import _ensure_playwright  # noqa: E402
from train_pipeline_common import _write_deck_file  # noqa: E402

CoachHintsCallback = Callable[[list[dict[str, Any]]], None]
StatusCallback = Callable[[bool], None]
FrontendUrlCallback = Callable[[str], None]
CoachReadyCallback = Callable[[bool], None]


@dataclass
class LivePlayContext:
    """Resolved checkpoint, decks, and agents for a live Talishar session."""

    p1_agent: Any
    p2_agent: Any
    p1_deck_name: str
    p2_deck_name: str
    game_format: str
    opponent_label: str
    cleanup_files: list[Path]
    p1_bundle: Optional[CheckpointBundle] = None
    p2_bundle: Optional[CheckpointBundle] = None
    human_vs_agent: bool = False
    human_deck: str = "opponent"  # "trained" | "opponent"
    human_player_id: int = 1
    agent_player_id: int = 2
    trained_deck_label: str = ""
    opponent_deck_label: str = ""
    trained_agent: Any = None
    opponent_policy: str = "agent"  # "agent" | "logic"
    cpp_engine_deck1: str = ""
    cpp_engine_deck2: str = ""


def deck_labels_from_bundle(
    bundle: CheckpointBundle,
    p2_bundle: Optional[CheckpointBundle] = None,
) -> tuple[str, str]:
    """Return display labels for the trained and opponent decks."""
    return deck_labels_from_checkpoints(bundle, p2_bundle)


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

    trained_label, opponent_label_name = deck_labels_from_bundle(p1_bundle, p2_bundle)
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


def _gui_human_deck_to_trained_opponent(human_deck: str) -> str:
    """Map GUI seat choice to configure_human_vs_agent vocabulary."""
    if human_deck == "player":
        return "trained"
    if human_deck == "opponent":
        return "opponent"
    if human_deck in ("trained", "opponent"):
        return human_deck
    raise ValueError("human_deck must be one of: player, opponent, trained")


def _is_logic_policy(agent: Any) -> bool:
    from train_play import LOGIC_POLICY  # noqa: PLC0415

    return agent is LOGIC_POLICY or str(agent) == LOGIC_POLICY


def _normalize_opponent_policy(opponent_policy: str) -> str:
    token = str(opponent_policy or "agent").strip().lower()
    if token not in {"agent", "logic"}:
        raise ValueError("opponent_policy must be 'agent' or 'logic'")
    return token


def _opponent_policy_label(opponent_policy: str) -> str:
    return "logic policy" if opponent_policy == "logic" else "unified agent"


def _resolve_opponent_agent(trained_agent: Any, opponent_policy: str) -> Any:
    if _normalize_opponent_policy(opponent_policy) == "logic":
        from train_play import LOGIC_POLICY  # noqa: PLC0415

        return LOGIC_POLICY
    return trained_agent


def unified_agent_cache_format(game_format: str) -> str:
    """Map deck/Talishar format labels to unified agent cache directory names."""
    from fab_tui.config import normalize_pipeline_format  # noqa: PLC0415

    return normalize_pipeline_format(game_format)


def unified_agent_weights_path(cache_dir: Path, game_format: str) -> Path:
    from flesh_and_blood_rlbridge.player_observation import (  # noqa: PLC0415
        PLAYER_OBS_SCHEMA_VERSION,
    )

    cache_format = unified_agent_cache_format(game_format)
    store_root = Path(cache_dir) / cache_format
    return store_root / f"unified_agent_v{PLAYER_OBS_SCHEMA_VERSION}.json"


def prepare_unified_live_play_context(
    *,
    player_deck: dict[str, int],
    opponent_asset_stem: str,
    player_equipment_header: str,
    game_format: str,
    assets_path: str,
    cache_dir: Path,
    base_url: str,
    fe_url: str,
    human_deck: str = "opponent",
    player_deck_label: str = "Your deck",
    opponent_deck_label: str = "Opponent deck",
    opponent_policy: str = "agent",
) -> LivePlayContext:
    """Build a live-play context from GUI decks and the unified agent cache."""
    from agent_cache import clone_agent_weights  # noqa: PLC0415
    from eval_sideboard_compare import _load_unified_agent  # noqa: PLC0415
    from rl_agents.ppo import PPOAgent  # noqa: PLC0415

    weights_path = unified_agent_weights_path(cache_dir, game_format)
    cache_format = unified_agent_cache_format(game_format)
    if not weights_path.is_file():
        extra = f" (cache key {cache_format!r})" if cache_format != game_format else ""
        raise FileNotFoundError(
            f"Unified agent not found for format {game_format!r}{extra}: {weights_path}"
        )

    p1_deck_name = f"live_p1_{uuid.uuid4().hex[:8]}"
    p1_deck_file = _write_deck_file(
        {str(k): int(v) for k, v in player_deck.items() if int(v) > 0},
        player_equipment_header,
        p1_deck_name,
        assets_path,
    )
    p2_deck_name = str(opponent_asset_stem).strip()
    if not p2_deck_name:
        raise ValueError("Opponent Talishar asset name is missing")

    from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: PLC0415
        resolve_talishar_deck_stem,
    )

    cpp_engine_deck1 = resolve_talishar_deck_stem(
        assets_path,
        player_equipment_header or p2_deck_name,
    )
    cpp_engine_deck2 = resolve_talishar_deck_stem(assets_path, p2_deck_name)

    probe_env = TalisharEngineEnvironment(
        base_url=base_url,
        frontend_url=fe_url,
        game_format=game_format,
        local_deck_name=p1_deck_name,
        opponent_deck_name=p2_deck_name,
        max_turns=60,
        self_play=True,
        render_mode=None,
        use_cpp_engine=False,
        enable_combat_tracker=False,
    )
    try:
        unified_agent = _load_unified_agent(cache_dir, cache_format, probe_env=probe_env)
    finally:
        probe_env.close()

    play_agent = PPOAgent()
    clone_agent_weights(unified_agent, play_agent)
    opponent_policy = _normalize_opponent_policy(opponent_policy)
    policy_label = _opponent_policy_label(opponent_policy)

    base_ctx = LivePlayContext(
        p1_bundle=None,
        p2_bundle=None,
        p1_agent=play_agent,
        p2_agent=play_agent,
        p1_deck_name=p1_deck_name,
        p2_deck_name=p2_deck_name,
        game_format=game_format,
        opponent_label=(
            f"Watch agent — {player_deck_label} vs {opponent_deck_label}"
            if human_deck == "watch"
            else f"{policy_label.title()} vs {opponent_deck_label}"
        ),
        cleanup_files=[p1_deck_file],
        trained_deck_label=player_deck_label,
        opponent_deck_label=opponent_deck_label,
        trained_agent=play_agent,
        opponent_policy=opponent_policy,
        cpp_engine_deck1=cpp_engine_deck1,
        cpp_engine_deck2=cpp_engine_deck2,
    )
    if human_deck == "watch":
        return base_ctx
    return configure_human_vs_agent(
        base_ctx,
        human_deck=_gui_human_deck_to_trained_opponent(human_deck),
        opponent_policy=opponent_policy,
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
    opponent_policy: str = "agent",
) -> LivePlayContext:
    """Assign player seats/agents from the human's chosen deck side."""
    if human_deck not in ("trained", "opponent"):
        raise ValueError("human_deck must be 'trained' or 'opponent'")

    opponent_policy = _normalize_opponent_policy(
        opponent_policy or ctx.opponent_policy or "agent"
    )
    trained_agent = ctx.trained_agent or ctx.p1_agent or ctx.p2_agent
    opponent_agent = _resolve_opponent_agent(trained_agent, opponent_policy)
    trained_label = ctx.trained_deck_label or "trained deck"
    opponent_label = ctx.opponent_deck_label or "opponent deck"
    policy_label = _opponent_policy_label(opponent_policy)

    if human_deck == "trained":
        human_player_id = 1
        agent_player_id = 2
        p1_agent = None
        p2_agent = opponent_agent
        human_deck_label = trained_label
        agent_deck_label = opponent_label
    else:
        human_player_id = 2
        agent_player_id = 1
        p1_agent = opponent_agent
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
            f"You ({human_deck_label}) vs {policy_label} ({agent_deck_label})"
        ),
        cleanup_files=ctx.cleanup_files,
        human_vs_agent=True,
        human_deck=human_deck,
        human_player_id=human_player_id,
        agent_player_id=agent_player_id,
        trained_deck_label=ctx.trained_deck_label,
        opponent_deck_label=ctx.opponent_deck_label,
        trained_agent=trained_agent,
        opponent_policy=opponent_policy,
        cpp_engine_deck1=ctx.cpp_engine_deck1,
        cpp_engine_deck2=ctx.cpp_engine_deck2,
    )


def _verify_talishar_reachable(base_url: str) -> None:
    import requests

    try:
        requests.get(base_url.rstrip("/") + "/", timeout=5.0)
    except Exception as exc:
        raise RuntimeError(
            f"\n  Cannot reach Talishar at {base_url}\n"
            f"  Error: {exc}\n"
            "  Start the server first:  python start_talishar.py\n"
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
            "  Start the FE dev server:  python start_talishar.py --fe-only\n"
            "  (or run `npm run dev` in Talishar-FE)"
        ) from exc


_CHROMIUM_GDPR_INIT_SCRIPT = (
    "localStorage.setItem('gdpr-analytics-enabled','true');"
    "localStorage.setItem('gdpr-consent-accepted','true');"
)


def open_frontend_in_chromium(url: str) -> dict[str, Any]:
    """Open a Talishar-FE game board in a visible Chromium window (Playwright)."""
    board_url = str(url or "").strip()
    if not board_url:
        raise ValueError("No Talishar game URL to open")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required to open Chromium. "
            "Run: pip install playwright && playwright install chromium"
        ) from exc

    def _worker() -> None:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()
            page.add_init_script(_CHROMIUM_GDPR_INIT_SCRIPT)
            page.goto(board_url, wait_until="domcontentloaded", timeout=30000)
            try:
                agree = page.locator("button", has_text="Agree").first
                if agree.is_visible(timeout=800):
                    agree.click()
            except Exception:
                pass
            try:
                page.wait_for_event("close", timeout=0)
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass

    thread = threading.Thread(
        target=_worker,
        name="talishar-chromium",
        daemon=True,
    )
    thread.start()
    return {"opened": True, "url": board_url}


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
    if ctx.trained_agent is not None and not _is_logic_policy(ctx.trained_agent):
        return ctx.trained_agent
    if ctx.agent_player_id == 2:
        agent = ctx.p2_agent
    else:
        agent = ctx.p1_agent
    if agent is not None and not _is_logic_policy(agent):
        return agent
    return ctx.trained_agent


def _refresh_human_action_coach(
    env: TalisharEngineEnvironment,
    ctx: LivePlayContext,
    action_coach: Optional[LiveActionCoach],
    state: dict[str, Any],
    *,
    on_hints: Optional[CoachHintsCallback] = None,
    last_hint_key: Optional[list[str]] = None,
) -> None:
    if action_coach is None:
        return
    state_key = _overlay_state_key(state)
    if on_hints is None and state_key == env._last_action_overlay_key:
        return
    if on_hints is not None and last_hint_key is not None and state_key == last_hint_key[0]:
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
    if on_hints is not None:
        on_hints([h.to_dict() for h in hints])
        if last_hint_key is not None:
            last_hint_key[0] = state_key
        return
    env.update_frontend_action_overlay(
        hints,
        state_key=state_key,
    )


def _refresh_human_action_overlay(
    env: TalisharEngineEnvironment,
    ctx: LivePlayContext,
    action_coach: Optional[LiveActionCoach],
    state: dict[str, Any],
) -> None:
    _refresh_human_action_coach(env, ctx, action_coach, state)


def _pick_action(
    env: TalisharEngineEnvironment,
    obs: Any,
    *,
    acting: int,
    p1_agent: Any,
    p2_agent: Any,
) -> Any:
    agent = p1_agent if acting == 1 else p2_agent
    if agent is not None and _is_logic_policy(agent):
        return env.sample_action()
    if agent is not None and hasattr(agent, "act_greedy"):
        return agent.act_greedy(obs)
    if agent is not None and hasattr(agent, "act"):
        return agent.act(obs)
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
    embedded: bool = False,
    on_coach_hints: Optional[CoachHintsCallback] = None,
    on_your_turn: Optional[StatusCallback] = None,
    on_frontend_url: Optional[FrontendUrlCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict[str, Any]:
    use_overlay = not embedded and getattr(env, "_render_mode", None) == "rgb_array"
    coach_active = action_coach is not None and (use_overlay or on_coach_hints is not None)
    last_hint_key: list[str] = [""]

    try:
        result = env.reset(seed=seed)
    except Exception:
        raise
    obs = result.observation
    fe_url = env._frontend_game_url() or ""
    if on_frontend_url is not None and fe_url:
        on_frontend_url(fe_url)

    if ctx.human_vs_agent:
        human_deck_label = (
            ctx.trained_deck_label
            if ctx.human_deck == "trained"
            else ctx.opponent_deck_label
        )
        if not embedded:
            print(
                f"\n  Game {episode_no}: you are playing {human_deck_label} "
                f"(P{ctx.human_player_id}) — click actions in the Talishar window",
                flush=True,
            )
    elif not embedded:
        if getattr(env, "_render_mode", None) == "human":
            render_result = env.render()
            if render_result.mode == "human" and render_result.text:
                fe_url = render_result.text
            print(f"\n  Game {episode_no}: browser opened — watch Talishar FE", flush=True)
        else:
            print(
                f"\n  Game {episode_no}: Chromium window opened — watch Talishar FE",
                flush=True,
            )

    if fe_url and not embedded:
        print(f"  Frontend URL : {fe_url}", flush=True)

    steps = 0
    terminated = False
    truncated = False
    human_prompted = False
    outcome = "draw"

    try:
        while not (terminated or truncated) and steps < max_steps:
            if cancel_event is not None and cancel_event.is_set():
                return {
                    "episode": episode_no,
                    "outcome": "cancelled",
                    "steps": steps,
                    "p1_hp": None,
                    "p2_hp": None,
                    "frontend_url": fe_url,
                    "terminated": False,
                    "truncated": False,
                }

            acting = parse_acting_player_id(env, obs)

            if ctx.human_vs_agent and acting == ctx.human_player_id:
                if on_your_turn is not None:
                    on_your_turn(True)
                if not human_prompted and not embedded:
                    print(
                        f"  Your turn (P{ctx.human_player_id}) — "
                        "click actions in the Talishar window",
                        flush=True,
                    )
                    if coach_active:
                        print(
                            "  Agent coach enabled (policy"
                            + (" + C++ win estimates)." if action_coach and action_coach.cpp_env else ")."),
                            flush=True,
                        )
                    human_prompted = True
                if coach_active and isinstance(env._last_state, dict) and env._last_state:
                    _refresh_human_action_coach(
                        env,
                        ctx,
                        action_coach,
                        env._last_state,
                        on_hints=on_coach_hints if embedded else None,
                        last_hint_key=last_hint_key if embedded else None,
                    )

                def _on_waiting(state: dict[str, Any]) -> None:
                    if coach_active:
                        _refresh_human_action_coach(
                            env,
                            ctx,
                            action_coach,
                            state,
                            on_hints=on_coach_hints if embedded else None,
                            last_hint_key=last_hint_key if embedded else None,
                        )

                obs = env.wait_for_human_player(
                    ctx.human_player_id,
                    on_waiting=_on_waiting if coach_active else None,
                    cancel_event=cancel_event,
                )
                if use_overlay:
                    env.clear_frontend_action_overlay()
                if on_your_turn is not None:
                    on_your_turn(False)
                if on_coach_hints is not None:
                    on_coach_hints([])
                human_prompted = False
                acting = parse_acting_player_id(env, obs)
                terminated = env._is_game_over(env._last_state)
                steps += 1
                if terminated:
                    break
                if acting == ctx.human_player_id:
                    continue
            elif on_your_turn is not None:
                on_your_turn(False)

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
        if ctx.human_vs_agent:
            outcome = _outcome_from_human_perspective(outcome, ctx.human_player_id)
            result_label = outcome.upper()
            if outcome == "win":
                result_label = "YOU WIN"
            elif outcome == "loss":
                result_label = "YOU LOSE"
        else:
            result_label = outcome.upper()
        if not embedded:
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
    except LivePlayCancelled:
        return {
            "episode": episode_no,
            "outcome": "cancelled",
            "steps": steps,
            "p1_hp": None,
            "p2_hp": None,
            "frontend_url": fe_url,
            "terminated": False,
            "truncated": False,
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
    embedded: bool = False,
    ctx: Optional[LivePlayContext] = None,
    on_coach_hints: Optional[CoachHintsCallback] = None,
    on_your_turn: Optional[StatusCallback] = None,
    on_frontend_url: Optional[FrontendUrlCallback] = None,
    on_coach_ready: Optional[CoachReadyCallback] = None,
    cancel_event: Optional[threading.Event] = None,
    use_chromium_browser: bool = True,
) -> dict[str, Any]:
    if games < 1:
        raise ValueError("games must be >= 1")
    if not assets_path:
        raise ValueError("TALISHAR_ASSETS_PATH or --assets-path is required")

    if not embedded:
        _ensure_playwright()
    elif use_chromium_browser:
        _ensure_playwright()
    _verify_talishar_reachable(base_url)
    _verify_frontend_reachable(fe_url)

    if ctx is None:
        ctx = prepare_live_play_context(
            results_dir,
            candidate_id=candidate_id,
            checkpoint_dir=checkpoint_dir,
            assets_path=assets_path,
        )
        if human_vs_agent:
            ctx = configure_human_vs_agent(ctx, human_deck=human_deck)

    mode_label = "Human vs trained agent" if ctx.human_vs_agent else "Real-time Talishar play"
    if not embedded:
        print(f"\n{'=' * 72}")
        print(f"  {mode_label}")
        print(f"{'=' * 72}")
        if ctx.p1_bundle is not None:
            print(f"  Checkpoint   : {ctx.p1_bundle.checkpoint_dir.name}")
            print(f"  Matchup      : {ctx.p1_bundle.matchup}")
            print(f"  Train eps    : {ctx.p1_bundle.episodes_completed}")
        else:
            print("  Agent        : unified policy cache")
        print(f"  Setup        : {ctx.opponent_label}")
        print(f"  Games        : {games}  |  max steps: {max_steps}")
        print(f"  Talishar URL : {base_url}")
        print(f"  Frontend URL : {fe_url}")
        print(f"{'=' * 72}\n", flush=True)

    use_playwright_window = (not embedded) or use_chromium_browser
    render_mode = "rgb_array" if use_playwright_window else None
    episode_logs: list[dict[str, Any]] = []
    action_coach: Optional[LiveActionCoach] = None
    rollouts = coach_rollouts_per_action if coach_rollouts_per_action > 0 else 0
    cpp_lookup1 = ctx.cpp_engine_deck1 or ctx.p1_deck_name
    cpp_lookup2 = ctx.cpp_engine_deck2 or ctx.p2_deck_name
    resolved_cpp_dir = cpp_engine_dir
    if rollouts > 0 and resolved_cpp_dir is None and assets_path:
        from cpp_engine_matchup import discover_cpp_engine_dir  # noqa: PLC0415

        discovered = discover_cpp_engine_dir(
            cpp_lookup1,
            cpp_lookup2,
            assets_path=assets_path,
        )
        if discovered is not None:
            resolved_cpp_dir = str(discovered)
    if ctx.human_vs_agent and enable_action_coach:
        trained_agent = _trained_agent_from_context(ctx)
        if trained_agent is not None:
            action_coach = LiveActionCoach.try_create(
                coach_agent=trained_agent,
                opponent_agent=trained_agent,
                base_url=base_url,
                game_format=ctx.game_format,
                deck1=cpp_lookup1,
                deck2=cpp_lookup2,
                rollouts_per_action=rollouts,
                max_rollout_steps=coach_max_rollout_steps,
                cpp_engine_dir=resolved_cpp_dir if rollouts > 0 else None,
            )
            coach_cpp_ready = action_coach.cpp_env is not None and rollouts > 0
            if on_coach_ready is not None:
                on_coach_ready(coach_cpp_ready)
            if not embedded:
                if not coach_cpp_ready:
                    print(
                        "  Agent coach: policy overlay only "
                        f"(no C++ engine for {cpp_lookup1} vs {cpp_lookup2}).",
                        flush=True,
                    )
                else:
                    print(
                        f"  Agent coach: policy + C++ rollouts "
                        f"({rollouts} per action).",
                        flush=True,
                    )
    elif on_coach_ready is not None:
        on_coach_ready(False)

    env = TalisharEngineEnvironment(
        base_url=base_url,
        frontend_url=fe_url,
        game_format=ctx.game_format,
        local_deck_name=ctx.p1_deck_name,
        opponent_deck_name=ctx.p2_deck_name,
        max_turns=max_steps,
        self_play=True,
        render_mode=render_mode,
        playwright_headless=False,
        frontend_player_id=ctx.human_player_id if ctx.human_vs_agent else None,
        enable_frontend_card_hover=ctx.human_vs_agent and (not embedded or use_chromium_browser),
        use_cpp_engine=False,
        enable_combat_tracker=False,
    )
    try:
        for ep in range(1, games + 1):
            if cancel_event is not None and cancel_event.is_set():
                break
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
                    embedded=embedded,
                    on_coach_hints=on_coach_hints,
                    on_your_turn=on_your_turn,
                    on_frontend_url=on_frontend_url if ep == 1 else None,
                    cancel_event=cancel_event,
                )
            )
            if episode_logs[-1].get("outcome") == "cancelled":
                break
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
    cancelled = any(row.get("outcome") == "cancelled" for row in episode_logs)

    summary: dict[str, Any] = {
        "games": games,
        "human_vs_agent": ctx.human_vs_agent,
        "record": {"wins": wins, "losses": losses, "draws": draws, "timeouts": timeouts},
        "episodes": episode_logs,
        "cancelled": cancelled,
    }
    if ctx.p1_bundle is not None:
        summary["checkpoint_dir"] = str(ctx.p1_bundle.checkpoint_dir)
        summary["matchup"] = ctx.p1_bundle.matchup
    if ctx.human_vs_agent and not embedded:
        print(
            f"\n  Your record: {wins}W  {losses}L  {draws}D  {timeouts}T",
            flush=True,
        )
    elif not embedded:
        print(
            f"\n  Session record: {wins}W  {losses}L  {draws}D  {timeouts}T",
            flush=True,
        )
    return summary


def run_embedded_unified_live_play_session(
    *,
    player_deck: dict[str, int],
    opponent_asset_stem: str,
    player_equipment_header: str,
    game_format: str,
    assets_path: str,
    cache_dir: Path,
    base_url: str,
    fe_url: str,
    human_deck: str = "opponent",
    player_deck_label: str = "Your deck",
    opponent_deck_label: str = "Opponent deck",
    opponent_policy: str = "agent",
    max_steps: int = 60,
    enable_action_coach: bool = True,
    coach_rollouts_per_action: int = 0,
    coach_max_rollout_steps: int = 60,
    cpp_engine_dir: Optional[str] = None,
    on_coach_hints: Optional[CoachHintsCallback] = None,
    on_your_turn: Optional[StatusCallback] = None,
    on_frontend_url: Optional[FrontendUrlCallback] = None,
    on_coach_ready: Optional[CoachReadyCallback] = None,
    cancel_event: Optional[threading.Event] = None,
    use_chromium_browser: bool = True,
) -> dict[str, Any]:
    """GUI live play: coach in the web UI, board in Playwright Chromium by default."""
    ctx = prepare_unified_live_play_context(
        player_deck=player_deck,
        opponent_asset_stem=opponent_asset_stem,
        player_equipment_header=player_equipment_header,
        game_format=game_format,
        assets_path=assets_path,
        cache_dir=cache_dir,
        base_url=base_url,
        fe_url=fe_url,
        human_deck=human_deck,
        player_deck_label=player_deck_label,
        opponent_deck_label=opponent_deck_label,
        opponent_policy=opponent_policy,
    )
    return run_live_play_session(
        Path("."),
        ctx=ctx,
        games=1,
        max_steps=max_steps,
        human_vs_agent=ctx.human_vs_agent,
        enable_action_coach=enable_action_coach,
        coach_rollouts_per_action=coach_rollouts_per_action,
        coach_max_rollout_steps=coach_max_rollout_steps,
        cpp_engine_dir=cpp_engine_dir,
        base_url=base_url,
        fe_url=fe_url,
        assets_path=assets_path,
        embedded=True,
        on_coach_hints=on_coach_hints,
        on_your_turn=on_your_turn,
        on_frontend_url=on_frontend_url,
        on_coach_ready=on_coach_ready,
        cancel_event=cancel_event,
        use_chromium_browser=use_chromium_browser,
    )


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
