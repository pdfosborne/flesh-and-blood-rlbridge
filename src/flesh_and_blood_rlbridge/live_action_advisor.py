"""Action coaching for live Talishar human-vs-agent play."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from flesh_and_blood_rlbridge.frontend_action_overlay import ActionCoachHint
from flesh_and_blood_rlbridge.talishar_engine_environment import (
    parse_acting_player_id,
    try_create_cpp_eval_environment,
)


def _parse_observation(obs: Any) -> dict[str, Any]:
    if isinstance(obs, str):
        try:
            parsed = json.loads(obs)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return obs if isinstance(obs, dict) else {}


def _legal_entries_from_obs(obs: Any) -> list[dict[str, Any]]:
    parsed = _parse_observation(obs)
    legal = parsed.get("legalActions", parsed.get("legal_actions", []))
    if not isinstance(legal, list):
        return []
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(legal):
        if not isinstance(entry, dict):
            continue
        entries.append(
            {
                "index": int(entry.get("index", index) or index),
                "label": str(entry.get("label", "") or f"Action {index}"),
                "zone": str(entry.get("zone", "") or ""),
                "match_text": str(entry.get("label", "") or f"Action {index}"),
            }
        )
    return entries


def normalize_legal_entries(
    legal_actions: Optional[list[dict[str, Any]]],
    *,
    obs: Any = None,
) -> list[dict[str, Any]]:
    """Normalize Talishar env or observation legal-action dicts for coaching."""
    if not legal_actions:
        return _legal_entries_from_obs(obs) if obs is not None else []

    entries: list[dict[str, Any]] = []
    for offset, entry in enumerate(legal_actions):
        if not isinstance(entry, dict):
            continue
        index = int(entry["index"]) if "index" in entry else offset
        card_id = str(entry.get("card_id", "") or entry.get("cardNumber", "") or "")
        label = str(
            entry.get("label")
            or card_id
            or entry.get("button_input")
            or f"Action {index}"
        )
        zone = str(entry.get("zone", "") or "")
        entries.append(
            {
                "index": index,
                "label": label,
                "zone": zone,
                "match_text": card_id or label,
            }
        )
    return entries


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0:
        return np.ones_like(logits) / max(len(logits), 1)
    return exp / total


def _card_ids_from_state_hand(state: dict[str, Any]) -> list[str]:
    cards: list[str] = []
    for card in state.get("playerHand", []) or []:
        if not isinstance(card, dict):
            continue
        card_id = card.get("cardNumber") or card.get("cardID") or card.get("card_id")
        if card_id:
            cards.append(str(card_id))
    return cards


def _cpp_inner_env(env: Any) -> Any:
    inner = getattr(env, "_cpp_env", None)
    return inner if inner is not None else env


def compute_agent_policy_scores(
    agent: Any,
    obs: Any,
    *,
    legal_actions: Optional[list[dict[str, Any]]] = None,
) -> list[ActionCoachHint]:
    """Return policy probabilities for each legal action from *agent*."""
    legal = normalize_legal_entries(legal_actions, obs=obs)
    if not legal or agent is None:
        return []

    n_legal = len(legal)
    if not hasattr(agent, "_actor") or agent._actor is None:
        return [
            ActionCoachHint(
                index=int(entry["index"]),
                label=str(entry["label"]),
                zone=str(entry.get("zone", "")),
                match_text=str(entry.get("match_text", entry["label"])),
            )
            for entry in legal
        ]

    obs_vec = agent._obs_to_vec(obs)
    logits = np.asarray(agent._actor.predict(obs_vec), dtype=np.float64).reshape(-1)
    if logits.size < n_legal:
        padded = np.full(n_legal, -1e9, dtype=np.float64)
        padded[: logits.size] = logits[:n_legal]
        logits = padded
    else:
        logits = logits[:n_legal]

    if getattr(agent, "_mask_actions", False):
        masked = np.full_like(logits, -1e9)
        masked[:n_legal] = logits[:n_legal]
        logits = masked

    probs = _softmax(logits)
    best_index = int(np.argmax(probs))
    hints: list[ActionCoachHint] = []
    for offset, entry in enumerate(legal):
        hints.append(
            ActionCoachHint(
                index=int(entry["index"]),
                label=str(entry["label"]),
                zone=str(entry.get("zone", "")),
                match_text=str(entry.get("match_text", entry["label"])),
                policy_pct=float(probs[offset]),
                is_best=(offset == best_index),
            )
        )
    return hints


def _seed_cpp_from_talishar(
    cpp_inner: Any,
    talishar_env: Any,
    *,
    human_player_id: int,
) -> None:
    """Best-effort C++ state seed from the live Talishar snapshot."""
    cpp_inner.reset()
    gs = getattr(cpp_inner, "_gs", None)
    if gs is None:
        return

    fetch_state = getattr(talishar_env, "_fetch_state", None)
    if callable(fetch_state):
        for player_id in (1, 2):
            try:
                state = fetch_state(player_id=player_id, last_update=0)
            except Exception:
                continue
            hand_ids = _card_ids_from_state_hand(state)
            if hand_ids and hasattr(gs, "sync_opening_hand"):
                gs.sync_opening_hand(player_id - 1, hand_ids)

    if hasattr(gs, "set_priority"):
        gs.set_priority(human_player_id - 1)
    cpp_inner._acting_player = int(human_player_id)
    if hasattr(cpp_inner, "clear_talishar_state"):
        cpp_inner.clear_talishar_state()


def _playout_cpp_after_action(
    cpp_inner: Any,
    *,
    first_action: str,
    human_player_id: int,
    coach_agent: Any,
    opponent_agent: Any,
    max_steps: int,
) -> Optional[bool]:
    """Play out a seeded C++ position and return whether *human_player_id* won."""
    try:
        cpp_inner.step(first_action)
    except Exception:
        return None

    obs: Any = None
    terminated = False
    truncated = False
    for _ in range(max(1, max_steps)):
        acting = parse_acting_player_id(cpp_inner, obs)
        if acting == human_player_id:
            policy = coach_agent
        elif acting == 1:
            policy = coach_agent if human_player_id != 1 else opponent_agent
        else:
            policy = opponent_agent if opponent_agent is not None else coach_agent

        if policy is not None and hasattr(policy, "act_greedy"):
            action = str(policy.act_greedy(obs))
        elif policy is not None and hasattr(policy, "act"):
            action = str(policy.act(obs))
        else:
            action = str(cpp_inner.sample_action())

        step = cpp_inner.step(action)
        obs = step.observation
        terminated = bool(step.terminated)
        truncated = bool(step.truncated)
        if terminated or truncated:
            break

    gs = getattr(cpp_inner, "_gs", None)
    if gs is None:
        return None
    if bool(getattr(gs, "game_over", False)):
        winner = int(getattr(gs, "winner", -1))
        if winner < 0:
            return None
        return (winner + 1) == int(human_player_id)
    return None


def compute_cpp_action_win_rates(
    cpp_env: Any,
    talishar_env: Any,
    obs: Any,
    *,
    human_player_id: int,
    coach_agent: Any,
    opponent_agent: Optional[Any] = None,
    legal_actions: Optional[list[dict[str, Any]]] = None,
    rollouts_per_action: int = 4,
    max_steps: int = 60,
) -> dict[int, float]:
    """Estimate win rate per action index using short C++ rollouts."""
    if cpp_env is None or rollouts_per_action <= 0:
        return {}

    legal = normalize_legal_entries(legal_actions, obs=obs)
    if not legal:
        return {}

    inner = _cpp_inner_env(cpp_env)
    win_rates: dict[int, float] = {}
    for entry in legal:
        action_index = int(entry["index"])
        wins = 0
        played = 0
        for _ in range(rollouts_per_action):
            _seed_cpp_from_talishar(
                inner,
                talishar_env,
                human_player_id=human_player_id,
            )
            outcome = _playout_cpp_after_action(
                inner,
                first_action=str(action_index),
                human_player_id=human_player_id,
                coach_agent=coach_agent,
                opponent_agent=opponent_agent,
                max_steps=max_steps,
            )
            if outcome is None:
                continue
            played += 1
            if outcome:
                wins += 1
        if played > 0:
            win_rates[action_index] = wins / played
    return win_rates


@dataclass
class LiveActionCoach:
    """Combines agent policy scores with optional C++ rollout win rates."""

    coach_agent: Any
    opponent_agent: Optional[Any] = None
    cpp_env: Any = None
    rollouts_per_action: int = 4
    max_rollout_steps: int = 60

    @classmethod
    def try_create(
        cls,
        *,
        coach_agent: Any,
        opponent_agent: Optional[Any],
        base_url: str,
        game_format: str,
        deck1: str,
        deck2: str,
        rollouts_per_action: int = 4,
        max_rollout_steps: int = 60,
        cpp_engine_dir: Optional[str] = None,
    ) -> LiveActionCoach:
        cpp_env = try_create_cpp_eval_environment(
            base_url=base_url,
            game_format=game_format,
            lookup_deck1=deck1,
            lookup_deck2=deck2,
            cpp_engine_dir=cpp_engine_dir,
            max_turns=max_rollout_steps,
            use_cpp_engine=True,
        )
        return cls(
            coach_agent=coach_agent,
            opponent_agent=opponent_agent,
            cpp_env=cpp_env,
            rollouts_per_action=rollouts_per_action,
            max_rollout_steps=max_rollout_steps,
        )

    def close(self) -> None:
        if self.cpp_env is not None:
            try:
                self.cpp_env.close()
            except Exception:
                pass
            self.cpp_env = None

    def build_hints(
        self,
        talishar_env: Any,
        obs: Any,
        *,
        human_player_id: int,
        legal_actions: Optional[list[dict[str, Any]]] = None,
    ) -> list[ActionCoachHint]:
        legal = normalize_legal_entries(legal_actions, obs=obs)
        if not legal and hasattr(talishar_env, "_legal_actions"):
            state = getattr(talishar_env, "_last_state", None)
            if isinstance(state, dict):
                legal = normalize_legal_entries(
                    talishar_env._legal_actions(state),
                    obs=obs,
                )

        hints = compute_agent_policy_scores(
            self.coach_agent,
            obs,
            legal_actions=legal,
        )
        if not hints:
            return []

        win_rates = compute_cpp_action_win_rates(
            self.cpp_env,
            talishar_env,
            obs,
            human_player_id=human_player_id,
            coach_agent=self.coach_agent,
            opponent_agent=self.opponent_agent,
            legal_actions=legal,
            rollouts_per_action=self.rollouts_per_action,
            max_steps=self.max_rollout_steps,
        )

        merged: list[ActionCoachHint] = []
        for hint in hints:
            merged.append(
                ActionCoachHint(
                    index=hint.index,
                    label=hint.label,
                    zone=hint.zone,
                    match_text=hint.match_text,
                    policy_pct=hint.policy_pct,
                    win_pct=win_rates.get(hint.index),
                    is_best=hint.is_best,
                )
            )
        return merged
