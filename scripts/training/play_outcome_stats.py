"""Classify play/eval episode outcomes and compute win-rate summaries."""

from __future__ import annotations

import json
from typing import Any, Optional

OUTCOME_WIN = "win"
OUTCOME_LOSS = "loss"
OUTCOME_DRAW = "draw"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_ERROR = "error"


def classify_p1_outcome_from_engine_winner(winner: int) -> Optional[str]:
    """Map C++ ``winner`` (0=P1, 1=P2, -1=undecided) to a P1 outcome."""
    if winner == 0:
        return OUTCOME_WIN
    if winner == 1:
        return OUTCOME_LOSS
    if winner < 0:
        return None
    return OUTCOME_DRAW


def classify_p1_fast_episode_outcome(
    state: dict[str, Any],
    *,
    max_steps_reached: bool = False,
) -> tuple[str, Optional[str]]:
    """Classify a C++ fast-path episode; return ``(outcome, anomaly)``."""
    terminated = bool(state.get("terminated", False))
    truncated = bool(state.get("truncated", False))
    if max_steps_reached or (truncated and not terminated):
        return OUTCOME_TIMEOUT, None

    p1_hp = state.get("p1_health")
    p2_hp = state.get("p2_health")
    if p1_hp is None or p2_hp is None:
        return OUTCOME_ERROR, "missing p1_hp/p2_hp at episode end"

    outcome = classify_p1_episode_outcome(
        p1_hp=p1_hp,
        p2_hp=p2_hp,
        terminated=terminated,
        truncated=truncated,
    )

    if not terminated:
        return outcome, None

    winner_raw = state.get("winner", -1)
    try:
        winner = int(winner_raw)
    except (TypeError, ValueError):
        winner = -1
    if winner < 0:
        return outcome, None

    winner_outcome = classify_p1_outcome_from_engine_winner(winner)
    if winner_outcome is None:
        if outcome != OUTCOME_TIMEOUT:
            return outcome, f"unexpected winner code {winner}"
        return outcome, None

    if winner_outcome != outcome:
        return outcome, (
            f"engine winner={winner} ({winner_outcome}) disagrees with "
            f"HP outcome {outcome} (p1_hp={p1_hp}, p2_hp={p2_hp})"
        )
    return outcome, None


def _observation_dict(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    if isinstance(obs, str):
        try:
            loaded = json.loads(obs)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}
    return {}


def absolute_p1_p2_hp_from_obs(obs: Any) -> tuple[Optional[float], Optional[float]]:
    """Return fixed-seat P1/P2 HP from a Talishar-shaped observation.

    ``playerHealth`` / ``opponentHealth`` are from the acting player's view;
    swap them back when P2 has priority so win/loss classification is stable.
    """
    obs_data = _observation_dict(obs)
    player_hp = obs_data.get("playerHealth")
    opponent_hp = obs_data.get("opponentHealth")
    if player_hp is None or opponent_hp is None:
        return None, None
    p_player = float(player_hp or 0)
    p_opponent = float(opponent_hp or 0)
    acting = int(obs_data.get("actingPlayerID", 1) or 1)
    if acting == 2:
        return p_opponent, p_player
    return p_player, p_opponent


def _acting_player_id_from_env(env: Any) -> Optional[int]:
    acting_id = getattr(env, "_acting_player_id", None)
    if acting_id in (1, 2):
        return int(acting_id)
    return None


def _absolute_p1_p2_from_acting_view(
    player_value: Any,
    opponent_value: Any,
    *,
    acting_player_id: int,
) -> tuple[int, int]:
    p_player = int(player_value or 0)
    p_opponent = int(opponent_value or 0)
    if acting_player_id == 2:
        return p_opponent, p_player
    return p_player, p_opponent


def absolute_p1_p2_hp_from_env(env: Any) -> tuple[Optional[int], Optional[int]]:
    """Return absolute P1/P2 HP from a training or eval environment."""
    cpp_env = getattr(env, "_cpp_env", None)
    if cpp_env is not None:
        gs = getattr(cpp_env, "_gs", None)
        if gs is not None:
            return int(gs.p1_health), int(gs.p2_health)

    if getattr(env, "_using_cpp", False):
        return int(getattr(env, "_player_hp", 0)), int(getattr(env, "_opp_hp", 0))

    acting_id = _acting_player_id_from_env(env)
    player_hp = getattr(env, "_player_hp", None)
    opp_hp = getattr(env, "_opp_hp", None)
    if acting_id is not None and player_hp is not None and opp_hp is not None:
        return _absolute_p1_p2_from_acting_view(
            player_hp,
            opp_hp,
            acting_player_id=acting_id,
        )

    last_state = getattr(env, "_last_state", None)
    if isinstance(last_state, dict) and last_state:
        view = dict(last_state)
        if acting_id is not None:
            view["actingPlayerID"] = acting_id
        p1_hp, p2_hp = absolute_p1_p2_hp_from_obs(view)
        if p1_hp is not None and p2_hp is not None:
            return int(p1_hp), int(p2_hp)

    return None, None


def absolute_p1_p2_deck_from_obs(obs: Any) -> tuple[Optional[int], Optional[int]]:
    """Return fixed-seat P1/P2 deck counts from a Talishar-shaped observation."""
    obs_data = _observation_dict(obs)
    player_deck = obs_data.get("playerDeckCount", obs_data.get("player_deck_count"))
    opponent_deck = obs_data.get("opponentDeckCount", obs_data.get("opponent_deck_count"))
    if player_deck is None or opponent_deck is None:
        return None, None
    p_player = int(player_deck or 0)
    p_opponent = int(opponent_deck or 0)
    acting = int(obs_data.get("actingPlayerID", 1) or 1)
    if acting == 2:
        return p_opponent, p_player
    return p_player, p_opponent


def absolute_p1_p2_deck_from_env(env: Any) -> tuple[Optional[int], Optional[int]]:
    """Return absolute P1/P2 deck counts from a training or eval environment."""
    cpp_env = getattr(env, "_cpp_env", None)
    if cpp_env is not None:
        gs = getattr(cpp_env, "_gs", None)
        if gs is not None:
            return int(gs.p1_deck_size), int(gs.p2_deck_size)

    if getattr(env, "_using_cpp", False):
        gs = getattr(env, "_gs", None)
        if gs is not None:
            return int(gs.p1_deck_size), int(gs.p2_deck_size)

    last_state = getattr(env, "_last_state", None)
    if isinstance(last_state, dict) and last_state:
        view = dict(last_state)
        acting_id = _acting_player_id_from_env(env)
        if acting_id is not None:
            view["actingPlayerID"] = acting_id
        return absolute_p1_p2_deck_from_obs(view)

    return None, None


def classify_p1_episode_outcome(
    *,
    p1_hp: Optional[float | int] = None,
    p2_hp: Optional[float | int] = None,
    p1_deck: Optional[float | int] = None,
    p2_deck: Optional[float | int] = None,
    terminated: bool = False,
    truncated: bool = False,
    skipped: bool = False,
) -> str:
    """Classify one episode from P1's perspective.

    Wins and losses require lethal HP (opponent ≤ 0 while self > 0, or both ≤ 0
    for a draw). Termination with both players still above 0 HP is a timeout,
    even if the engine reports a winner flag. Step limits and missing HP also
    count as timeouts unless lethal HP is already decided.
    """
    del p1_deck, p2_deck  # HP-only; decks do not decide W/L.

    if skipped:
        return OUTCOME_TIMEOUT
    if truncated and not terminated:
        return OUTCOME_TIMEOUT
    if p1_hp is None or p2_hp is None:
        return OUTCOME_TIMEOUT

    p1 = float(p1_hp)
    p2 = float(p2_hp)
    if p1 <= 0 and p2 <= 0:
        return OUTCOME_DRAW
    if p2 <= 0:
        return OUTCOME_WIN
    if p1 <= 0:
        return OUTCOME_LOSS
    return OUTCOME_TIMEOUT


def summarize_p1_outcomes(
    outcomes: list[str],
    *,
    episodes: Optional[int] = None,
) -> dict[str, Any]:
    """Aggregate W/L/D/T counts; win% = wins / all episodes."""
    wins = sum(1 for outcome in outcomes if outcome == OUTCOME_WIN)
    losses = sum(1 for outcome in outcomes if outcome == OUTCOME_LOSS)
    draws = sum(1 for outcome in outcomes if outcome == OUTCOME_DRAW)
    timeouts = sum(
        1 for outcome in outcomes
        if outcome in {OUTCOME_TIMEOUT, "stall_timeout"}
    )
    counted = wins + losses + draws + timeouts
    total = int(episodes) if episodes is not None else counted
    if total < counted:
        timeouts += counted - total
    elif total > counted:
        timeouts += total - counted
    total = max(1, total) if total > 0 else 1
    decided = max(1, wins + losses + draws)
    return {
        "episodes": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "win_rate": wins / total,
        "win_rate_decided": wins / decided,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "timeout_rate": timeouts / total,
    }


DEFAULT_EVAL_STABILITY_WINDOW = 3
DEFAULT_EVAL_STABILITY_MAX_STD = 0.01
DEFAULT_EVAL_STABILITY_MIN_POINTS = 3


def rolling_std(values: list[float], window: int) -> Optional[float]:
    """Population std-dev of the last *window* values."""
    if len(values) < 2:
        return None
    recent = values[-window:]
    if len(recent) < 2:
        return None
    mean = sum(recent) / len(recent)
    variance = sum((x - mean) ** 2 for x in recent) / len(recent)
    return variance ** 0.5


def compute_eval_stability(
    checkpoint_win_rates: list[float],
    *,
    window: int = DEFAULT_EVAL_STABILITY_WINDOW,
    max_std: float = DEFAULT_EVAL_STABILITY_MAX_STD,
    min_points: int = DEFAULT_EVAL_STABILITY_MIN_POINTS,
    episodes_completed: Optional[int] = None,
    target_episodes: Optional[int] = None,
) -> dict[str, Any]:
    """Measure checkpoint-eval stability to judge training sufficiency.

    Uses rolling σ on fixed-opponent checkpoint eval win rates (not self-play
    train win%).  When σ stays below *max_std* for the last *window* checkpoints,
    the policy is treated as converged for the current episode budget.
    """
    rates = [float(r) for r in checkpoint_win_rates if r is not None]
    points = len(rates)
    std = rolling_std(rates, window) if points >= min_points else None
    mean_wr = (sum(rates[-window:]) / min(window, points)) if points else None

    converged = (
        points >= window
        and std is not None
        and std <= max_std
    )
    episodes_complete = (
        episodes_completed is not None
        and target_episodes is not None
        and target_episodes > 0
        and episodes_completed >= target_episodes
    )
    sufficient = converged or episodes_complete

    if points < min_points:
        status = "insufficient"
        label = "Need more checkpoints"
        detail = f"need ≥{min_points} eval points ({points} so far)"
    elif converged:
        status = "converged"
        label = "Converged"
        detail = f"σ={std:.1%} over last {min(window, points)} checkpoints"
    else:
        status = "learning"
        label = "Still learning"
        detail = f"σ={std:.1%} (target ≤{max_std:.1%})" if std is not None else "eval drifting"

    if sufficient and converged:
        recommendation = "Checkpoint eval stable — episode budget is sufficient."
    elif sufficient and episodes_complete and not converged:
        recommendation = (
            "Training budget complete but checkpoint eval still shifting — "
            "consider more episodes or a longer run."
        )
    elif converged and not episodes_complete:
        recommendation = (
            "Checkpoint eval looks stable early — you may have enough training "
            "already, or continue to the full budget for safety."
        )
    else:
        recommendation = "Keep training until checkpoint eval stabilizes."

    return {
        "status": status,
        "label": label,
        "detail": detail,
        "recommendation": recommendation,
        "converged": converged,
        "sufficient": sufficient,
        "episodes_complete": episodes_complete,
        "rolling_std": std,
        "mean_win_rate": mean_wr,
        "window": window,
        "max_std": max_std,
        "points": points,
    }


def win_rate_standard_error(wins: int, total: int) -> Optional[float]:
    """Binomial standard error for a win-rate estimate (W / all outcomes)."""
    if total <= 0:
        return None
    p = max(0.0, min(1.0, float(wins) / float(total)))
    return (p * (1.0 - p) / float(total)) ** 0.5


def win_rate_standard_error_from_rate(rate: float, total: int) -> Optional[float]:
    """Binomial SE when only the aggregate win rate and sample size are known."""
    if total <= 0:
        return None
    p = max(0.0, min(1.0, float(rate)))
    return (p * (1.0 - p) / float(total)) ** 0.5
