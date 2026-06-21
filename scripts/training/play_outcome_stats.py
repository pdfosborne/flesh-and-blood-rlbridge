"""Classify play/eval episode outcomes and compute win-rate summaries."""

from __future__ import annotations

from typing import Any, Optional

OUTCOME_WIN = "win"
OUTCOME_LOSS = "loss"
OUTCOME_DRAW = "draw"
OUTCOME_TIMEOUT = "timeout"


def classify_p1_episode_outcome(
    *,
    p1_hp: Optional[float | int] = None,
    p2_hp: Optional[float | int] = None,
    terminated: bool = False,
    truncated: bool = False,
    skipped: bool = False,
) -> str:
    """Classify one episode from P1's perspective.

    Wins and losses require lethal damage (opponent or player at 0 HP).
    Step-limit and other non-lethal endings count as timeouts, not wins
    from HP advantage alone.
    """
    if skipped:
        return OUTCOME_TIMEOUT
    if truncated:
        return OUTCOME_TIMEOUT
    if terminated:
        if p1_hp is None or p2_hp is None:
            return OUTCOME_TIMEOUT
        p1 = float(p1_hp)
        p2 = float(p2_hp)
        if p2 <= 0 and p1 > 0:
            return OUTCOME_WIN
        if p1 <= 0 and p2 > 0:
            return OUTCOME_LOSS
        if p1 <= 0 and p2 <= 0:
            return OUTCOME_DRAW
        # Stalemate / deck-out without lethal: both still alive.
        return OUTCOME_DRAW
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
    return {
        "episodes": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "timeout_rate": timeouts / total,
    }


DEFAULT_EVAL_STABILITY_WINDOW = 3
DEFAULT_EVAL_STABILITY_MAX_STD = 0.03
DEFAULT_EVAL_STABILITY_MIN_POINTS = 2


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
