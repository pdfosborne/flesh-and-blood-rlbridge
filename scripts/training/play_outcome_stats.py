"""Classify play/eval episode outcomes and compute win-rate summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class RenderEpisodeResult:
    """Resolved render-episode outcome with absolute seat HP."""

    outcome: str
    p1_hp: Optional[float]
    p2_hp: Optional[float]
    winning_seat: Optional[int] = None

NominalHeroSlot = Literal["hero1", "hero2"]

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


def classify_p1_outcome_from_talishar_winner(winner: int) -> Optional[str]:
    """Map Talishar HTTP ``winner`` (1=P1, 2=P2) to a P1-seat outcome."""
    if winner == 1:
        return OUTCOME_WIN
    if winner == 2:
        return OUTCOME_LOSS
    if winner <= 0:
        return None
    return OUTCOME_DRAW


def hero_display_name(raw: str) -> str:
    """Human-readable hero label from a deck stem or hero id."""
    slug = str(raw or "").strip().removeprefix("fab_")
    if not slug:
        return "?"
    if "-vs-" in slug:
        slug = slug.split("-vs-", 1)[0]
    slug = slug.replace("-", "_")
    if slug.startswith("precon_"):
        parts = slug.split("_")
        slug = parts[-1] if len(parts) >= 2 else slug
    elif "_" in slug:
        slug = slug.rsplit("_", 1)[-1]
    return slug.replace("_", " ").title() or "?"


def render_outcome_banner_label(
    result: RenderEpisodeResult,
    *,
    p1_hero: str = "",
    p2_hero: str = "",
) -> tuple[str, tuple[int, int, int]]:
    """Return ``(banner_text, rgb)`` for a render end-state overlay."""
    if result.outcome == OUTCOME_DRAW:
        return "DRAW", (250, 204, 21)
    if result.outcome in {OUTCOME_TIMEOUT, OUTCOME_ERROR, "stall_timeout"}:
        text = result.outcome.upper().replace("_", " ")
        return text, (249, 115, 22)

    winning_seat = result.winning_seat
    if winning_seat is None:
        if result.outcome == OUTCOME_WIN:
            winning_seat = 1
        elif result.outcome == OUTCOME_LOSS:
            winning_seat = 2

    if winning_seat == 1:
        hero = hero_display_name(p1_hero or "P1")
        return f"{hero} Won", (34, 197, 94)
    if winning_seat == 2:
        hero = hero_display_name(p2_hero or "P2")
        return f"{hero} Won", (34, 197, 94)
    return result.outcome.upper(), (200, 200, 200)


def _parse_talishar_winner_seat(raw: Any) -> Optional[int]:
    try:
        winner = int(raw)
    except (TypeError, ValueError):
        return None
    if winner in (1, 2):
        return winner
    return None


def talishar_winner_seat_from_env(env: Any) -> Optional[int]:
    """Return Talishar winner seat (1 or 2) from dual-seat or last-state snapshots."""
    if env is None or getattr(env, "_using_cpp", False):
        return None
    try:
        from flesh_and_blood_rlbridge.game_state_parity import extract_talishar_state

        snap = extract_talishar_state(env)
        if snap:
            winner = _parse_talishar_winner_seat(snap.get("winner", -1))
            if winner is not None:
                return winner
    except Exception:
        pass

    last_state = getattr(env, "_last_state", None)
    if isinstance(last_state, dict):
        winner = _parse_talishar_winner_seat(last_state.get("winner"))
        if winner is not None:
            return winner
    return None


def absolute_p1_p2_hp_from_dual_seat_env(
    env: Any,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return ``(p1_hp, p2_hp, talishar_winner)`` from both Talishar seat snapshots."""
    if env is None or getattr(env, "_using_cpp", False):
        return None, None, None
    try:
        from flesh_and_blood_rlbridge.game_state_parity import extract_talishar_state

        snap = extract_talishar_state(env)
        if not snap:
            return None, None, None
        p1_hp = snap.get("p1_health")
        p2_hp = snap.get("p2_health")
        if p1_hp is None or p2_hp is None:
            return None, None, None
        winner = _parse_talishar_winner_seat(snap.get("winner", -1))
        return int(p1_hp), int(p2_hp), winner
    except Exception:
        return None, None, None


def infer_render_episode_outcome(
    obs: Any,
    *,
    terminated: bool,
    truncated: bool,
    env: Any = None,
) -> RenderEpisodeResult:
    """Resolve render outcome using dual-seat HP and Talishar ``winner`` when available."""
    p1_hp: Optional[float] = None
    p2_hp: Optional[float] = None
    talishar_winner: Optional[int] = None

    dual_p1, dual_p2, dual_winner = absolute_p1_p2_hp_from_dual_seat_env(env)
    if dual_p1 is not None and dual_p2 is not None:
        p1_hp, p2_hp = float(dual_p1), float(dual_p2)
        talishar_winner = dual_winner
    else:
        p1_deck = p2_deck = None
        if env is not None:
            env_p1, env_p2 = absolute_p1_p2_hp_from_env(env)
            if env_p1 is not None and env_p2 is not None:
                p1_hp, p2_hp = float(env_p1), float(env_p2)
            p1_deck, p2_deck = absolute_p1_p2_deck_from_env(env)
        if p1_hp is None or p2_hp is None:
            obs_p1, obs_p2 = absolute_p1_p2_hp_from_obs(obs)
            if obs_p1 is not None and obs_p2 is not None:
                p1_hp, p2_hp = obs_p1, obs_p2
        if p1_deck is None or p2_deck is None:
            p1_deck, p2_deck = absolute_p1_p2_deck_from_obs(obs)
        talishar_winner = talishar_winner_seat_from_env(env)

    outcome = classify_p1_episode_outcome(
        p1_hp=p1_hp,
        p2_hp=p2_hp,
        terminated=terminated,
        truncated=truncated,
    )

    if terminated and talishar_winner is not None:
        winner_outcome = classify_p1_outcome_from_talishar_winner(talishar_winner)
        if winner_outcome is not None:
            outcome = winner_outcome

    winning_seat: Optional[int] = None
    if outcome == OUTCOME_WIN:
        winning_seat = 1
    elif outcome == OUTCOME_LOSS:
        winning_seat = 2
    elif talishar_winner in (1, 2):
        winning_seat = talishar_winner

    return RenderEpisodeResult(
        outcome=outcome,
        p1_hp=p1_hp,
        p2_hp=p2_hp,
        winning_seat=winning_seat,
    )


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


def winning_hero_id_from_seat_outcome(
    outcome: str,
    *,
    active_p1_hero: str,
    active_p2_hero: str,
) -> Optional[str]:
    """Return the winning hero id from a P1-seat classified outcome."""
    if outcome == OUTCOME_WIN:
        return active_p1_hero
    if outcome == OUTCOME_LOSS:
        return active_p2_hero
    return None


def nominal_hero_slot(
    hero_id: str,
    *,
    hero1_id: str,
    hero2_id: str,
) -> Optional[NominalHeroSlot]:
    """Map a hero id to nominal hero1 / hero2 slot."""
    if hero_id == hero1_id:
        return "hero1"
    if hero_id == hero2_id:
        return "hero2"
    return None


def agent_won_seat_outcome(outcome: str, *, agent_on_p1_seat: bool) -> bool:
    """True when the agent (on P1 or P2 seat) won the episode."""
    if agent_on_p1_seat:
        return outcome == OUTCOME_WIN
    return outcome == OUTCOME_LOSS


def agent_hero_id_for_episode(
    *,
    agent_on_p1_seat: bool,
    use_swap: bool,
    nominal_hero1: str,
    nominal_hero2: str,
) -> str:
    """Nominal hero id the agent controls this episode."""
    if agent_on_p1_seat:
        return nominal_hero2 if use_swap else nominal_hero1
    return nominal_hero1 if use_swap else nominal_hero2


@dataclass
class OutcomeCounters:
    """Dual seat + nominal-hero outcome accumulators."""

    p1_wins: int = 0
    p2_wins: int = 0
    hero1_wins: int = 0
    hero2_wins: int = 0
    agent_hero1_wins: int = 0
    agent_hero2_wins: int = 0
    agent_hero1_eps: int = 0
    agent_hero2_eps: int = 0
    draws: int = 0
    timeouts: int = 0
    errors: int = 0
    nominal_hero1: str = ""
    nominal_hero2: str = ""

    def record_seat_outcome(
        self,
        outcome: str,
        *,
        active_p1_hero: str,
        active_p2_hero: str,
        nominal_hero1: Optional[str] = None,
        nominal_hero2: Optional[str] = None,
    ) -> None:
        if nominal_hero1 is not None:
            self.nominal_hero1 = nominal_hero1
        if nominal_hero2 is not None:
            self.nominal_hero2 = nominal_hero2

        if outcome == OUTCOME_WIN:
            self.p1_wins += 1
        elif outcome == OUTCOME_LOSS:
            self.p2_wins += 1
        elif outcome == OUTCOME_DRAW:
            self.draws += 1
        elif outcome == OUTCOME_ERROR:
            self.errors += 1
        else:
            self.timeouts += 1

        winner_id = winning_hero_id_from_seat_outcome(
            outcome,
            active_p1_hero=active_p1_hero,
            active_p2_hero=active_p2_hero,
        )
        if winner_id is None:
            return
        slot = nominal_hero_slot(
            winner_id,
            hero1_id=self.nominal_hero1,
            hero2_id=self.nominal_hero2,
        )
        if slot == "hero1":
            self.hero1_wins += 1
        elif slot == "hero2":
            self.hero2_wins += 1

    def record_agent_hero_outcome(
        self,
        outcome: str,
        *,
        agent_on_p1_seat: bool,
        use_swap: bool,
        nominal_hero1: Optional[str] = None,
        nominal_hero2: Optional[str] = None,
    ) -> None:
        if nominal_hero1 is not None:
            self.nominal_hero1 = nominal_hero1
        if nominal_hero2 is not None:
            self.nominal_hero2 = nominal_hero2
        hero_id = agent_hero_id_for_episode(
            agent_on_p1_seat=agent_on_p1_seat,
            use_swap=use_swap,
            nominal_hero1=self.nominal_hero1,
            nominal_hero2=self.nominal_hero2,
        )
        slot = nominal_hero_slot(
            hero_id,
            hero1_id=self.nominal_hero1,
            hero2_id=self.nominal_hero2,
        )
        if slot == "hero1":
            self.agent_hero1_eps += 1
        elif slot == "hero2":
            self.agent_hero2_eps += 1
        if not agent_won_seat_outcome(outcome, agent_on_p1_seat=agent_on_p1_seat):
            return
        if slot == "hero1":
            self.agent_hero1_wins += 1
        elif slot == "hero2":
            self.agent_hero2_wins += 1

    def to_summary(
        self,
        episodes: int,
        *,
        deck_swap_eval: bool = False,
        track_agent: bool = False,
    ) -> dict[str, Any]:
        total = max(1, int(episodes))
        decided = max(1, self.p1_wins + self.p2_wins + self.draws)
        heroes_block: dict[str, Any] = {
            "hero1_wins": self.hero1_wins,
            "hero2_wins": self.hero2_wins,
            "draws": self.draws,
            "timeouts": self.timeouts + self.errors,
            "errors": self.errors,
            "hero1_win_rate": self.hero1_wins / total,
            "hero2_win_rate": self.hero2_wins / total,
            "draw_rate": self.draws / total,
            "timeout_rate": (self.timeouts + self.errors) / total,
            "win_rate_decided": self.hero1_wins / decided,
        }
        if track_agent:
            heroes_block["agent_hero1_wins"] = self.agent_hero1_wins
            heroes_block["agent_hero2_wins"] = self.agent_hero2_wins
            heroes_block["agent_hero1_eps"] = self.agent_hero1_eps
            heroes_block["agent_hero2_eps"] = self.agent_hero2_eps
            heroes_block["agent_hero1_win_rate"] = (
                self.agent_hero1_wins / max(1, self.agent_hero1_eps)
            )
            heroes_block["agent_hero2_win_rate"] = (
                self.agent_hero2_wins / max(1, self.agent_hero2_eps)
            )
        seats_block: dict[str, Any] = {
            "p1_wins": self.p1_wins,
            "p2_wins": self.p2_wins,
            "draws": self.draws,
            "timeouts": self.timeouts + self.errors,
            "errors": self.errors,
            "p1_win_rate": self.p1_wins / total,
            "p2_win_rate": self.p2_wins / total,
            "draw_rate": self.draws / total,
            "timeout_rate": (self.timeouts + self.errors) / total,
            "win_rate_decided": self.p1_wins / decided,
        }
        summary: dict[str, Any] = {
            "episodes": total,
            "nominal_heroes": {
                "hero1": self.nominal_hero1,
                "hero2": self.nominal_hero2,
            },
            "heroes": heroes_block,
            "seats": seats_block,
            # Legacy top-level seat aliases
            "p1_wins": self.p1_wins,
            "p2_wins": self.p2_wins,
            "draws": self.draws,
            "timeouts": self.timeouts + self.errors,
            "errors": self.errors,
            "losses": self.p2_wins,
            "p1_win_rate": self.p1_wins / total,
            "p2_win_rate": self.p2_wins / total,
            "win_rate": self.p1_wins / total,
            "win_rate_decided": self.p1_wins / decided,
            "loss_rate": self.p2_wins / total,
            "draw_rate": self.draws / total,
            "timeout_rate": (self.timeouts + self.errors) / total,
            "hero1_win_rate": self.hero1_wins / total,
            "hero2_win_rate": self.hero2_wins / total,
            "deck_swap_eval": deck_swap_eval,
        }
        return summary


def classify_and_record_fast_episode(
    state: dict[str, Any],
    counters: OutcomeCounters,
    *,
    active_p1_hero: str,
    active_p2_hero: str,
    nominal_hero1: str,
    nominal_hero2: str,
    max_steps_reached: bool = False,
) -> tuple[str, Optional[str]]:
    """Classify a fast-path episode and update *counters*."""
    outcome, anomaly = classify_p1_fast_episode_outcome(
        state,
        max_steps_reached=max_steps_reached,
    )
    counters.record_seat_outcome(
        outcome,
        active_p1_hero=active_p1_hero,
        active_p2_hero=active_p2_hero,
        nominal_hero1=nominal_hero1,
        nominal_hero2=nominal_hero2,
    )
    return outcome, anomaly


def legacy_hero_rates_from_seat_summary(
    summary: dict[str, Any],
    *,
    deck_swap_eval: bool = False,
) -> tuple[Optional[float], Optional[float]]:
    """Derive hero win rates from legacy seat-only summaries when deck swap was used."""
    heroes = summary.get("heroes")
    if isinstance(heroes, dict):
        h1 = heroes.get("hero1_win_rate")
        h2 = heroes.get("hero2_win_rate")
        if h1 is not None and h2 is not None:
            return float(h1), float(h2)

    p1_wr = summary.get("p1_win_rate")
    p2_wr = summary.get("p2_win_rate")
    if p1_wr is None or p2_wr is None:
        return None, None
    p1 = float(p1_wr)
    p2 = float(p2_wr)
    if not deck_swap_eval:
        return p1, p2
    # With alternating deck swap, seat rates average to 50/50 for symmetric matchups;
    # invert odd/even attribution is not recoverable from aggregates alone — use
    # the midpoint as a conservative fallback for old records.
    return (p1 + p2) / 2.0, (p1 + p2) / 2.0


def summarize_hero_outcomes(
    counters: OutcomeCounters,
    *,
    episodes: Optional[int] = None,
) -> dict[str, Any]:
    """Build hero + seat summary from accumulated counters."""
    total = int(episodes) if episodes is not None else (
        counters.p1_wins
        + counters.p2_wins
        + counters.draws
        + counters.timeouts
        + counters.errors
    )
    return counters.to_summary(max(1, total))


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
