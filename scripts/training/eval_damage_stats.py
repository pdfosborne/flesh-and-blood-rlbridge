"""Aggregate per-card damage dealt/taken from combat tracker traces."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_PLAYER_TOOK_RE = re.compile(r"Player\s+(\d)\s+took\s+(\d+)\s+damage", re.IGNORECASE)
_COMBAT_HIT_RE = re.compile(r"hit for\s+(\d+)\s+damage", re.IGNORECASE)
_ABOUT_TO_TAKE_RE = re.compile(
    r"Player\s+(\d)\s+is about to take\s+(\d+)\s+damage from\s+([a-z0-9_]+)",
    re.IGNORECASE,
)
_ARCANE_FROM_CARD_RE = re.compile(
    r"Player\s+(\d)\s+is dealing\s+(\d+)\s+arcane damage from\s+([a-z0-9_]+)",
    re.IGNORECASE,
)
_CARD_DEALS_RE = re.compile(
    r"([a-z0-9_]+)\s+deals?\s+(\d+)\s+(?:arcane\s+)?damage",
    re.IGNORECASE,
)
_CARD_DEALT_RE = re.compile(
    r"([a-z0-9_]+)\s+dealt\s+(\d+)\s+damage",
    re.IGNORECASE,
)


def _absolute_hp(snapshot: dict[str, Any]) -> tuple[int, int]:
    acting = int(snapshot.get("acting_player_id", 1) or 1)
    player_hp = int(snapshot.get("player_health", 0) or 0)
    opponent_hp = int(snapshot.get("opponent_health", 0) or 0)
    if acting == 1:
        return player_hp, opponent_hp
    return opponent_hp, player_hp


class EvalDamageAccumulator:
    """Collect P1-centric damage dealt/taken with per-card attribution."""

    def __init__(self, *, deck_card_ids: set[str] | None = None) -> None:
        self._deck_card_ids = deck_card_ids or set()
        self.dealt_by_card: Counter[str] = Counter()
        self.taken_from_card: Counter[str] = Counter()
        self.total_dealt = 0
        self.total_taken = 0
        self._last_p1_attack: str | None = None
        self._last_p2_attack: str | None = None

    def ingest_trace(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self._ingest_event(event)

    def _ingest_event(self, event: dict[str, Any]) -> None:
        before = event.get("before") or {}
        after = event.get("after") or {}
        action = event.get("action") or {}
        action_class = str(event.get("action_class") or "")

        acting = int(before.get("acting_player_id", 1) or 1)
        card_id = str(action.get("card_id") or "").strip()
        if action_class == "attack" and card_id:
            if acting == 1:
                self._last_p1_attack = card_id
            elif acting == 2:
                self._last_p2_attack = card_id

        for line in event.get("combat_log_delta") or []:
            self._parse_log_line(str(line))

        # Fallback when Talishar omits explicit log lines (e.g. C++ engine).
        p1_before, p2_before = _absolute_hp(before)
        p1_after, p2_after = _absolute_hp(after)
        p1_loss = max(0, p1_before - p1_after)
        p2_loss = max(0, p2_before - p2_after)
        if p2_loss > 0:
            source = card_id if acting == 1 and action_class == "attack" and card_id else self._last_p1_attack
            if source:
                self._credit_dealt(source, p2_loss)
            else:
                self.total_dealt += p2_loss
        if p1_loss > 0:
            source = card_id if acting == 2 and action_class == "attack" and card_id else self._last_p2_attack
            if source:
                self._credit_taken(source, p1_loss)
            else:
                self.total_taken += p1_loss

        if int(before.get("turn_no", 0) or 0) != int(after.get("turn_no", 0) or 0):
            self._last_p1_attack = None
            self._last_p2_attack = None

    def _parse_log_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return

        match = _ABOUT_TO_TAKE_RE.search(text)
        if match:
            player = int(match.group(1))
            amount = int(match.group(2))
            card_id = match.group(3)
            if player == 2:
                self._credit_dealt(card_id, amount)
            elif player == 1:
                self._credit_taken(card_id, amount)
            return

        match = _ARCANE_FROM_CARD_RE.search(text)
        if match:
            player = int(match.group(1))
            amount = int(match.group(2))
            card_id = match.group(3)
            if player == 1:
                self._credit_dealt(card_id, amount)
            elif player == 2:
                self._credit_taken(card_id, amount)
            return

        match = _PLAYER_TOOK_RE.search(text)
        if match:
            player = int(match.group(1))
            amount = int(match.group(2))
            if player == 2:
                source = self._last_p1_attack
                if source:
                    self._credit_dealt(source, amount)
                else:
                    self.total_dealt += amount
            elif player == 1:
                source = self._last_p2_attack
                if source:
                    self._credit_taken(source, amount)
                else:
                    self.total_taken += amount
            return

        match = _COMBAT_HIT_RE.search(text)
        if match:
            amount = int(match.group(1))
            if self._last_p1_attack:
                self._credit_dealt(self._last_p1_attack, amount)
            return

        for pattern in (_CARD_DEALS_RE, _CARD_DEALT_RE):
            match = pattern.search(text)
            if match:
                card_id = match.group(1)
                amount = int(match.group(2))
                if card_id in self._deck_card_ids:
                    self._credit_dealt(card_id, amount)
                else:
                    self._credit_taken(card_id, amount)
                return

    def _credit_dealt(self, card_id: str, amount: int) -> None:
        if amount <= 0 or not card_id:
            return
        self.dealt_by_card[card_id] += amount
        self.total_dealt += amount

    def _credit_taken(self, card_id: str, amount: int) -> None:
        if amount <= 0 or not card_id:
            return
        self.taken_from_card[card_id] += amount
        self.total_taken += amount

    def to_dict(self, *, top_k: int = 12) -> dict[str, Any]:
        return {
            "total_dealt": int(self.total_dealt),
            "total_taken": int(self.total_taken),
            "cards_dealt": _top_card_rows(self.dealt_by_card, top_k=top_k),
            "cards_taken_from": _top_card_rows(self.taken_from_card, top_k=top_k),
        }


def _top_card_rows(counter: Counter[str], *, top_k: int) -> list[dict[str, Any]]:
    return [
        {"card_id": card_id, "damage": int(amount)}
        for card_id, amount in counter.most_common(top_k)
        if amount > 0
    ]


def merge_damage_breakdowns(breakdowns: list[dict[str, Any]], *, top_k: int = 12) -> dict[str, Any]:
    """Merge per-episode breakdown dicts produced by EvalDamageAccumulator."""
    dealt = Counter[str]()
    taken = Counter[str]()
    total_dealt = 0
    total_taken = 0
    for row in breakdowns:
        total_dealt += int(row.get("total_dealt", 0) or 0)
        total_taken += int(row.get("total_taken", 0) or 0)
        for entry in row.get("cards_dealt") or []:
            dealt[str(entry.get("card_id") or "")] += int(entry.get("damage", 0) or 0)
        for entry in row.get("cards_taken_from") or []:
            taken[str(entry.get("card_id") or "")] += int(entry.get("damage", 0) or 0)
    return {
        "episodes": len(breakdowns),
        "total_dealt": int(total_dealt),
        "total_taken": int(total_taken),
        "avg_dealt_per_episode": round(total_dealt / max(1, len(breakdowns)), 2),
        "avg_taken_per_episode": round(total_taken / max(1, len(breakdowns)), 2),
        "cards_dealt": _top_card_rows(dealt, top_k=top_k),
        "cards_taken_from": _top_card_rows(taken, top_k=top_k),
    }
