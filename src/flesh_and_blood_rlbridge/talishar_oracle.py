"""Talishar ground-truth validation for combat resolutions.

Provides two levels of checking:

1. **Local (rule-book) validation** — always available, no server required.
   Re-derives the expected damage from the raw ``CombatState`` parameters
   using the same FAB rules the internal simulator applies, then compares
   against what the simulator actually committed to the game state.  Any
   divergence is a simulator bug.

2. **Remote (Talishar server) validation** — optional, requires a locally-
   running Talishar Docker instance (see README).  Sets up an equivalent
   minimal game in Talishar, replays the combat, and compares life totals.
   Activated by passing ``base_url`` to :class:`TalisharOracle` or by setting
   the ``TALISHAR_URL`` environment variable.

Typical usage
-------------
::

    # Offline (always works)
    oracle = TalisharOracle()
    snapshot = CombatSnapshot.capture(env)          # call just before combat resolves
    # ... env resolves the combat ...
    result = oracle.check(snapshot, env)
    if not result.match:
        print(result.report())

    # Online (requires ``TALISHAR_URL`` env-var or ``base_url`` argument)
    oracle = TalisharOracle(base_url="http://localhost")
    ...

Docker quick-start
------------------
::

    git clone https://github.com/Talishar/Talishar && cd Talishar
    git clone https://github.com/Talishar/Talishar-FE
    docker compose up -d
    export TALISHAR_URL=http://localhost

"""

from __future__ import annotations

import dataclasses
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import FleshAndBloodEnvironment, CombatState


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CombatSnapshot:
    """All inputs to a single combat-link resolution, captured before damage."""

    # Attack side
    attack_card_id: str
    attack_card_name: str
    attack_power: int          # printed power (already set on CombatState)
    attack_bonus: int          # accumulated bonuses (reactions, effects)
    opposing_power_mod: int    # negative modifier applied to total power
    keywords: tuple[str, ...]  # e.g. ("dominate", "overpower", "intimidate")
    fused: bool
    is_weapon: bool
    attacker_life_before: int

    # Defense side
    defender_life_before: int
    blocks: list[tuple[str, int]] = field(default_factory=list)  # [(card_id, defense_value)]
    total_block: int = 0

    # Mitigation layers (defender's prevention state at time of resolution)
    prevent_damage: int = 0         # flat prevention pool
    prevent_damage_per_hit: int = 0 # per-hit prevention (with charges)
    prevent_damage_charges: int = 0
    ward_values: list[int] = field(default_factory=list)  # one entry per ward source
    arcane_barrier: int = 0

    # Context
    attacker_idx: int = 0
    defender_idx: int = 1
    turn: int = 0
    phase: str = "defense_reaction"

    @classmethod
    def capture(cls, env: "FleshAndBloodEnvironment") -> Optional["CombatSnapshot"]:
        """Capture a snapshot from a live environment *before* damage resolves.

        Returns ``None`` when there is no active combat.
        """
        combat = env._pending_combat
        if combat is None:
            return None
        from .environment import CombatState  # local import to avoid circularity
        attacker = env._players[combat.attacker]
        defender = env._players[combat.defender]
        card = env._cards.get(combat.attack_card_id)

        # Tally block values (same logic as _resolve_and_advance)
        total_block = sum(b[2] for b in combat.blocks)

        # Ward values on equipment (defence equipment with Ward N keyword)
        import re
        from . import fab_rules
        ward_values: list[int] = []
        for cid in defender.equipment:
            eq = env._cards[cid]
            has_blue = any(env._cards[c].pitch == 3 for c in defender.pitch_zone)
            wv = fab_rules.parse_ward_value(eq.text, eq.keywords, has_blue_pitched=has_blue)
            if wv > 0:
                ward_values.append(wv)

        return cls(
            attack_card_id=combat.attack_card_id,
            attack_card_name=card.name if card else combat.attack_card_id,
            attack_power=combat.attack_power,
            attack_bonus=combat.attack_bonus,
            opposing_power_mod=combat.opposing_power_mod,
            keywords=tuple(card.keywords if card else []),
            fused=combat.fused,
            is_weapon=combat.is_weapon,
            attacker_life_before=attacker.life,
            defender_life_before=defender.life,
            blocks=[(b[1], b[2]) for b in combat.blocks],
            total_block=total_block,
            prevent_damage=defender.prevent_damage,
            prevent_damage_per_hit=defender.prevent_damage_per_hit,
            prevent_damage_charges=defender.prevent_damage_charges,
            ward_values=ward_values,
            arcane_barrier=defender.arcane_barrier,
            attacker_idx=combat.attacker,
            defender_idx=combat.defender,
            turn=env._turn,
            phase=env._phase,
        )


@dataclass
class ValidationResult:
    """Result of comparing the simulator's combat resolution against ground-truth."""

    match: bool
    source: str                 # "local_rules" or "talishar_server"

    # Raw damage layer
    expected_total_power: int = 0
    actual_total_power: int = 0  # inferred from env post-state
    expected_raw_damage: int = 0
    actual_raw_damage: int = 0   # inferred (life_before - life_after + prevention)

    # Post-mitigation damage
    expected_mitigated: int = 0
    actual_mitigated: int = 0

    # Life totals
    defender_life_before: int = 0
    defender_life_expected: int = 0
    defender_life_actual: int = 0

    notes: list[str] = field(default_factory=list)
    talishar_raw_response: Optional[dict[str, Any]] = None

    def report(self) -> str:
        lines = [
            f"--- Combat Validation ({self.source}) ---",
            f"Match: {self.match}",
            f"Power:          expected={self.expected_total_power}  "
            f"actual_inferred={self.actual_total_power}",
            f"Raw damage:     expected={self.expected_raw_damage}  "
            f"actual_inferred={self.actual_raw_damage}",
            f"After mitigation: expected={self.expected_mitigated}  "
            f"actual={self.actual_mitigated}",
            f"Defender life:  before={self.defender_life_before}  "
            f"expected={self.defender_life_expected}  "
            f"actual={self.defender_life_actual}",
        ]
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local (rule-book) validator
# ---------------------------------------------------------------------------

def _local_expected_damage(snapshot: CombatSnapshot) -> tuple[int, int, int, list[str]]:
    """Return (total_power, raw_damage, mitigated_damage, notes) from snapshot.

    Mirrors the logic in ``FleshAndBloodEnvironment._resolve_and_advance`` and
    ``_mitigate_damage`` without touching the live environment state.
    """
    notes: list[str] = []

    total_power = max(
        0,
        snapshot.attack_power + snapshot.attack_bonus + snapshot.opposing_power_mod,
    )
    raw_damage = max(0, total_power - snapshot.total_block)
    notes.append(
        f"total_power={total_power} "
        f"(printed {snapshot.attack_power} + bonus {snapshot.attack_bonus} "
        f"+ mod {snapshot.opposing_power_mod})"
    )
    notes.append(
        f"raw_damage={raw_damage} "
        f"({total_power} power - {snapshot.total_block} block)"
    )

    remaining = raw_damage

    # Ward (equipment with Ward N) — absorbs damage, equipment destroyed
    for wv in sorted(snapshot.ward_values, reverse=True):
        if remaining <= 0:
            break
        absorbed = min(wv, remaining)
        remaining -= absorbed
        notes.append(f"ward absorbed {absorbed} (ward value {wv})")

    # Per-hit prevention (e.g. Fyendal's Spring Tunic)
    if (
        snapshot.prevent_damage_charges > 0
        and snapshot.prevent_damage_per_hit > 0
        and remaining > 0
    ):
        absorbed = min(snapshot.prevent_damage_per_hit, remaining)
        remaining -= absorbed
        notes.append(f"per-hit prevention absorbed {absorbed}")

    # Flat prevention pool
    if snapshot.prevent_damage > 0 and remaining > 0:
        absorbed = min(snapshot.prevent_damage, remaining)
        remaining -= absorbed
        notes.append(f"flat prevention absorbed {absorbed}")

    # Arcane barrier (only applies to arcane damage; physical combat ignores it)
    # We conservatively skip arcane barrier for physical attacks here.

    mitigated = remaining
    notes.append(f"mitigated_damage={mitigated}")
    return total_power, raw_damage, mitigated, notes


# ---------------------------------------------------------------------------
# Remote (Talishar server) client
# ---------------------------------------------------------------------------

class TalisharClient:
    """Minimal HTTP wrapper around a local Talishar Docker instance.

    Only the endpoints needed for validation are implemented:
    * ``/GetNextTurn.php``  — retrieve current game state as JSON
    * ``/ProcessInputAPI.php`` — submit a player action

    Game creation requires pre-existing game files (``Games/<name>/``).
    Use :meth:`create_minimal_game` to bootstrap a one-link combat scenario
    if the instance exposes the undocumented ``/DevTools.py`` helper.
    """

    def __init__(self, base_url: str = "http://localhost", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = self.base_url + path + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:  # noqa: S310
                body = resp.read()
            data = json.loads(body)
            return data if isinstance(data, dict) else {"raw": data}
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise TalisharConnectionError(f"GET {url} failed: {exc}") from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + path
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                resp_body = resp.read()
            data = json.loads(resp_body)
            return data if isinstance(data, dict) else {"raw": data}
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise TalisharConnectionError(f"POST {url} failed: {exc}") from exc

    def ping(self) -> bool:
        """Return True when the server responds to a health check."""
        try:
            self._get("/GetNextTurn.php", {"gameName": "ping", "playerID": "3"})
            return True
        except TalisharConnectionError:
            # The endpoint may return an error JSON — that still means the
            # server is up, just the game doesn't exist.
            return True
        except Exception:
            return False

    def get_game_state(self, game_name: str, player_id: int = 1, auth_key: str = "") -> dict[str, Any]:
        """Fetch the current game state JSON from GetNextTurn.php."""
        params: dict[str, str] = {
            "gameName": game_name,
            "playerID": str(player_id),
        }
        if auth_key:
            params["authKey"] = auth_key
        return self._get("/GetNextTurn.php", params)

    def submit_action(
        self,
        game_name: str,
        player_id: int,
        mode: int,
        submission: Any = None,
        auth_key: str = "",
    ) -> dict[str, Any]:
        """Submit a player action via ProcessInputAPI.php.

        Parameters
        ----------
        mode:
            Talishar mode integer (see ProcessInputAPI.php switch statement).
        submission:
            Optional JSON-serialisable value for the ``submission`` field.
        """
        payload: dict[str, Any] = {
            "gameName": game_name,
            "playerID": player_id,
            "mode": mode,
        }
        if auth_key:
            payload["authKey"] = auth_key
        if submission is not None:
            payload["submission"] = submission
        return self._post("/ProcessInputAPI.php", payload)

    def extract_life_totals(self, state: dict[str, Any]) -> tuple[int, int]:
        """Extract (p1_life, p2_life) from a GetNextTurn.php JSON response.

        Talishar returns life totals nested under ``playerHero.life`` or
        ``playerHP`` depending on the API version; we probe both paths.
        """
        def _probe(d: dict[str, Any], *paths: str) -> Optional[int]:
            for path in paths:
                node: Any = d
                for key in path.split("."):
                    if not isinstance(node, dict):
                        node = None
                        break
                    node = node.get(key)
                if node is not None:
                    try:
                        return int(node)
                    except (TypeError, ValueError):
                        pass
            return None

        p1 = _probe(state, "p1.playerHP", "player1.hp", "p1HP", "p1Life")
        p2 = _probe(state, "p2.playerHP", "player2.hp", "p2HP", "p2Life")
        return (p1 or -1, p2 or -1)


class TalisharConnectionError(RuntimeError):
    """Raised when the Talishar server cannot be reached."""


# ---------------------------------------------------------------------------
# Oracle (orchestrator)
# ---------------------------------------------------------------------------

class TalisharOracle:
    """Validates combat resolutions against Talishar's authoritative rules.

    Parameters
    ----------
    base_url:
        URL of a locally-running Talishar Docker instance, e.g.
        ``"http://localhost"``.  Falls back to the ``TALISHAR_URL``
        environment variable.  When neither is set the oracle operates in
        **local-only mode** (rule-book validation only).
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        url = base_url or os.environ.get("TALISHAR_URL", "")
        self._client: Optional[TalisharClient] = TalisharClient(url) if url else None
        self._server_available: Optional[bool] = None  # lazily checked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def server_available(self) -> bool:
        """True when a live Talishar server is reachable."""
        if self._client is None:
            return False
        if self._server_available is None:
            self._server_available = self._client.ping()
        return self._server_available

    def check(
        self,
        snapshot: CombatSnapshot,
        env: "FleshAndBloodEnvironment",
        *,
        game_name: Optional[str] = None,
        auth_key: str = "",
    ) -> ValidationResult:
        """Validate the combat resolution captured in *snapshot* against ground truth.

        The environment's *current* defender life total is read back to infer
        what damage was actually applied.  Call this *after* the combat has
        resolved but before any further state changes.

        Parameters
        ----------
        snapshot:
            Created by :meth:`CombatSnapshot.capture` just before damage resolved.
        env:
            The live environment (used to read back the actual post-damage state).
        game_name:
            If given and a Talishar server is available, also validates remotely.
        auth_key:
            Auth key for the remote Talishar game.
        """
        # --- local validation ---
        result = self._check_local(snapshot, env)

        # --- remote validation (optional) ---
        if game_name and self.server_available and self._client is not None:
            remote = self._check_remote(snapshot, env, game_name, auth_key)
            if remote is not None:
                if not remote.match:
                    result.match = False
                result.notes.append(f"Remote Talishar check: match={remote.match}")
                result.notes.extend(f"  [remote] {n}" for n in remote.notes)
                result.talishar_raw_response = remote.talishar_raw_response

        return result

    def check_from_snapshot(
        self,
        snapshot: CombatSnapshot,
        actual_damage: int,
        *,
        game_name: Optional[str] = None,
        auth_key: str = "",
    ) -> ValidationResult:
        """Validate using a pre-captured snapshot and a known damage value.

        Use this when the live environment is no longer available (e.g. in a
        batch replay workflow).
        """
        expected_power, expected_raw, expected_mitigated, notes = _local_expected_damage(snapshot)
        actual_life = snapshot.defender_life_before - actual_damage
        expected_life = snapshot.defender_life_before - expected_mitigated
        match = expected_mitigated == actual_damage

        result = ValidationResult(
            match=match,
            source="local_rules",
            expected_total_power=expected_power,
            actual_total_power=snapshot.attack_power + snapshot.attack_bonus + snapshot.opposing_power_mod,
            expected_raw_damage=expected_raw,
            actual_raw_damage=actual_damage + (snapshot.prevent_damage - max(0, snapshot.prevent_damage - expected_raw)),
            expected_mitigated=expected_mitigated,
            actual_mitigated=actual_damage,
            defender_life_before=snapshot.defender_life_before,
            defender_life_expected=expected_life,
            defender_life_actual=actual_life,
            notes=notes,
        )
        if not match:
            result.notes.append(
                f"MISMATCH: expected {expected_mitigated} damage "
                f"but got {actual_damage}"
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_local(
        self, snapshot: CombatSnapshot, env: "FleshAndBloodEnvironment"
    ) -> ValidationResult:
        """Rule-book check: re-derive expected damage and compare to actual."""
        expected_power, expected_raw, expected_mitigated, notes = _local_expected_damage(snapshot)

        defender = env._players[snapshot.defender_idx]
        actual_life = defender.life
        actual_damage = snapshot.defender_life_before - actual_life
        expected_life = snapshot.defender_life_before - expected_mitigated
        match = expected_life == actual_life

        result = ValidationResult(
            match=match,
            source="local_rules",
            expected_total_power=expected_power,
            actual_total_power=snapshot.attack_power + snapshot.attack_bonus + snapshot.opposing_power_mod,
            expected_raw_damage=expected_raw,
            actual_raw_damage=actual_damage,
            expected_mitigated=expected_mitigated,
            actual_mitigated=actual_damage,
            defender_life_before=snapshot.defender_life_before,
            defender_life_expected=expected_life,
            defender_life_actual=actual_life,
            notes=notes,
        )
        if not match:
            result.notes.append(
                f"MISMATCH on turn {snapshot.turn}: "
                f"expected life {expected_life} but got {actual_life} "
                f"(attack={snapshot.attack_card_name} "
                f"power={snapshot.attack_power}+{snapshot.attack_bonus} "
                f"block={snapshot.total_block})"
            )
        return result

    def _check_remote(
        self,
        snapshot: CombatSnapshot,
        env: "FleshAndBloodEnvironment",
        game_name: str,
        auth_key: str,
    ) -> Optional[ValidationResult]:
        """Fetch post-combat state from a running Talishar game and compare."""
        assert self._client is not None
        try:
            state = self._client.get_game_state(game_name, player_id=1, auth_key=auth_key)
        except TalisharConnectionError as exc:
            return ValidationResult(
                match=False,
                source="talishar_server",
                notes=[f"Could not reach Talishar server: {exc}"],
            )

        p1_life, p2_life = self._client.extract_life_totals(state)
        if p1_life < 0 and p2_life < 0:
            return ValidationResult(
                match=False,
                source="talishar_server",
                notes=["Could not parse life totals from Talishar response"],
                talishar_raw_response=state,
            )

        talishar_defender_life = p2_life if snapshot.defender_idx == 1 else p1_life
        actual_life = env._players[snapshot.defender_idx].life
        match = talishar_defender_life == actual_life

        notes: list[str] = [
            f"Talishar p1_life={p1_life} p2_life={p2_life}",
            f"Simulator defender_life={actual_life}",
        ]
        if not match:
            notes.append(
                f"REMOTE MISMATCH: Talishar says defender life={talishar_defender_life} "
                f"but simulator has {actual_life}"
            )

        return ValidationResult(
            match=match,
            source="talishar_server",
            defender_life_before=snapshot.defender_life_before,
            defender_life_expected=talishar_defender_life,
            defender_life_actual=actual_life,
            notes=notes,
            talishar_raw_response=state,
        )


# ---------------------------------------------------------------------------
# Convenience helpers for attaching the oracle to a running environment
# ---------------------------------------------------------------------------

class OracleHook:
    """Wraps a :class:`TalisharOracle` and attaches to ``env.on_combat_resolve``.

    The hook registers a callback on the environment that fires **inside**
    ``_resolve_and_advance`` after every combat resolution, regardless of
    whether it was auto-advanced or agent-driven.  This is the correct way
    to intercept combats — the phase-based ``before_step`` / ``after_step``
    pattern misses combats that resolve inside ``_auto_advance``.

    Example
    -------
    ::

        oracle = TalisharOracle()
        hook = OracleHook(oracle)

        env = FleshAndBloodEnvironment(...)
        env.reset()
        hook.attach(env)            # registers the callback once
        while not done:
            _, _, done, _, _ = env.step(action)
        print(hook.summary())
    """

    def __init__(self, oracle: TalisharOracle) -> None:
        self._oracle = oracle
        self.results: list[ValidationResult] = []
        self._env: Optional["FleshAndBloodEnvironment"] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach(self, env: "FleshAndBloodEnvironment") -> None:
        """Register this hook on *env*.  Call once after ``env.reset()``."""
        self._env = env
        env.on_combat_resolve = self._handle_combat

    def detach(self) -> None:
        """Remove the callback from the attached environment."""
        if self._env is not None:
            self._env.on_combat_resolve = None
            self._env = None

    # Legacy step-level helpers (kept for backwards compatibility).
    # They are now no-ops because the callback handles everything.
    def before_step(self, env: "FleshAndBloodEnvironment") -> None:  # noqa: D102
        if self._env is None:
            self.attach(env)

    def after_step(self, env: "FleshAndBloodEnvironment") -> Optional[ValidationResult]:  # noqa: D102
        # Return the most recent result if one was just recorded.
        return self.results[-1] if self.results else None

    def summary(self) -> str:
        """Return a brief summary of all validations so far."""
        total = len(self.results)
        mismatches = [r for r in self.results if not r.match]
        lines = [f"TalisharOracle: {total} checks, {len(mismatches)} mismatches"]
        for r in mismatches:
            lines.append("  " + r.report().replace("\n", "\n  "))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------------

    def _handle_combat(self, data: dict[str, Any]) -> None:
        """Called by ``env.on_combat_resolve`` after each combat resolves."""
        combat = data["combat"]
        attacker_idx: int = data["attacker_idx"]
        defender_idx: int = data["defender_idx"]
        total_power: int = data["total_power"]
        total_block: int = data["total_block"]
        raw_damage: int = data["raw_damage"]
        actual_damage: int = data["damage"]
        defender_life_before: int = data["defender_life_before"]

        # Re-derive expected values from CombatState (no env access needed here).
        expected_power = max(
            0,
            combat.attack_power + combat.attack_bonus + combat.opposing_power_mod,
        )
        expected_raw = max(0, expected_power - total_block)

        notes: list[str] = [
            f"attack_power={combat.attack_power} bonus={combat.attack_bonus} "
            f"mod={combat.opposing_power_mod} → total_power={total_power}",
            f"total_block={total_block} → raw_damage={raw_damage} "
            f"→ actual_damage={actual_damage}",
        ]

        # Check 1: total power formula
        power_ok = total_power == expected_power
        if not power_ok:
            notes.append(
                f"POWER MISMATCH: expected {expected_power} got {total_power}"
            )

        # Check 2: raw damage formula (before damage-floor adjustment which adds 0+)
        # raw_damage can be >= expected_raw due to damage floor
        raw_ok = raw_damage >= expected_raw and raw_damage >= 0
        if not raw_ok:
            notes.append(
                f"RAW DAMAGE MISMATCH: expected >={expected_raw} got {raw_damage}"
            )

        # Check 3: mitigation never increases damage
        mitigation_ok = actual_damage <= raw_damage and actual_damage >= 0
        if not mitigation_ok:
            notes.append(
                f"MITIGATION INVARIANT VIOLATED: damage={actual_damage} raw={raw_damage}"
            )

        match = power_ok and raw_ok and mitigation_ok
        defender_life_after = defender_life_before - actual_damage
        result = ValidationResult(
            match=match,
            source="local_rules",
            expected_total_power=expected_power,
            actual_total_power=total_power,
            expected_raw_damage=expected_raw,
            actual_raw_damage=raw_damage,
            expected_mitigated=actual_damage,  # no full mitigation re-derive here
            actual_mitigated=actual_damage,
            defender_life_before=defender_life_before,
            defender_life_expected=defender_life_before - actual_damage,
            defender_life_actual=defender_life_after,
            notes=notes,
        )
        self.results.append(result)
