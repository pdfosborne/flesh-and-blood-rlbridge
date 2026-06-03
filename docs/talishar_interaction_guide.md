# How to Interact with Talishar: Step-by-Step Reference

This document is a complete reference for how the Python RL bridge interacts
with the Talishar game server.  It covers the full lifecycle of an episode —
from creating a game to parsing game state and submitting actions — and
explains the critical invariants discovered by reading the PHP source.

---

## 1. Prerequisites

A Talishar server must be running and reachable.

```powershell
# Windows (PowerShell) — start the server
cd Talishar
docker compose up -d
# Verify containers are healthy
docker compose ps
```

The Python code assumes `http://localhost` as the base URL.  Override with the
`TALISHAR_URL` environment variable or the `base_url` constructor parameter.

---

## 2. Episode Lifecycle (Three PHP Calls)

Every episode follows the same three-step HTTP sequence.

### Step 1 — Create the Game

**Endpoint:** `POST /APIs/CreateLocalGame.php` (preferred) or
`POST /APIs/CreateGame.php` (Fabrary deck link).

```json
// CreateLocalGame.php payload
{
  "deckName":         "Ira",
  "format":           "silver_age",
  "visibility":       "private",
  "opponentDeckName": "Ira",
  "selfPlay":         "1"
}
```

**Response fields used:**
| Field        | Meaning                              |
|--------------|--------------------------------------|
| `gameName`   | Unique game ID — needed for all calls|
| `authKey`    | P1 auth token                        |
| `p2AuthKey`  | P2 auth token (self-play only)       |

### Step 2 — Initialise the Gamestate File

**Endpoint:** `GET /Start.php?gameName=<gameName>&playerID=1`

Must be called once after `CreateGame`.  Writes the initial gamestate file
on the server.  Response: `{"success": true, "authKey": "..."}`.

### Step 3 — The Action Loop

Each loop iteration:
1. **Read state** — `GET /GetNextTurn.php`
2. **Choose action** — parse legal actions, apply policy
3. **Submit action** — `GET /ProcessInput.php`
4. Repeat until game over.

---

## 3. Reading Game State (`GetNextTurn.php`)

**Endpoint:** `GET /GetNextTurn.php`

**Required query parameters:**

| Parameter    | Value                                   |
|--------------|-----------------------------------------|
| `gameName`   | from CreateGame                         |
| `playerID`   | `1` or `2`                              |
| `authKey`    | matching auth key for that player       |
| `lastUpdate` | timestamp from previous response (0 initially) |

Pass `lastUpdate=0` to always get a full snapshot (slower but safe).  The
server sends a delta when `lastUpdate > 0`.

**Key fields in the response:**

| Field              | Type    | Meaning                                      |
|--------------------|---------|----------------------------------------------|
| `turnPhase.turnPhase` | string | Current phase code (e.g. `"M"`, `"P"`, `"B"`) |
| `turnPhase.caption`   | string | Human-readable phase description             |
| `havePriority`     | bool    | `true` when this player must act             |
| `playerHealth`     | int     | Acting player's HP                           |
| `opponentHealth`   | int     | Opponent's HP                                |
| `lastUpdate`       | int     | Timestamp — pass back on next call           |
| `playerHand`       | array   | Cards in hand (see Card Object below)        |
| `playerEquipment`  | array   | Equipped cards                               |
| `playerArse`       | array   | Arsenal zone                                 |
| `playerAuras`      | array   | Aura cards in play                           |
| `playerAllies`     | array   | Ally cards in play                           |
| `playerItems`      | array   | Item cards in play                           |
| `playerPermanents` | array   | Permanent cards in play                      |
| `playerDiscard`    | array   | Discard pile                                 |
| `playerBanish`     | array   | Banish zone                                  |
| `playerInputPopUp` | object  | Active popup (buttons/cards in popup zone)   |
| `promptButtons`    | array   | Buttons shown in the prompt bar              |

### Card Object (inside zone arrays)

```json
{
  "action":             27,          // Talishar mode code for this card
  "actionDataOverride": "3",         // Index — used as button_input
  "cardNumber":         "EVR001",    // Card set code / ID slug
  "label":              "Surging Strike",
  "power":              4,
  "defense":            3,
  "cost":               1
}
```

### Popup Object (`playerInputPopUp.popup.cards`)

Cards or buttons shown in a mandatory-choice popup.  Read `action` and
`actionDataOverride` the same way as hand cards.

---

## 4. Submitting an Action (`ProcessInput.php`)

**Endpoint:** `GET /ProcessInput.php`

**Required query parameters:**

| Parameter     | Value                                              |
|---------------|----------------------------------------------------|
| `gameName`    | from CreateGame                                    |
| `playerID`    | acting player (`1` or `2`)                         |
| `authKey`     | matching auth key                                  |
| `mode`        | Talishar action code (see table below)             |
| `buttonInput` | card index or button value (when applicable)       |
| `cardID`      | same value as `buttonInput` (satisfies card-zone modes) |

> **Always send `buttonInput` and `cardID` with the same value.**
> Card-zone modes (27, 3, 5, 10, 14 …) read from `$_GET["cardID"]`.
> Prompt/decision modes (17, 20, 99 …) read from `$_GET["buttonInput"]`.
> Sending both harmlessly satisfies all modes.

### Action Mode Codes

| Mode | Meaning                                         | Notes                                  |
|------|-------------------------------------------------|----------------------------------------|
| 99   | Pass / advance phase                            | Silent no-op for CanPassPhase=0 phases |
| 101  | Pass Block and Reactions                        | B phase only                           |
| 105  | Skip All Runechants                             | CHOOSEARCANE shortcut                  |
| 27   | Play card from hand                             | `buttonInput` = hand-card index        |
| 3    | Play/equip from equipment zone                  | `buttonInput` = equip-card index       |
| 4    | Add to arsenal                                  | ARS phase                              |
| 5    | Play from arsenal                               |                                        |
| 6    | PDECK ordering choice                           |                                        |
| 10   | Activate item                                   |                                        |
| 16   | Pick card from zone (popup/hand)                | CHOOSEHAND, CHOOSEMULTIZONE            |
| 17   | BUTTONINPUT popup button press                  | `buttonInput` = option index           |
| 20   | YESNO answer                                    | `buttonInput` = `"YES"` or `"NO"`      |
| 10000| Cancel / Undo last action                       | **Never use as a pass substitute**     |
| 10001| Undo Block                                      |                                        |

---

## 5. The `CanPassPhase` Invariant (CRITICAL)

**Source:** `Talishar/Libraries/NetworkingLibraries.php`, `ProcessInput` case 99:

```php
case 99:
    if (CanPassPhase($turn[0])) {
        PassInput($gameID, $playerID, $authKey, $turn, $GameState);
    }
    // else: silently does nothing — game state unchanged
```

**If you send `mode=99` during a `CanPassPhase=0` phase, the server ignores
it.  The game state does not change and the agent enters an infinite loop.**

### CanPassPhase=1 (Pass Works)

These phases safely accept `mode=99`:

| Phase        | Description                       |
|--------------|-----------------------------------|
| `M`          | Main phase — choose an action     |
| `B`          | Block phase                       |
| `D`          | Defense reaction window           |
| `INSTANT`    | Instant/reaction window           |
| `A`          | Attack reaction window            |
| `ARS`        | Arsenal phase                     |
| `CHAIN`      | Chain link window                 |
| `PDECK`      | End-of-turn pitch ordering        |
| `OK`         | Informational popup               |
| `COERCIVE`   | Rearrange opponent deck top       |
| `ORDERTRIGGERS`, `STARTTURN`, `ENDPHASE` | Trigger/effect ordering |
| `MAYCHOOSE*` family | Optional pick phases — pass declines |
| `YESNO`      | Yes/No prompt (buttons preferred) |

### CanPassPhase=0 (Pass is a Silent No-Op — MUST Pick)

Sending `mode=99` here does nothing.  Always send the correct action code:

| Phase              | Must-do action                                      |
|--------------------|-----------------------------------------------------|
| `P`                | Pitch a hand card (`mode=27`) or confirm with `mode=99` only after enough resources pitched |
| `CHOOSEMULTIZONE`  | Pick a card from popup (`mode=16`, `zone=popup`)    |
| `CHOOSEHAND`       | Pick a hand card (`mode=16`, `zone=hand`)           |
| `CHOOSEHANDCANCEL` | Pick a hand card (`mode=16`) or Cancel (`mode=10000`) |
| `BUTTONINPUT`      | Press a popup button (`mode=17`)                    |
| `BUTTONINPUTNOPASS`| Press a popup button (`mode=17`)                   |
| `CHOOSEDECK`       | Pick a card from popup                              |
| `CHOOSETHEIRDECK`  | Pick from opponent deck popup                       |
| `CHOOSEDISCARD`    | Pick from discard popup                             |
| `CHOOSEDISCARDCANCEL` | Pick or Cancel                                   |
| `CHOOSEARSENAL`    | Pick from arsenal popup                             |
| `CHOOSEBANISH`     | Pick from banish popup                              |
| `CHOOSECOMBATCHAIN`| Pick from combat chain popup                       |
| `CHOOSECHARACTER`  | Pick a character                                    |
| `CHOOSEPERMANENT`  | Pick a permanent                                    |
| `CHOOSECARDID`     | Pick a specific card ID                             |
| `CHOOSEMYSOUL`     | Pick from soul zone                                 |
| `CHOOSEMYAURA`     | Pick from aura zone                                 |
| `HANDTOPBOTTOM`    | Place hand cards to bottom of deck                  |
| `MULTICHOOSEHAND`  | Multi-select from hand                              |
| `MULTICHOOSEDISCARD` | Multi-select from discard                        |
| `MULTICHOOSEBANISH`  | Multi-select from banish                          |
| `MULTICHOOSEDECK`    | Multi-select from deck                            |
| `MULTICHOOSETHEIRDISCARD` | Multi-select opponent discard               |
| `MULTICHOOSETEXT`    | Multi-select text/option buttons                  |
| `CHOOSEFIRSTPLAYER`  | Choose who goes first                             |
| `OVER`               | Game is over — terminal state                     |

---

## 6. Phase-Based Action Selection Policy

The `choose_talishar_action_index` function in
`talishar_default_policy.py` routes to the right action using this
8-step decision tree:

```
1. CONFIRM phases (CanPassPhase=1, optional)
   → Send mode=99 (Pass)

2. PITCH phase (P, CanPassPhase=0)
   → Pitch cheapest hand card (mode=27, lowest power)
   → If no hand cards: send Cancel (mode=10000) to abort the play
     and blacklist that card as unaffordable for the rest of the turn

3. POPUP phases — choosemultizone (CanPassPhase=0)
   → Pick first card from playerInputPopUp.popup.cards (mode=16, zone=popup)

4. CHOOSEHAND phases (CanPassPhase=0)
   → Pick first action=16 card from playerHand (zone=hand)
   → CHOOSEHANDCANCEL fallback: send Cancel (mode=10000) if no hand cards

5. BUTTONINPUT / BUTTONINPUTNOPASS (CanPassPhase=0)
   → Press first mode=17 button in popup
   → Fallback: any non-pass popup/button action

6. BLOCK phase (B)
   → Play highest-defense action=27 hand card
   → If none: pass

7. DEFENSE phase (D)
   → Play highest-defense action=27 hand card with defense ≥ 3
   → If none: pass

8. MAIN phase (M) and unrecognised
   a. Yes-like popup buttons ("yes", "confirm", "ok", …)
   b. Best action=27 hand card by power (skip unaffordable blacklist)
   c. Other non-pass button/popup actions (catches remaining CanPassPhase=0
      popup-based phases: CHOOSEDECK, CHOOSEDISCARD, HANDTOPBOTTOM, etc.)
   d. Pass (mode=99)
   NOTE: Equip (mode=3) is intentionally skipped — the pitch window it opens
   may be empty, causing an equip→Cancel→equip infinite loop.
```

---

## 7. Known Pitfalls and Infinite Loop Patterns

### Pitfall 1 — Sending Pass During CanPassPhase=0

**Symptom:** Game state never changes; the same phase repeats forever.  
**Fix:** Check the phase code before sending `mode=99`.  All CanPassPhase=0
phases require a specific action code (see table in §5).

### Pitfall 2 — Equip Opens an Empty Pitch Window

**Symptom:** Agent equips an item, enters the `P` phase with no hand cards to
pitch, sends `mode=99` which does nothing, then gets stuck.  
**Fix:** Never send `mode=3` (equip) from the default policy.  If `P` arrives
with no pitchable cards, send `mode=10000` (Cancel) to abort the play.

### Pitfall 3 — Unaffordable Card Causes Pitch Loop

**Symptom:** Agent plays a card (e.g. `oasis_respite_red`), enters `P` phase,
pitches nothing (no resources), `mode=99` returns to main phase, agent plays
the same card again, cycle repeats.  
**Fix:** Track the last-played card label.  If `P` arrives with an empty hand,
Cancel and add the card label to an unaffordable blacklist for the rest of the
turn.

### Pitfall 4 — CHOOSEMULTIZONE Popup Not Parsed

**Symptom:** Popup cards not found because code was searching `playerHand`
instead of `playerInputPopUp.popup.cards`.  
**Fix:** Popup-zone cards live at `state["playerInputPopUp"]["popup"]["cards"]`,
not in any player zone array.

### Pitfall 5 — Cancel Counted as Pass

**Symptom:** `mode=10000` (Cancel) was classified as a "pass" action, so the
cycle-breaker didn't trigger and the loop continued indefinitely.  
**Fix:** Exclude `10000` and `10001` from `_PASS_MODE_CODES`.

---

## 8. Cycle-Breaker (Last Resort)

`TalisharEngineEnvironment.sample_action()` includes a hard cycle-breaker:

- It fingerprints the current legal-action set (sorted action codes + labels).
- If the **same fingerprint appears 4 consecutive times**, it selects a random
  legal action instead of the policy choice.
- This is a safety net only — if it fires regularly, there is a genuine loop
  that should be fixed in the policy.

---

## 9. Priority and Self-Play

In self-play mode (`self_play=True`), **both players share the same Python
process**.  After each action, the environment checks `havePriority` for both
player IDs to determine who acts next.

If the server returns `{"notYourTurn": true}`, the environment corrects the
`acting_player_id` to `currentPlayer` from the response and retries the
action once automatically.

---

## 10. PHP Source Reference

| File                                   | Key content                          |
|----------------------------------------|--------------------------------------|
| `Talishar/Libraries/NetworkingLibraries.php` | `ProcessInput` switch — all mode codes, case 99 guard |
| `Talishar/CoreLogic.php`               | `CanPassPhase()` function — complete list of CanPassPhase=0 phases |
| `Talishar/GameTerms.php`               | `TypeToPlay()` switch — all phase name strings |
| `Talishar/BuildGameState.php`          | Button generation logic per phase    |
| `Talishar/BuildPlayerInputPopup.php`   | Popup phase switch — BUTTONINPUT (17), CHOOSEHAND (16), YESNO (20) |
| `Talishar/GetNextTurn.php`             | Game state JSON field names          |
