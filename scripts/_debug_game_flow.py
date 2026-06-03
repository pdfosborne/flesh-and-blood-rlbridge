"""Debug script: mirror the real training loop — run sample_action() for N steps
and log every decision so we can see exactly where turns stall.

Usage:
    python scripts/_debug_game_flow.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment
from flesh_and_blood_rlbridge.talishar_default_policy import (
    choose_talishar_action_index, _get_phase, _is_pass_action,
)

MAX_STEPS = 300

env = TalisharEngineEnvironment(
    base_url="http://localhost:8080/game",
    local_deck_name="BriarSAGEPrecon",
    opponent_deck_name="DorintheaSAGEPrecon",
    game_format="sage",
    self_play=True,
    max_turns=500,
)
env.reset()
print("=== game started ===\n")

for step_no in range(1, MAX_STEPS + 1):
    state = env._last_state
    legal = env._extract_legal_actions(state)
    phase = _get_phase(state)

    prompt = state.get("playerPrompt", {})
    btns   = [f"{b.get('mode')}:{b.get('caption','?')}" for b in prompt.get("buttons", [])]
    popup  = state.get("playerInputPopUp", {})
    pbtn   = [f"{b.get('mode')}:{b.get('caption','?')}" for b in popup.get("buttons", [])] if popup.get("active") else []

    idx = choose_talishar_action_index(legal, state)

    p_deck = state.get("playerDeckCount", "?")
    o_deck = state.get("opponentDeckCount", "?")
    print(
        f"step={step_no:3d} turn={state.get('turnNo',0):2} phase={phase or '?':8} "
        f"acting=P{env._acting_player_id} hp={state.get('playerHealth',20)}/{state.get('opponentHealth',20)} "
        f"deck={p_deck}/{o_deck}"
    )
    print(f"  prompt_btns : {btns}")
    if pbtn:
        print(f"  popup_btns  : {pbtn}")
    print(f"  legal ({len(legal)}):")
    for i, a in enumerate(legal):
        marker = ">>> " if i == idx else "    "
        is_p = _is_pass_action(a)
        print(f"  {marker}[{i}] code={a['action_code']:6} zone={a['zone']:10} label={a['label']}  {'(PASS)' if is_p else ''}")
    print()

    action_str = env.sample_action()
    result = env.step(action_str)
    # Dump full raw state on unknown phases OR on pitch phase (p) to verify pitching
    new_phase = _get_phase(env._last_state)
    dump_phase = new_phase not in (
        # Standard combat phases
        "m", "d", "instant", "a", "p", "ars", "b", "chain",
        # Mandatory pick (CanPassPhase=0)
        "choosemultizone", "choosehand", "choosehandcancel",
        "buttoninput", "buttoninputnopass",
        "choosedeck", "choosetheirdeck", "choosediscard", "choosediscardcancel",
        "choosearsenal", "choosearsenalcancel", "choosebanish",
        "choosecombatchain", "choosecharacter", "choosemysoul", "choosemyaura",
        "choosepermanent", "choosecardid", "multichoosehand", "multichoosediscard",
        # Optional pick (CanPassPhase=1)
        "maychoosemultizone", "maychoosehand", "maychoosediscard", "maychoosedeck",
        "maychoosearsenal", "maychoosecombatchain", "maymultichoosehand",
        "maymultichoosetext", "maychoosetheirdiscard", "maychoosemysoul",
        "maychoosepermanent",
        # End-of-turn / triggers / misc
        "ordertriggers", "startturn", "endphase", "pdeck", "ok", "yesno",
        "coercive", "choosearcane", "dynpitch", "choosenumber",
        "choosetop", "choosebottom", "choosetopopponent",
        # Terminal
        "over", "",
    )
    if dump_phase or (new_phase == "p" and step_no <= 60):
        import json as _json
        raw = env._last_state
        tag = f"UNKNOWN '{new_phase}'" if dump_phase else f"PITCH PHASE '{new_phase}'"
        print(f"  !! {tag} — raw state keys: {list(raw.keys())[:10]}...")
        for k in ("playerInputPopUp", "playerPrompt", "playerHand", "playerPitch"):
            if k in raw:
                val = _json.dumps(raw[k])
                print(f"    {k}: {val[:400]}")
    terminated = result.get("terminated", False) if isinstance(result, dict) else getattr(result, "terminated", False)
    truncated  = result.get("truncated",  False) if isinstance(result, dict) else getattr(result, "truncated",  False)
    if terminated or truncated:
        print(f"=== DONE at step {step_no} (terminated={terminated} truncated={truncated}) ===")
        break
    time.sleep(0.05)

print("\n=== done ===")
env.close()
