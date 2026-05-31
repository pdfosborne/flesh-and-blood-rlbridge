# Unimplemented Card Effects

Generated: 2026-05-31T21:54:00.929085+00:00

This report lists Flesh and Blood card interactions **not faithfully modeled**
by `flesh_and_blood_rlbridge` as of the scan date. Cards with at least one
implemented effect on the same card are marked **partial**.

## Summary

| Metric | Count |
|--------|------:|
| Unique cards (name+pitch) | 6904 |
| Cards with interaction text | 6010 |
| Fully implemented | 5857 |
| Partial (some gaps) | 0 |
| Parsed but nothing works | 8 |
| Nothing parsed | 145 |
| Distinct gap categories | 21 |

## Supported effect kinds (implemented)

`additional_pay`, `amp`, `arcane_barrier`, `arcane_damage`, `attack`, `banish_arsenal`, `banish_combo`, `banish_defending`, `banish_graveyard`, `banish_gy_variable`, `banish_hand_play`, `banish_self_play`, `banish_top`, `block_arcane_prevention`, `block_arsenal_play`, `block_defense_reactions`, `block_gold_gain`, `block_hit_effects`, `block_opponent_hit_effects`, `block_pitch_color`, `block_power_gain`, `block_weapon_attacks`, `blood_debt`, `boost`, `chain_defend`, `choose_card`, `choose_mode`, `clash`, `contract`, `cost_per_token`, `cost_reduction`, `counts_as_gold`, `create_banished`, `create_in_hand`, `create_token`, `create_token_triple`, `crowd_boo`, `crowd_cheer`, `dagger_damage`, `damage`, `damage_floor`, `damage_redirect`, `destroy_arsenal`, `destroy_hand`, `destroy_item`, `destroy_top`, `discard`, `dominate`, `draw`, `enable_banish_play`, `enable_gy_play`, `equip_inventory`, `extra_attack_targets`, `extra_bow_activations`, `extra_turn`, `extra_weapon_attack`, `for_each`, `freeze`, `fusion`, `gain_action_point`, `gain_gold`, `gain_life`, `gain_resources`, `galvanize`, `go_again`, `grade_increase`, `grant_draconic`, `grant_hit_bonus`, `grant_light_block`, `grant_may_play`, `gy_to_bottom`, `halve_base_power`, `heave`, `hit_bonus_damage`, `hit_rider`, `intellect_mod`, `intimidate`, `inventory_to_hand`, `limit_actions_next_turn`, `look_deck`, `look_face_down`, `lose_all_abilities`, `lose_game`, `lose_gold`, `lose_life`, `lose_life_per_dagger_hit`, `lose_life_per_hand_card`, `lose_phantasm`, `mark`, `modify_attack_power`, `modular_equip`, `name_card`, `named_power_bonus`, `next_ability_cost_reduction`, `next_action_go_again`, `next_attack_power`, `next_defense_bonus`, `next_naa_go_again`, `opponent_cost_increase`, `opt`, `overpower`, `pitch_bonus`, `pitch_deck_top`, `pitch_pay`, `pitch_restriction`, `play_as_instant`, `play_from_deck_top`, `play_power_cap`, `play_restriction`, `power`, `power_per_block`, `prevent_damage`, `prevention_reduction`, `put_arrow_arsenal`, `put_bottom`, `put_counter`, `put_deck_top_arsenal`, `put_hand_top`, `put_item_in_arena`, `put_soul`, `random_banished_pick`, `reduce_defense`, `reduce_next_power_gain`, `reload`, `remove_counter`, `retrieve_gy`, `return_arena_tapped`, `return_arsenal_hand`, `return_gy_to_deck`, `return_gy_to_hand`, `return_self_hand`, `reveal_for_blue_bonus`, `reveal_hand`, `reveal_named_hand`, `reveal_top`, `reveal_top_draconic_power`, `schedule_end_phase`, `scrap`, `search`, `set_next_instant_return_aura`, `set_next_instant_return_self`, `silence`, `stash_hand`, `steal_ally`, `steal_aura`, `steal_equipment`, `steal_token`, `taunt`, `transcend`, `transform_equip`, `transform_hero`, `transform_token`, `turn_arsenal_face_up`, `turn_banished_face`, `turn_equipment_face_up`, `unless_pay`, `upkeep_or_destroy`, `wager`, `ward`, `weapon_swing_cost_reduction`

## Unimplemented categories (by rules-text pattern)

Cards are counted in every category their text matches.

| Category | Cards | Examples |
|----------|------:|---------|
| other | 84 | Arc Light Sentinel, arc-light-sentinel-2---Arc Light Sentinel, Blanch, blanch-1---Blanch, Code of Conduct: Kill or Be Killed |
| combo | 8 | Lord of Wind, lord-of-wind-3---Lord of Wind, Pounding Gale, pounding-gale-1---Pounding Gale, Recoil |
| create_token | 8 | Iyslander, Stormbind, iyslander-stormbind---Iyslander, Stormbind, Pilfer the Wreck, pilfer-the-wreck-1---Pilfer the Wreck, Squizzy & Floof |
| discard | 8 | Consuming Volition, consuming-volition-1---Consuming Volition, Reek of Corruption, reek-of-corruption-1---Reek of Corruption |
| whenever | 8 | invoke-nekria-1--nekria---Nekria, Iyslander, Stormbind, iyslander-stormbind---Iyslander, Stormbind, Nekria, Sigil of Parapets |
| equipment_mod | 6 | Lay Down the Law, lay-down-the-law-1---Lay Down the Law, Seasoned Saviour, seasoned-saviour---Seasoned Saviour, Visit the Golden Anvil |
| for_each | 6 | Plague Hive, plague-hive-2---Plague Hive, Show of Strength, show-of-strength-1---Show of Strength, The Moat Exchange |
| may_play | 6 | Iyslander, Stormbind, iyslander-stormbind---Iyslander, Stormbind, Spirit of Eirina, spirit-of-eirina-2---Spirit of Eirina, Warmonger's Diplomacy |
| put_into_soul | 6 | Ray of Hope, ray-of-hope-2---Ray of Hope, Spirit of Eirina, spirit-of-eirina-2---Spirit of Eirina, Tales of Adventure |
| surge | 6 | Tales of Adventure, tales-of-adventure-3---Tales of Adventure, Testament of Valahai, testament-of-valahai---Testament of Valahai, Tremor of Resistance |
| crush | 3 | Renounce Grandeur, renounce-grandeur-1---Renounce Grandeur, walk-in-my-shoes-2---Walk in My Shoes |
| when_deals_damage | 3 | Renounce Grandeur, renounce-grandeur-1---Renounce Grandeur, walk-in-my-shoes-2---Walk in My Shoes |
| blood_debt | 2 | Levia, Shadowborn Abomination, levia-shadowborn-abomination---Levia, Shadowborn Abomination |
| clash | 2 | Brutus, Summa Rudis, brutus-summa-rudis---Brutus, Summa Rudis |
| contract | 2 | Pay Day, pay-day-3---Pay Day |
| destroy | 2 | Visit the Golden Anvil, visit-the-golden-anvil-3---Visit the Golden Anvil |
| put_counter | 2 | invoke-nekria-1--nekria---Nekria, Nekria |
| stealth | 2 | Horrors of the Past, horrors-of-the-past-2---Horrors of the Past |
| target_gets | 2 | Lord of Wind, lord-of-wind-3---Lord of Wind |
| when_defends | 2 | Mask of Deceit, mask-of-deceit---Mask of Deceit |
| when_leaves | 2 | Turn Heads, turn-heads-3---Turn Heads |

## Parsed but unimplemented effect clauses

Text the parser recognizes as a trigger/ability but cannot resolve.

| Category | Occurrences | Effect text (sample) |
|----------|------------:|---------------------|
| other | 4 | cards they own lose all colors until the end of their next turn |
| other | 2 | cards they own lose all class and talent types until the end of their next turn |
| stealth | 2 | it gets the base abilities of the last attack action card with stealth you control on the  |

## Keywords on cards without full mechanic support

| Keyword | Cards |
|---------|------:|

## Full per-card detail

See `unimplemented_effects.json` → `cards` for the complete list.
