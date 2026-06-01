# Unimplemented Card Effects

Generated: 2026-06-01T12:25:31.704360+00:00

This report lists Flesh and Blood card interactions **not faithfully modeled**
by `flesh_and_blood_rlbridge` as of the scan date. Cards with at least one
implemented effect on the same card are marked **partial**.

## Summary

| Metric | Count |
|--------|------:|
| Unique cards (name+pitch) | 6904 |
| Cards with interaction text | 6010 |
| Fully implemented | 6002 |
| Partial (some gaps) | 0 |
| Parsed but nothing works | 2 |
| Nothing parsed | 6 |
| Distinct gap categories | 3 |

## Supported effect kinds (implemented)

`additional_pay`, `amp`, `arcane_barrier`, `arcane_damage`, `attack`, `banish_arsenal`, `banish_combo`, `banish_defending`, `banish_graveyard`, `banish_gy_variable`, `banish_hand_play`, `banish_self_play`, `banish_top`, `block_arcane_prevention`, `block_arsenal_play`, `block_defense_reactions`, `block_gold_gain`, `block_hit_effects`, `block_opponent_hit_effects`, `block_pitch_color`, `block_power_gain`, `block_weapon_attacks`, `blood_debt`, `boost`, `chain_defend`, `choose_card`, `choose_mode`, `clash`, `contract`, `cost_per_token`, `cost_reduction`, `counts_as_gold`, `create_banished`, `create_extra_token`, `create_in_hand`, `create_random_token`, `create_token`, `create_token_per_defender`, `create_token_triple`, `crowd_boo`, `crowd_cheer`, `dagger_damage`, `damage`, `damage_floor`, `damage_redirect`, `destroy_arsenal`, `destroy_hand`, `destroy_item`, `destroy_top`, `discard`, `dominate`, `double_damage`, `draw`, `enable_banish_play`, `enable_gy_play`, `equip_inventory`, `extra_attack_targets`, `extra_bow_activations`, `extra_turn`, `extra_weapon_attack`, `flip_face_up`, `for_each`, `freeze`, `fusion`, `gain_action_point`, `gain_gold`, `gain_life`, `gain_resources`, `galvanize`, `go_again`, `grade_increase`, `grant_draconic`, `grant_hit_bonus`, `grant_light_block`, `grant_may_play`, `gy_to_bottom`, `halve_base_power`, `heave`, `hit_bonus_damage`, `hit_rider`, `intellect_mod`, `intimidate`, `inventory_to_hand`, `invert_next_life_gain`, `limit_actions_next_turn`, `look_deck`, `look_face_down`, `lose_all_abilities`, `lose_class_talent`, `lose_colors`, `lose_game`, `lose_gold`, `lose_life`, `lose_life_per_dagger_hit`, `lose_life_per_hand_card`, `lose_phantasm`, `mark`, `modify_attack_power`, `modular_equip`, `name_card`, `named_power_bonus`, `next_ability_cost_reduction`, `next_action_go_again`, `next_attack_power`, `next_defense_bonus`, `next_naa_go_again`, `opponent_cost_increase`, `opt`, `overpower`, `pitch_bonus`, `pitch_deck_top`, `pitch_pay`, `pitch_restriction`, `play_as_instant`, `play_from_deck_top`, `play_power_cap`, `play_restriction`, `power`, `power_per_block`, `prevent_damage`, `prevention_reduction`, `put_arrow_arsenal`, `put_bottom`, `put_counter`, `put_deck_top_arsenal`, `put_hand_top`, `put_item_in_arena`, `put_soul`, `random_banished_pick`, `receive_token`, `reduce_defense`, `reduce_next_power_gain`, `reload`, `remove_blood_debt`, `remove_counter`, `remove_from_game`, `retrieve_banished_aura`, `retrieve_gy`, `return_arena_tapped`, `return_arsenal_hand`, `return_chain_cards`, `return_gy_to_deck`, `return_gy_to_hand`, `return_self_hand`, `reveal_for_blue_bonus`, `reveal_hand`, `reveal_named_hand`, `reveal_top`, `reveal_top_draconic_power`, `schedule_end_phase`, `scrap`, `search`, `set_next_instant_return_aura`, `set_next_instant_return_self`, `shuffle_put_top_arsenal`, `silence`, `stash_hand`, `steal_ally`, `steal_aura`, `steal_equipment`, `steal_token`, `taunt`, `transcend`, `transform_equip`, `transform_hero`, `transform_token`, `turn_arsenal_face_up`, `turn_banished_face`, `turn_equipment_face_up`, `unless_pay`, `upkeep_or_destroy`, `wager`, `ward`, `weapon_swing_cost_reduction`

## Unimplemented categories (by rules-text pattern)

Cards are counted in every category their text matches.

| Category | Cards | Examples |
|----------|------:|---------|
| other | 4 | Cracker Bauble, cracker-bauble-2---Cracker Bauble, Fractal Replication, fractal-replication-1---Fractal Replication |
| stealth | 2 | Horrors of the Past, horrors-of-the-past-2---Horrors of the Past |
| when_defends | 2 | Mask of Deceit, mask-of-deceit---Mask of Deceit |

## Parsed but unimplemented effect clauses

Text the parser recognizes as a trigger/ability but cannot resolve.

| Category | Occurrences | Effect text (sample) |
|----------|------------:|---------------------|
| stealth | 2 | it gets the base abilities of the last attack action card with stealth you control on the  |

## Keywords on cards without full mechanic support

| Keyword | Cards |
|---------|------:|

## Full per-card detail

See `unimplemented_effects.json` → `cards` for the complete list.
