# Unimplemented Card Effects

Generated: 2026-05-31T09:10:56.041215+00:00

This report lists Flesh and Blood card interactions **not faithfully modeled**
by `flesh_and_blood_rlbridge` as of the scan date. Cards with at least one
implemented effect on the same card are marked **partial**.

## Summary

| Metric | Count |
|--------|------:|
| Unique cards (name+pitch) | 6904 |
| Cards with interaction text | 6010 |
| Fully implemented | 5421 |
| Partial (some gaps) | 228 |
| Parsed but nothing works | 182 |
| Nothing parsed | 179 |
| Distinct gap categories | 35 |

## Supported effect kinds (implemented)

`additional_pay`, `amp`, `arcane_barrier`, `arcane_damage`, `attack`, `banish_arsenal`, `banish_combo`, `banish_defending`, `banish_graveyard`, `banish_top`, `block_arsenal_play`, `block_defense_reactions`, `block_hit_effects`, `block_power_gain`, `block_weapon_attacks`, `blood_debt`, `boost`, `choose_card`, `choose_mode`, `clash`, `contract`, `cost_per_token`, `cost_reduction`, `counts_as_gold`, `create_banished`, `create_in_hand`, `create_token`, `create_token_triple`, `crowd_boo`, `crowd_cheer`, `damage`, `damage_floor`, `destroy_arsenal`, `destroy_hand`, `destroy_item`, `destroy_top`, `discard`, `dominate`, `draw`, `enable_banish_play`, `enable_gy_play`, `extra_weapon_attack`, `for_each`, `fusion`, `gain_action_point`, `gain_gold`, `gain_life`, `gain_resources`, `galvanize`, `go_again`, `grant_hit_bonus`, `grant_light_block`, `grant_may_play`, `gy_to_bottom`, `heave`, `hit_bonus_damage`, `intimidate`, `look_deck`, `lose_all_abilities`, `lose_gold`, `lose_life`, `mark`, `modify_attack_power`, `modular_equip`, `named_power_bonus`, `next_ability_cost_reduction`, `next_action_go_again`, `next_attack_power`, `next_defense_bonus`, `next_naa_go_again`, `opponent_cost_increase`, `opt`, `overpower`, `pitch_pay`, `play_as_instant`, `play_from_deck_top`, `play_power_cap`, `power`, `prevent_damage`, `put_arrow_arsenal`, `put_bottom`, `put_counter`, `put_item_in_arena`, `put_soul`, `reduce_defense`, `reduce_next_power_gain`, `reload`, `remove_counter`, `return_gy_to_deck`, `return_gy_to_hand`, `reveal_hand`, `reveal_top`, `scrap`, `search`, `silence`, `stash_hand`, `steal_aura`, `steal_equipment`, `steal_token`, `transcend`, `transform_equip`, `unless_pay`, `upkeep_or_destroy`, `wager`, `ward`

## Unimplemented categories (by rules-text pattern)

Cards are counted in every category their text matches.

| Category | Cards | Examples |
|----------|------:|---------|
| other | 140 | Arc Light Sentinel, arc-light-sentinel-2---Arc Light Sentinel, Attune with Cosmic Vibrations, attune-with-cosmic-vibrations-3---Attune with Cosmic Vibrations, Blanch |
| pitch_pay | 137 | Barbed Castaway, Barbed Undertow, barbed-castaway---Barbed Castaway, barbed-undertow-1---Barbed Undertow, Barkbone Strapping |
| activated_ability | 136 | Amethyst Tiara, amethyst-tiara---Amethyst Tiara, Barbed Castaway, barbed-castaway---Barbed Castaway, Barkbone Strapping |
| go_again | 128 | Be Like Water, be-like-water-1---Be Like Water, Billowing Mirage, billowing-mirage-1---Billowing Mirage, Bingo |
| destroy | 124 | Amethyst Tiara, amethyst-tiara---Amethyst Tiara, Barkbone Strapping, barkbone-strapping---Barkbone Strapping, Blood Runs Deep |
| whenever | 44 | Cindra, Dracai of Retribution, cindra-dracai-of-retribution---Cindra, Dracai of Retribution, Evo Atom Breaker, Evo Circuit Breaker, Evo Face Breaker |
| banish | 36 | Bonds of Agony, bonds-of-agony-3---Bonds of Agony, Frankie, Make Ends Meat, frankie-make-ends-meat---Frankie, Make Ends Meat, Gore Belching |
| fusion | 36 | Awakening, awakening-3---Awakening, Brain Freeze, brain-freeze-1---Brain Freeze, Encase |
| draw | 32 | Bingo, bingo-1---Bingo, Cryptic Crossing, cryptic-crossing-2---Cryptic Crossing, Genis Wotchuneed |
| create_token | 30 | Benefactor of Bloodworth Goldmane, benefactor-of-bloodworth-goldmane---Benefactor of Bloodworth Goldmane, Cindra, Dracai of Retribution, cindra-dracai-of-retribution---Cindra, Dracai of Retribution, Evo Mach Breaker |
| equipment_mod | 30 | Demolition Protocol, demolition-protocol-1---Demolition Protocol, Encase, encase-1---Encase, Enigma, New Moon |
| deal_damage | 18 | Deny Redemption, deny-redemption-1---Deny Redemption, Encase, encase-1---Encase, Flicker Wisp |
| stealth | 18 | Bonds of Agony, bonds-of-agony-3---Bonds of Agony, Horrors of the Past, horrors-of-the-past-2---Horrors of the Past, Kiss of Death |
| discard | 16 | Cryptic Crossing, cryptic-crossing-2---Cryptic Crossing, Dead Eye, dead-eye-2---Dead Eye, Deny Redemption |
| for_each | 16 | Blood Runs Deep, blood-runs-deep-1---Blood Runs Deep, Cindra, Dracai of Retribution, cindra-dracai-of-retribution---Cindra, Dracai of Retribution, Moonshot |
| put_into_soul | 16 | Prism, Advent of Thrones, Prism, Awakener of Sol, prism-advent-of-thrones---Prism, Advent of Thrones, prism-awakener-of-sol---Prism, Awakener of Sol, Ray of Hope |
| boost | 14 | Evo Atom Breaker, Evo Circuit Breaker, Evo Face Breaker, Evo Mach Breaker, evo-atom-breaker-1---Evo Atom Breaker |
| may_play | 12 | Iyslander, Stormbind, iyslander-stormbind---Iyslander, Stormbind, Nuu, Alluring Desire, nuu-alluring-desire---Nuu, Alluring Desire, Proclamation of Combat |
| combo | 10 | Dishonor, dishonor-3---Dishonor, Hurricane Technique, hurricane-technique-2---Hurricane Technique, Lord of Wind |
| put_counter | 10 | Barbed Castaway, barbed-castaway---Barbed Castaway, Hala, Bladesaint of the Vow, hala-bladesaint-of-the-vow---Hala, Bladesaint of the Vow, invoke-nekria-1--nekria---Nekria |
| search | 10 | Awakening, awakening-3---Awakening, Bonds of Agony, bonds-of-agony-3---Bonds of Agony, Prism, Advent of Thrones |
| target_gets | 10 | Concealed Blade, concealed-blade-3---Concealed Blade, Lord of Wind, lord-of-wind-3---Lord of Wind, Sharpening Sparks |
| blood_debt | 8 | Guardian of the Shadowrealm, guardian-of-the-shadowrealm-1---Guardian of the Shadowrealm, Levia, Redeemed, Levia, Shadowborn Abomination, levia-redeemed--blasmophet-levia-consumed---Levia, Redeemed |
| dominate | 8 | Bravo, Flattering Showman, bravo-flattering-showman---Bravo, Flattering Showman, Nourishing Emptiness, nourishing-emptiness-1---Nourishing Emptiness, Oaken Old |
| surge | 8 | Awakening, awakening-3---Awakening, Tales of Adventure, tales-of-adventure-3---Tales of Adventure, Testament of Valahai |
| unless | 8 | Grinding Gears, grinding-gears-3---Grinding Gears, Mistcloak Gully, mistcloak-gully--inner-chi-3---Mistcloak Gully, Parched Terrain |
| when_defends | 8 | Alluring Inducement, alluring-inducement-2---Alluring Inducement, Mask of Deceit, mask-of-deceit---Mask of Deceit, Vambrace of Determination |
| crush | 7 | Bolfar, Bear Hands, bolfar-bear-hands---Bolfar, Bear Hands, Bravo, Flattering Showman, bravo-flattering-showman---Bravo, Flattering Showman, Renounce Grandeur |
| clash | 6 | Brutus, Summa Rudis, brutus-summa-rudis---Brutus, Summa Rudis, Miller's Grindstone, millers-grindstone---Miller's Grindstone, Victor Goldmane, Match Fixer |
| when_enters | 6 | Figment of Judgment, figment-of-judgment-2--themis-archangel-of-judgment---Figment of Judgment, Stasis Cell, stasis-cell-3---Stasis Cell, Truce |
| when_deals_damage | 3 | Renounce Grandeur, renounce-grandeur-1---Renounce Grandeur, walk-in-my-shoes-2---Walk in My Shoes |
| contract | 2 | Pay Day, pay-day-3---Pay Day |
| overpower | 2 | Moonshot, moonshot-2---Moonshot |
| transcend | 2 | Mistcloak Gully, mistcloak-gully--inner-chi-3---Mistcloak Gully |
| when_leaves | 2 | Turn Heads, turn-heads-3---Turn Heads |

## Parsed but unimplemented effect clauses

Text the parser recognizes as a trigger/ability but cannot resolve.

| Category | Occurrences | Effect text (sample) |
|----------|------------:|---------------------|
| other | 4 | choose red, yellow, or blue |
| other | 4 | Roll a 6 sided die |
| pitch_pay | 4 | you may pay {r} |
| other | 4 | transform up to 1 ash you control into an Aether Ashwing |
| other | 4 | cards they own lose all colors until the end of their next turn |
| other | 4 | the next time you play an instant card this chain link, you may return target aura permane |
| other | 4 | look at their hand and choose a card |
| other | 4 | put an action card with cost 2 or less from their hand on top of their deck |
| other | 4 | your next attack this combat chain is Draconic |
| other | 4 | name a card |
| other | 4 | a hero, they reveal the top card of their deck |
| other | 4 | you may attack an additional time with this weapon this turn |
| equipment_mod | 4 | you may look at a face-down card in their arsenal or equipment zones |
| other | 4 | the base {p} of the first attack action card they play during their next turn is halved, r |
| pitch_pay | 4 | or defends, you may pay {r}{r} |
| other | 4 | until end of turn if an attack would deal damage, instead it deals that much damage plus 1 |
| other | 4 | you may look at the defending hero's hand |
| pitch_pay | 4 | their first attack during their next turn costs an additional {r} to play or activate |
| other | 4 | target dagger you control deals 1 damage to target hero |
| other | 4 | and deals damage to a hero, freeze a card in their arsenal |
| other | 4 | Name a card |
| other | 4 | Target attack action card you control has 6 base {p} |
| other | 4 | you may turn a card in their banished zone face-down |
| other | 4 | your hero gets +1 until end of turn |
| other | 4 | you may put a card from your hand face-down into your arsenal |
| other | 4 | you may retrieve a dagger from your graveyard |
| other | 4 | you may turn a card in their graveyard face-down |
| other | 4 | Awaken target figment you control |
| other | 4 | each hero who doesn't have a card in their arsenal puts the top card of their deck face-do |
| pitch_pay | 4 | you may pay {r}{r}{r} |
| other | 4 | you may activate abilities of bows you control an additional time this turn and as though  |
| equipment_mod | 4 | remove all steam counters from an equipment, item, or weapon they control |
| other | 4 | it gets +2{d} |
| other | 4 | you may put a Mechanologist item from your hand into the arena with cost less than or equa |
| other | 4 | a marked hero, equip a Graphene Chelicera token |
| other | 3 | each hero puts the top card of their deck face-down into their arsenal |
| pitch_pay | 3 | cards cost {r} more to play this turn |
| other | 2 | the defending hero reveals their hand |
| other | 2 | Runechants you control get Spellvoid 1 this turn |
| other | 2 | a hero or defends a hero's attack, reveal the top card of their deck |
| other | 2 | instead create twice that many |
| other | 2 | You may turn a face-down arrow in your arsenal face-up |
| other | 2 | it gets +X{p}, where X is the number of Gold you control |
| other | 2 | they reveal a card from their hand |
| other | 2 | a hero, each dagger you control deals 1 damage to them |
| crush | 2 | The next attack action card with crush you play this turn may attack an additional hero |
| other | 2 | Turn a face-down card in your arsenal face-up |
| other | 2 | Equip an off-hand with Proclamation in its name from your inventory |
| other | 2 | You may add an action card from your arsenal to the active chain link as a defending card |
| other | 2 | Equip up to 2 Draconic daggers from your graveyard |
| other | 2 | you may deal that much damage to another ally controlled by the same hero |
| other | 2 | {u} all cogs you control |
| other | 2 | equip a dagger from your inventory |
| other | 2 | Attack" |
| pitch_pay | 2 | Your sword attacks cost {r} less to activate this turn |
| other | 2 | Target dagger you control that isn't on the active chain link deals 1 damage to the defend |
| equipment_mod | 2 | a hero, remove all steam counters from up to X equipment, items, and/or weapons they contr |
| other | 2 | Heroes can't gain {g} this turn |
| other | 2 | if you control Surging Strike, Descendent Gustwave, and Bonds of Ancestry, that hero loses |
| other | 2 | create Runechant tokens equal to the number of non-attack action cards you've played this  |
| other | 2 | The next Illusionist attack action card you play this turn loses and can't gain phantasm |
| equipment_mod | 2 | and deals damage to a hero, freeze them and all equipment they control |
| other | 2 | put it into its owner's hand |
| equipment_mod | 2 | Turn target face-down equipment you have equipped face-up |
| other | 2 | cards they own lose all class and talent types until the end of their next turn |
| other | 2 | if you control a Toughness token, create 3 more |
| other | 2 | if you control a Might token, create 3 more |
| pitch_pay | 2 | Your next weapon attack this turn costs {r} less to activate |
| other | 2 | You may attack an additional time with target weapon this turn |
| other | 2 | put this on the bottom of its owner's deck |
| other | 2 | Look at the top 3 cards of the event deck, then put them back in any order |
| pitch_pay | 2 | it gets +X{p}, where X is twice the number of cards in all pitch zones |
| other | 2 | Target dagger you control that isn't on the active chain link deals 1 damage to target her |
| other | 2 | until end of turn, action card effects you control that deal arcane damage, instead deal t |
| equipment_mod | 2 | Equip an equipment from a graveyard |
| other | 2 | instead deal X arcane damage, where X is 5 plus the number of Frostbites, Ice afflictions, |
| pitch_pay | 2 | until the end of their next turn, they can't pitch or play cards with base cost 0 |
| other | 2 | Until end of turn, effects controlled by opponents don't trigger when their attacks hit |
| other | 2 | Each other hero may put a card from their hand on the bottom of their deck |
| other | 2 | Choose a hero |
| … | … | *90 more in JSON* |

## Keywords on cards without full mechanic support

| Keyword | Cards |
|---------|------:|

## Full per-card detail

See `unimplemented_effects.json` → `cards` for the complete list.
