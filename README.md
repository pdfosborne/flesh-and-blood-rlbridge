# Flesh and Blood rlbridge Environments

Talishar-inspired [Flesh and Blood](https://fabtcg.com/) TCG simulation environments for [rlbridge](https://github.com/your-org/rlbridge).

This package is **not** bundled with rlbridge. Install it separately when you want FaB training, evaluation, or MCP tooling.

## Environments

| Env ID | Description |
|--------|-------------|
| `FleshAndBlood-Talishar-v0` | Self-play: one policy controls both players (live Talishar server) |
| `FleshAndBlood-Talishar-SelfPlay-v0` | Alias for `FleshAndBlood-Talishar-v0` |
| `FleshAndBlood-Talishar-VsAI-v0` | Agent vs CombatDummy AI (live Talishar server) |
| `FleshAndBlood-SelfPlay-v0` | Single policy controls both heroes (local simulator) |
| `FleshAndBlood-DeckBuild-v0` | Two-phase deck selection before play |

## Install

Install rlbridge first, then this package from GitHub:

```bash
pip install rlbridge
pip install git+https://github.com/pdfosborne/flesh-and-blood-rlbridge.git
```

For local development from a checkout:

```bash
cd /path/to/rlbridge
pip install -e .

cd /path/to/flesh-and-blood
pip install -e .
```

RL Bridge discovers installed environments through the `rlbridge.environments` entry point and registers FaB-specific MCP tools through `rlbridge.environment_mcp_tools`.

## Usage with rlbridge

```python
from rlbridge.environments.registry import registry

env = registry.create("FleshAndBlood-Talishar-v0", format="silver_age")
result = env.reset(seed=0)
print(result.observation)
```

List registered FaB environments:

```python
registry.list_environments(namespace="flesh_and_blood")
```

## MCP tools

When installed, the rlbridge MCP plugin exposes FaB-specific tools:

- `fab_list_deck_options`
- `fab_estimate_win_probabilities`
- `fab_evaluate_deck_matchup`
- `fab_meta_reward_for_deck`
- `fab_resolve_deck_from_url`
- `fab_update_cards_db_from_fabtcg`
- `fab_list_dual_agent_training_programs` — SAGE precon / Silver Age / CC Talishar training scripts
- `fab_run_dual_agent_training` — run one of those scripts (matchup, episodes, cache, etc.)

## Card database

Card and hero metadata live in `src/flesh_and_blood_rlbridge/card_db/`. To refresh from upstream exports:

```bash
cd src/flesh_and_blood_rlbridge/card_db
python import_from_talishar.py --source /path/to/upstream_cards.json --out cards.json
```

## Repository

https://github.com/pdfosborne/flesh-and-blood-rlbridge
