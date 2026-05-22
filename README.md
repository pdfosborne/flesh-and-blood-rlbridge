# Flesh and Blood RLIP Environments

Talishar-inspired [Flesh and Blood](https://fabtcg.com/) TCG simulation environments for [RLIP](https://github.com/your-org/rlip).

This package is **not** bundled with RLIP. Install it separately when you want FaB training, evaluation, or MCP tooling.

## Environments

| Env ID | Description |
|--------|-------------|
| `FleshAndBlood-Talishar-v0` | Agent vs scripted opponent |
| `FleshAndBlood-SelfPlay-v0` | Single policy controls both heroes |
| `FleshAndBlood-DeckBuild-v0` | Two-phase deck selection before play |

## Install (local development)

Install RLIP first, then this package in editable mode from a sibling checkout:

```bash
cd /path/to/RL-IP
pip install -e .

cd /path/to/flesh-and-blood
pip install -e .
```

RLIP discovers installed environments through the `rlip.environments` entry point and registers FaB-specific MCP tools through `rlip.environment_mcp_tools`.

## Usage with RLIP

```python
from rlip.environments.registry import registry

env = registry.create("FleshAndBlood-Talishar-v0", format="silver_age")
result = env.reset(seed=0)
print(result.observation)
```

List registered FaB environments:

```python
registry.list_environments(namespace="flesh_and_blood")
```

## MCP tools

When installed, the RLIP MCP plugin exposes FaB-specific tools:

- `fab_list_deck_options`
- `fab_estimate_win_probabilities`
- `fab_evaluate_deck_matchup`
- `fab_meta_reward_for_deck`
- `fab_resolve_deck_from_url`

## Card database

Card and hero metadata live in `src/flesh_and_blood_rlip/card_db/`. To refresh from upstream exports:

```bash
cd src/flesh_and_blood_rlip/card_db
python import_from_talishar.py --source /path/to/upstream_cards.json --out cards.json
```

## Publishing to GitHub

```bash
git init
git add .
git commit -m "Initial Flesh and Blood RLIP environments package"
git remote add origin git@github.com:your-org/flesh-and-blood-rlip.git
git push -u origin main
```

After publishing, users can install with:

```bash
pip install git+https://github.com/your-org/flesh-and-blood-rlip.git
```
