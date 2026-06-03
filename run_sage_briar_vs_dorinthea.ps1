#!/usr/bin/env pwsh
# Train -> Evaluate -> Render pipeline for SAGE precon: Briar vs Dorinthea
# Open results\sage_precon_agents\briar-vs-dorinthea\eval_live_state.png in JPEGView before running
# to watch the evaluation board state update in real time.

python main.py train-eval-render `
    --trainer sage-precons `
    --matchup briar-vs-dorinthea `
    --format sage `
    --episodes 300 `
    --max-steps 500 `
    --eval-episodes 20 `
    --eval-max-steps 500 `
    --render-max-steps 200 `
    --show-frontend-eval `
    --workers 4 `
    --out-dir results/sage_precon_agents `
    --cache-dir results/agent_cache
