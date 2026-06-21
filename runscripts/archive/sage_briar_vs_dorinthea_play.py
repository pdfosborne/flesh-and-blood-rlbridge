#!/usr/bin/env python3
"""Train, evaluate, and render the SAGE precon matchup Briar vs Dorinthea.

Open results/sage_precon_agents/briar-vs-dorinthea/eval_live_state.png in JPEGView
before running to watch the evaluation board state update in real time.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runscripts._common import REPO_ROOT, run_python


def main() -> int:
    return run_python(
        REPO_ROOT / "scripts" / "training" / "train_eval_render_pipeline.py",
        "--trainer",
        "sage-precons",
        "--matchup",
        "briar-vs-dorinthea",
        "--format",
        "sage",
        "--episodes",
        "300",
        "--max-steps",
        "500",
        "--eval-episodes",
        "20",
        "--eval-max-steps",
        "500",
        "--render-max-steps",
        "200",
        "--show-frontend-eval",
        "--workers",
        "2",
        "--out-dir",
        str(REPO_ROOT / "results" / "sage_precon_agents"),
        "--cache-dir",
        str(REPO_ROOT / "results" / "agent_cache"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
