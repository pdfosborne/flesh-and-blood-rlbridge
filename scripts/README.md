# Scripts

Utility and pipeline scripts are grouped by purpose:

| Directory | Contents |
|-----------|----------|
| `training/` | RL training pipelines, shared PPO loop, agent/episode caches |
| `eval/` | Checkpoint evaluation and head-to-head agent comparison |
| `cpp/` | C++ engine generation, build, and Talishar parity checking |
| `deck/` | FaBrary deck fetch and conversion |

Cross-directory imports rely on `scripts/_bootstrap.py`, which adds each
subdirectory to `sys.path`. Entry-point scripts call `configure_paths()` at
startup; tests pick up the same paths via `pyproject.toml`.

High-level orchestration lives in `runscripts/` at the repo root.
