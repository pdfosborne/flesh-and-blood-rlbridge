from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import pickle
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .environment_factory import TalisharEngineFactory
from .talishar_engine_environment import (
    run_talishar_eval_episode,
    talishar_deck_player_won,
)

_FAB_DB_DIR = Path(__file__).with_name("card_db")
_FABRARY_DECKS_PATH = _FAB_DB_DIR / "fabrary_decks.json"
_UPDATE_CARDS_SCRIPT_PATH = _FAB_DB_DIR / "update_cards_db_from_fabtcg.py"
_FAB_AGENT_CACHE_DIR = _FAB_DB_DIR / "agent_cache"
_FAB_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FAB_DUAL_AGENT_TRAINING_PROGRAMS: dict[str, dict[str, Any]] = {
    "sage_precons": {
        "script": _FAB_REPO_ROOT / "scripts" / "train_sage_precons.py",
        "description": (
            "Train PPO agents on all SAGE precon cross-matchups (Assets decks, blitz)."
        ),
        "default_format": "blitz",
        "default_max_steps": 60,
        "default_out_dir": "results/sage_precon_agents",
        "matchup_example": "briar-vs-dorinthea",
    },
    "silver_age": {
        "script": _FAB_REPO_ROOT / "scripts" / "train_silver_age_decks.py",
        "description": (
            "Train PPO agents on Silver Age fabrary deck cross-matchups "
            "(fabrary_decks.json, four-tier cache)."
        ),
        "default_format": "silver_age",
        "default_max_steps": 60,
        "default_out_dir": "results/silver_age_agents",
        "matchup_example": "ira_crimson_haze_sage_aggro-vs-fai_sage_chain",
    },
    "classic_constructed": {
        "script": _FAB_REPO_ROOT / "scripts" / "train_classic_constructed_decks.py",
        "description": (
            "Train PPO agents on Classic Constructed fabrary deck cross-matchups "
            "(fabrary_decks.json, four-tier cache)."
        ),
        "default_format": "classic_constructed",
        "default_max_steps": 120,
        "default_out_dir": "results/classic_constructed_agents",
        "matchup_example": "dorinthea_ironsong_cc_aggro-vs-chane_cc_shadow",
    },
}

_FAB_CUSTOM_TOOLS_REGISTERED = False
_FAB_REGISTERED_MATCHUP_ENVS: set[str] = set()
_FAB_CURRENT_AGENT_KEY: Optional[str] = None
_FAB_CURRENT_AGENT: Any = None


def _dual_agent_training_programs_payload() -> dict[str, Any]:
    programs: dict[str, Any] = {}
    for key, spec in _FAB_DUAL_AGENT_TRAINING_PROGRAMS.items():
        script_path = Path(spec["script"])
        programs[key] = {
            "description": spec["description"],
            "script_path": str(script_path),
            "script_exists": script_path.is_file(),
            "default_format": spec["default_format"],
            "default_max_steps": spec["default_max_steps"],
            "default_out_dir": spec["default_out_dir"],
            "matchup_example": spec["matchup_example"],
            "cli_options": {
                "matchup": "all or a slug (see matchup_example)",
                "episodes": "training episodes per matchup (default 300)",
                "max_steps": "max steps per episode (program-specific default)",
                "format": "Talishar game format (program-specific default)",
                "seed": "optional RNG seed",
                "out_dir": "where matchup agent packages are saved",
                "cache_dir": "four-tier weight cache root (fabrary programs only)",
            },
        }
    return {
        "programs": programs,
        "program_ids": list(_FAB_DUAL_AGENT_TRAINING_PROGRAMS),
        "notes": (
            "Requires a running Talishar server (set TALISHAR_URL or pass talishar_url). "
            "Training can take a long time when matchup=all."
        ),
    }


def _build_dual_agent_training_command(
    training_program: str,
    *,
    matchup: str,
    episodes: int,
    max_steps: Optional[int],
    game_format: Optional[str],
    seed: Optional[int],
    out_dir: Optional[str],
    cache_dir: Optional[str],
) -> tuple[list[str], dict[str, Any]]:
    spec = _FAB_DUAL_AGENT_TRAINING_PROGRAMS[training_program]
    script_path = Path(spec["script"])
    if not script_path.is_file():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    fmt = game_format or spec["default_format"]
    steps = int(max_steps) if max_steps is not None else int(spec["default_max_steps"])
    output = out_dir or str(_FAB_REPO_ROOT / spec["default_out_dir"])

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--matchup",
        matchup,
        "--episodes",
        str(int(episodes)),
        "--max-steps",
        str(steps),
        "--format",
        fmt,
        "--out-dir",
        output,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])

    meta = {
        "training_program": training_program,
        "script_path": str(script_path),
        "matchup": matchup,
        "episodes": int(episodes),
        "max_steps": steps,
        "format": fmt,
        "out_dir": output,
        "cache_dir": cache_dir,
        "seed": seed,
    }
    return cmd, meta


def register_mcp_tools(
    *, mcp: Any, registry: Any, log: Any, trained_agents: Optional[dict] = None
) -> int:
    """Register environment-specific MCP tools for Flesh and Blood.

    This function is discovered and called by the MCP plugin at startup.
    Returning an integer allows the plugin to report how many tools were added.

    Parameters
    ----------
    trained_agents:
        When provided (passed by the MCP plugin), any agent trained by
        ``fab_evaluate_deck_matchup`` will be stored here so that
        ``rl_render_policy`` can replay it later.
    """
    global _FAB_CUSTOM_TOOLS_REGISTERED
    if _FAB_CUSTOM_TOOLS_REGISTERED:
        return 0
    if registry is None:
        return 0

    def _build_agent(agent_type: str, hyperparams: dict[str, Any]) -> Any:
        if agent_type == "tabular_q":
            mod = importlib.import_module("rlbridge.rl_agents.tabular_q")
            return mod.TabularQAgent(
                alpha=float(hyperparams.get("alpha", 0.1)),
                gamma=float(hyperparams.get("gamma", 0.99)),
                epsilon=float(hyperparams.get("epsilon", 1.0)),
                epsilon_min=float(hyperparams.get("epsilon_min", 0.01)),
                epsilon_decay=float(hyperparams.get("epsilon_decay", 0.995)),
                seed=hyperparams.get("seed"),
            )
        if agent_type == "dqn":
            mod = importlib.import_module("rlbridge.rl_agents.dqn")
            return mod.DQNAgent(
                hidden_size=int(hyperparams.get("hidden_size", 64)),
                lr=float(hyperparams.get("lr", 1e-3)),
                gamma=float(hyperparams.get("gamma", 0.99)),
                epsilon=float(hyperparams.get("epsilon", 1.0)),
                epsilon_min=float(hyperparams.get("epsilon_min", 0.01)),
                epsilon_decay=float(hyperparams.get("epsilon_decay", 0.995)),
                buffer_size=int(hyperparams.get("buffer_size", 10000)),
                batch_size=int(hyperparams.get("batch_size", 64)),
                target_update_freq=int(hyperparams.get("target_update_freq", 100)),
                seed=hyperparams.get("seed"),
            )
        if agent_type == "ppo":
            mod = importlib.import_module("rlbridge.rl_agents.ppo")
            return mod.PPOAgent(
                hidden_size=int(hyperparams.get("hidden_size", 64)),
                lr_actor=float(hyperparams.get("lr_actor", 1e-3)),
                lr_critic=float(hyperparams.get("lr_critic", 1e-3)),
                gamma=float(hyperparams.get("gamma", 0.99)),
                lam=float(hyperparams.get("lam", 0.95)),
                clip_eps=float(hyperparams.get("clip_eps", 0.2)),
                n_steps=int(hyperparams.get("n_steps", 256)),
                ppo_epochs=int(hyperparams.get("ppo_epochs", 4)),
                mini_batch_size=int(hyperparams.get("mini_batch_size", 64)),
                seed=hyperparams.get("seed"),
            )
        raise ValueError(f"Unsupported agent type: {agent_type!r}")

    def _matchup_cache_key(
        *,
        format_name: str,
        inner_agent_type: str,
        deck_key: str,
        matchup_key: str,
        deck_name: str,
    ) -> str:
        return "|".join(
            [
                str(format_name),
                str(inner_agent_type),
                str(deck_key),
                str(matchup_key),
                str(deck_name),
            ]
        )

    def _run_eval_episode(
        env: Any,
        agent: Any,
        max_steps: int,
        seed: Optional[int],
        *,
        opponent_agent: Any | None = None,
    ) -> dict[str, Any]:
        return run_talishar_eval_episode(
            env,
            agent,
            max_steps,
            seed,
            p2_agent=opponent_agent,
            deck_player_id=1,
        )

    def _fab_win_probabilities(obs: Any) -> tuple[float, float]:
        if isinstance(obs, str):
            try:
                obs = json.loads(obs)
            except Exception:
                return 0.5, 0.5
        if not isinstance(obs, dict):
            return 0.5, 0.5

        player_hp = float(obs.get("playerHealth", 0.0))
        opp_hp = float(obs.get("opponentHealth", 0.0))

        if opp_hp <= 0 < player_hp:
            return 1.0, 0.0
        if player_hp <= 0 < opp_hp:
            return 0.0, 1.0

        player_hand = float(obs.get("playerHandSize", 0))
        opp_hand = float(obs.get("opponentHandSize", 0))
        player_deck = float(obs.get("playerDeckCount", 0))
        opp_deck = float(obs.get("opponentDeckCount", 0))
        player_pitch = float(obs.get("playerPitchCount", 0))

        player_score = (
            1.8 * player_hp
            + 1.0 * player_hand
            + 0.05 * player_deck
            + 0.3 * player_pitch
        )
        opp_score = (
            1.8 * opp_hp
            + 1.0 * opp_hand
            + 0.05 * opp_deck
        )

        if bool(obs.get("havePriority", False)):
            player_score += 0.4

        diff = (player_score - opp_score) / 8.0
        player_p = 1.0 / (1.0 + math.exp(-diff))
        player_p = max(0.0, min(1.0, player_p))
        return player_p, 1.0 - player_p

    def _fab_outcome_score(
        obs: Any,
        *,
        terminated: bool,
        deck_player_won: bool | None = None,
        deck_player_id: int = 1,
    ) -> float:
        if deck_player_won is True:
            return 1.0
        if deck_player_won is False:
            return 0.0
        if deck_player_won is None and terminated:
            return 0.5

        resolved = talishar_deck_player_won(
            obs,
            deck_player_id=deck_player_id,
            terminated=terminated,
        )
        if resolved is True:
            return 1.0
        if resolved is False:
            return 0.0
        if resolved is None and terminated:
            return 0.5

        p_agent, _ = _fab_win_probabilities(obs)
        if isinstance(obs, str):
            try:
                obs = json.loads(obs)
            except Exception:
                return float(p_agent)
        if isinstance(obs, dict) and int(obs.get("actingPlayerID", 1)) != deck_player_id:
            return 1.0 - float(p_agent)
        return float(p_agent)

    def _get_deck_options(format_name: str, seed: Optional[int]) -> list[dict[str, Any]]:
        assets_path = Path(
            os.environ.get(
                "TALISHAR_ASSETS_PATH",
                str(Path.home() / "Documents/flesh-and-blood/Talishar/Assets"),
            )
        )
        options: list[dict[str, Any]] = []
        if assets_path.is_dir():
            for deck_file in sorted(assets_path.glob("*.txt")):
                deck_name = deck_file.stem
                options.append({"key": deck_name, "label": deck_name, "deck_name": deck_name})
        if not options:
            options = [{"key": "Ira", "label": "Ira", "deck_name": "Ira"}]
        return options

    def _cache_paths(matchup_cache_key: str) -> tuple[Path, Path]:
        digest = hashlib.sha1(matchup_cache_key.encode("utf-8")).hexdigest()
        return (
            _FAB_AGENT_CACHE_DIR / f"{digest}.pkl",
            _FAB_AGENT_CACHE_DIR / f"{digest}.json",
        )

    def _load_cached_matchup_entry(matchup_cache_key: str) -> tuple[dict[str, Any], Any, bool]:
        global _FAB_CURRENT_AGENT_KEY, _FAB_CURRENT_AGENT

        agent_path, meta_path = _cache_paths(matchup_cache_key)
        metadata: dict[str, Any] = {}
        if meta_path.exists():
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    metadata = raw
            except Exception:
                metadata = {}

        if _FAB_CURRENT_AGENT_KEY != matchup_cache_key:
            _FAB_CURRENT_AGENT_KEY = None
            _FAB_CURRENT_AGENT = None

        if _FAB_CURRENT_AGENT_KEY == matchup_cache_key and _FAB_CURRENT_AGENT is not None:
            return metadata, _FAB_CURRENT_AGENT, True

        if not agent_path.exists():
            return metadata, None, False

        try:
            with agent_path.open("rb") as fh:
                agent = pickle.load(fh)  # noqa: S301
        except Exception:
            return metadata, None, False

        _FAB_CURRENT_AGENT_KEY = matchup_cache_key
        _FAB_CURRENT_AGENT = agent
        return metadata, agent, True

    def _save_cached_matchup_entry(matchup_cache_key: str, *, metadata: dict[str, Any], agent: Any) -> None:
        global _FAB_CURRENT_AGENT_KEY, _FAB_CURRENT_AGENT

        _FAB_AGENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        agent_path, meta_path = _cache_paths(matchup_cache_key)

        with agent_path.open("wb") as fh:
            pickle.dump(agent, fh)
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        _FAB_CURRENT_AGENT_KEY = matchup_cache_key
        _FAB_CURRENT_AGENT = agent

    def _list_cached_agent_entries() -> list[dict[str, Any]]:
        if not _FAB_AGENT_CACHE_DIR.exists():
            return []
        entries: list[dict[str, Any]] = []
        for meta_path in sorted(_FAB_AGENT_CACHE_DIR.glob("*.json")):
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            cache_key = str(raw.get("cache_key", ""))
            if not cache_key:
                continue
            agent_path, _ = _cache_paths(cache_key)
            item = dict(raw)
            item["agent_file_exists"] = agent_path.exists()
            item["metadata_file"] = str(meta_path)
            entries.append(item)
        return entries

    def _resolve_supported_formats(format_names_csv: Optional[str]) -> list[str]:
        if format_names_csv:
            return [f.strip() for f in str(format_names_csv).split(",") if f.strip()]
        return ["blitz", "classic_constructed"]

    def _evaluate_deck_vs_matchup(
        *,
        deck_option: dict[str, Any],
        matchup_option: dict[str, Any],
        deck_key: str,
        matchup_key: str,
        format_name: str,
        inner_agent_type: str,
        inner_train_episodes: int,
        inner_eval_episodes: int,
        inner_max_steps: int,
        seed: Optional[int],
    ) -> dict[str, Any]:
        p1_deck = str(deck_option.get("deck_name", "Ira"))
        p2_deck = str(matchup_option.get("deck_name", p1_deck))
        matchup_cache_key = _matchup_cache_key(
            format_name=format_name,
            inner_agent_type=inner_agent_type,
            deck_key=deck_key,
            matchup_key=matchup_key,
            deck_name=p1_deck,
        )
        cached_metadata, cached_agent, used_cached_agent = _load_cached_matchup_entry(matchup_cache_key)

        reverse_cache_key = _matchup_cache_key(
            format_name=format_name,
            inner_agent_type=inner_agent_type,
            deck_key=matchup_key,
            matchup_key=deck_key,
            deck_name=p2_deck,
        )
        _, opponent_agent, _ = _load_cached_matchup_entry(reverse_cache_key)

        env_kwargs: dict[str, Any] = {
            "render_mode": None,
            "local_deck_name": p1_deck,
            "opponent_deck_name": p2_deck,
            "game_format": str(format_name),
        }

        train_result: Any = None
        if cached_agent is not None and bool(cached_metadata.get("converged", False)):
            agent = cached_agent
            trained_now = False
        else:
            agent = cached_agent if cached_agent is not None else _build_agent(inner_agent_type, {})
            train_env = registry.create("FleshAndBlood-Talishar-v0", **env_kwargs)
            try:
                prior_eps = int(cached_metadata.get("total_train_episodes", 0))
                seed_offset = 0 if seed is None else int(seed)
                train_seed = seed_offset + prior_eps
                train_result = agent.train(
                    train_env,
                    n_episodes=inner_train_episodes,
                    max_steps=inner_max_steps,
                    seed=train_seed,
                )
            finally:
                train_env.close()
            trained_now = True

        eval_env = registry.create("FleshAndBlood-Talishar-v0", **env_kwargs)
        eval_scores: list[float] = []
        try:
            base_seed = 0 if seed is None else int(seed)
            for ep in range(inner_eval_episodes):
                ep_seed = base_seed + 10_000 + ep
                out = _run_eval_episode(
                    eval_env,
                    agent,
                    max_steps=inner_max_steps,
                    seed=ep_seed,
                    opponent_agent=opponent_agent,
                )
                eval_scores.append(
                    _fab_outcome_score(
                        out.get("final_observation"),
                        terminated=bool(out.get("terminated", False)),
                        deck_player_won=out.get("deck_player_won"),
                    )
                )
        finally:
            eval_env.close()

        win_rate = (sum(eval_scores) / len(eval_scores)) if eval_scores else 0.5
        previous_win_rate = float(cached_metadata.get("last_win_rate", 0.5))
        delta = abs(float(win_rate) - previous_win_rate)
        stable_rounds = int(cached_metadata.get("stable_rounds", 0))
        if delta < 0.01:
            stable_rounds += 1
        else:
            stable_rounds = 0
        total_train_episodes = int(cached_metadata.get("total_train_episodes", 0)) + (
            inner_train_episodes if trained_now else 0
        )
        converged = stable_rounds >= 3 and total_train_episodes >= max(inner_train_episodes * 3, 30)

        if trained_now and train_result is not None:
            train_mean_reward = float(train_result.mean_reward)
            train_best_reward = float(train_result.best_reward)
        else:
            train_mean_reward = float(cached_metadata.get("last_train_mean_reward", 0.0))
            train_best_reward = float(cached_metadata.get("last_train_best_reward", 0.0))

        cache_record = {
            "cache_key": matchup_cache_key,
            "format_name": format_name,
            "last_win_rate": float(win_rate),
            "stable_rounds": stable_rounds,
            "total_train_episodes": total_train_episodes,
            "converged": converged,
            "deck_key": deck_key,
            "matchup_key": matchup_key,
            "env_kwargs": dict(env_kwargs),
            "inner_agent_type": inner_agent_type,
            "last_train_mean_reward": train_mean_reward,
            "last_train_best_reward": train_best_reward,
        }
        _save_cached_matchup_entry(matchup_cache_key, metadata=cache_record, agent=agent)

        return {
            "win_rate": float(win_rate),
            "train_mean_reward": train_mean_reward,
            "train_best_reward": train_best_reward,
            "cached_agent_key": matchup_cache_key,
            "used_cached_agent": used_cached_agent,
            "trained_now": trained_now,
            "total_train_episodes": total_train_episodes,
            "converged": converged,
            "convergence_delta": float(delta),
            "stable_rounds": stable_rounds,
            "_agent": agent,
            "_train_result": train_result,
        }

    @mcp.tool()
    def fab_list_deck_options(
        format_name: str = "silver_age",
        seed: Optional[int] = None,
    ) -> str:
        """List all available hero/deck options for a Flesh and Blood format."""
        try:
            options = _get_deck_options(format_name, seed)
        except Exception as exc:
            log.exception("fab_list_deck_options error")
            return f"Error listing deck options: {exc}"

        result = {
            "format": format_name,
            "deck_options_count": len(options),
            "deck_options": options,
        }
        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_estimate_win_probabilities(observation_json: str) -> str:
        """Estimate win probabilities for both players from a FaB observation."""
        try:
            obs = json.loads(observation_json)
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON - {exc}"

        try:
            agent_p, opp_p = _fab_win_probabilities(obs)
        except Exception as exc:
            log.exception("fab_estimate_win_probabilities error")
            return f"Error computing win probabilities: {exc}"

        result = {
            "agent_win_probability": round(agent_p, 4),
            "opponent_win_probability": round(opp_p, 4),
            "inputs": {
                "agent_life": obs.get("playerHealth"),
                "opponent_life": obs.get("opponentHealth"),
                "agent_hand_size": obs.get("playerHandSize"),
                "opponent_hand_size": obs.get("opponentHandSize"),
                "have_priority": obs.get("havePriority"),
            },
            "reasoning": (
                "Logistic model over life totals (x1.8), hand size (x1.0), "
                "deck size (x0.05), pitch zone count (x0.3), "
                "and initiative bonus (+0.4 when agent has priority)."
            ),
        }
        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_evaluate_deck_matchup(
        deck_key: str,
        matchup_key: str,
        format_name: str = "silver_age",
        inner_agent_type: str = "ppo",
        inner_train_episodes: int = 50,
        inner_eval_episodes: int = 10,
        inner_max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> str:
        """Train and evaluate an inner gameplay agent for one FaB deck/matchup pair."""
        try:
            all_options = _get_deck_options(format_name, seed)
        except Exception as exc:
            return f"Error fetching deck options: {exc}"

        deck_option = next((o for o in all_options if str(o.get("key")) == deck_key), None)
        matchup_option = next((o for o in all_options if str(o.get("key")) == matchup_key), None)

        if deck_option is None:
            known = [str(o.get("key")) for o in all_options]
            return (
                f"Error: deck_key {deck_key!r} not found for format {format_name!r}.\n"
                f"Known keys: {known}"
            )
        if matchup_option is None:
            known = [str(o.get("key")) for o in all_options]
            return (
                f"Error: matchup_key {matchup_key!r} not found for format {format_name!r}.\n"
                f"Known keys: {known}"
            )

        try:
            stats = _evaluate_deck_vs_matchup(
                deck_option=deck_option,
                matchup_option=matchup_option,
                deck_key=deck_key,
                matchup_key=matchup_key,
                format_name=format_name,
                inner_agent_type=inner_agent_type,
                inner_train_episodes=inner_train_episodes,
                inner_eval_episodes=inner_eval_episodes,
                inner_max_steps=inner_max_steps,
                seed=seed,
            )
        except Exception as exc:
            log.exception("fab_evaluate_deck_matchup error")
            return f"Error evaluating deck matchup: {exc}"

        # Extract non-serialisable internal keys before building the result dict.
        _trained_agent = stats.pop("_agent", None)
        _train_result = stats.pop("_train_result", None)
        stats.pop("_env_kwargs", None)

        result: dict[str, Any] = {
            "deck_key": deck_key,
            "deck_label": deck_option.get("label", deck_key),
            "matchup_key": matchup_key,
            "matchup_label": matchup_option.get("label", matchup_key),
            "format": format_name,
            "inner_agent_type": inner_agent_type,
            "inner_train_episodes": inner_train_episodes,
            "inner_eval_episodes": inner_eval_episodes,
            **stats,
        }

        # If the plugin passed a trained_agents store, register the agent so
        # rl_render_policy can replay the policy directly.
        if trained_agents is not None and _trained_agent is not None:
            # Register a factory baked with this matchup's hero / format so
            # rl_render_policy can recreate the exact environment.
            _key_digest = hashlib.sha1(
                (
                    f"{format_name}|{inner_agent_type}|{deck_key}|{matchup_key}|"
                    f"{deck_option.get('hero_id', '')}|{matchup_option.get('hero_id', '')}"
                ).encode("utf-8")
            ).hexdigest()[:12]
            _agent_id = f"fabm_{_key_digest}"
            _registered_env_id = f"FleshAndBlood-matchup-{_key_digest}"
            matchup_factory = TalisharEngineFactory(
                _registered_env_id,
                game_format=format_name,
            )
            if _registered_env_id not in _FAB_REGISTERED_MATCHUP_ENVS:
                registry.register(matchup_factory)
                _FAB_REGISTERED_MATCHUP_ENVS.add(_registered_env_id)

            trained_agents[_agent_id] = {
                "agent": _trained_agent,
                "env_id": _registered_env_id,
                "agent_type": inner_agent_type,
                "best_episode_history": getattr(_train_result, "best_episode_history", []),
                "use_language_state": False,
                "train_result": _train_result,
                "training_config": {
                    "n_episodes": inner_train_episodes,
                    "max_steps": inner_max_steps,
                    "seed": seed,
                    "deck_key": deck_key,
                    "matchup_key": matchup_key,
                    "cached_agent_key": result.get("cached_agent_key"),
                    "converged": bool(result.get("converged", False)),
                },
            }

            result["agent_id"] = _agent_id
            result["registered_env_id"] = _registered_env_id

        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_meta_reward_for_deck(
        deck_key: str,
        format_name: str = "silver_age",
        inner_agent_type: str = "ppo",
        inner_train_episodes: int = 50,
        inner_eval_episodes: int = 10,
        matchups_per_deck: int = 3,
        inner_max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> str:
        """Compute the meta-reward for a FaB deck by sampling matchups."""
        try:
            all_options = _get_deck_options(format_name, seed)
        except Exception as exc:
            return f"Error fetching deck options: {exc}"

        deck_option = next((o for o in all_options if str(o.get("key")) == deck_key), None)
        if deck_option is None:
            known = [str(o.get("key")) for o in all_options]
            return (
                f"Error: deck_key {deck_key!r} not found for format {format_name!r}.\n"
                f"Known keys: {known}"
            )

        opponent_pool = [o for o in all_options if str(o.get("hero_id")) != str(deck_option.get("hero_id"))]
        if not opponent_pool:
            opponent_pool = [o for o in all_options if str(o.get("key")) != deck_key]
        if not opponent_pool:
            opponent_pool = list(all_options)

        rng = random.Random(seed)
        if matchups_per_deck >= len(opponent_pool):
            sampled = list(opponent_pool)
        else:
            sampled = rng.sample(opponent_pool, matchups_per_deck)

        matchup_results: list[dict[str, Any]] = []
        base_seed = 0 if seed is None else int(seed)

        for i, matchup in enumerate(sampled):
            ep_seed = base_seed + i * 1000
            try:
                stats = _evaluate_deck_vs_matchup(
                    deck_option=deck_option,
                    matchup_option=matchup,
                    deck_key=deck_key,
                    matchup_key=str(matchup.get("key", "")),
                    format_name=format_name,
                    inner_agent_type=inner_agent_type,
                    inner_train_episodes=inner_train_episodes,
                    inner_eval_episodes=inner_eval_episodes,
                    inner_max_steps=inner_max_steps,
                    seed=ep_seed,
                )
                stats.pop("_agent", None)
                stats.pop("_train_result", None)
                stats.pop("_env_kwargs", None)
                matchup_results.append(
                    {
                        "matchup_key": str(matchup.get("key", "")),
                        "matchup_label": str(matchup.get("label", matchup.get("key", "unknown"))),
                        "hero_id": matchup.get("hero_id"),
                        "win_rate": float(stats["win_rate"]),
                        "train_mean_reward": float(stats["train_mean_reward"]),
                        "error": None,
                    }
                )
            except Exception as exc:
                log.exception("fab_meta_reward_for_deck matchup error")
                matchup_results.append(
                    {
                        "matchup_key": str(matchup.get("key", "")),
                        "matchup_label": str(matchup.get("label", "")),
                        "hero_id": matchup.get("hero_id"),
                        "win_rate": 0.5,
                        "error": str(exc),
                    }
                )

        valid = [r for r in matchup_results if r.get("error") is None]
        meta_reward = (sum(r["win_rate"] for r in valid) / len(valid)) if valid else 0.5

        result = {
            "deck_key": deck_key,
            "deck_label": deck_option.get("label", deck_key),
            "format": format_name,
            "inner_agent_type": inner_agent_type,
            "inner_train_episodes": inner_train_episodes,
            "inner_eval_episodes": inner_eval_episodes,
            "matchups_per_deck": matchups_per_deck,
            "meta_reward": round(meta_reward, 4),
            "matchups_evaluated": len(matchup_results),
            "matchup_results": matchup_results,
        }
        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_resolve_deck_from_url(
        fabrary_url: str,
        side: str = "agent",
        format_name: str = "silver_age",
    ) -> str:
        """Resolve a Flesh and Blood deck from a fabrary.net public link.

        Parses the fabrary.net deck URL to extract the deck ID, looks up the deck
        in the static database, and resolves it to a card ID list legal for the
        specified format. The resolved deck can be used directly with environment
        setup or the fab_evaluate_deck_matchup tool.

        Args:
            fabrary_url: Full fabrary.net deck URL, e.g.
                "https://fabrary.net/decks/01KR40W4Z2ZS9EQPT6VT6CDSPE".
            side: Descriptive context - "agent" or "opponent". Does not affect
                resolution but is included in response for clarity.
            format_name: FaB format (silver_age, classic_constructed, cc, sa, blitz).
                Defaults to silver_age. The deck is verified against this format's
                legality.

        Returns:
            JSON string with deck_id, deck_name, hero_id, format, style,
            _card_ids (pre-resolved legal card list), source_url, and error
            info if resolution failed.
        """
        # Normalize format name
        _FORMAT_ALIASES: dict[str, str] = {
            "sa": "silver_age",
            "cc": "classic_constructed",
            "ll": "living_legend",
        }
        normalized_format = _FORMAT_ALIASES.get(format_name.lower(), format_name.lower())

        # Extract deck ID from URL: https://fabrary.net/decks/01KR40W4Z2ZS9EQPT6VT6CDSPE
        match = re.search(r"/decks/([a-zA-Z0-9]+)\b", fabrary_url)
        if not match:
            return json.dumps(
                {
                    "error": f"Could not parse deck ID from URL: {fabrary_url!r}",
                    "expected_format": "https://fabrary.net/decks/{{DECK_ID}}",
                    "side": side,
                    "format": normalized_format,
                },
                indent=2,
            )

        deck_id = match.group(1)
        deck_key = f"fab_{deck_id.lower()}"

        # Load fabrary database and find the deck
        try:
            if not _FABRARY_DECKS_PATH.exists():
                return json.dumps(
                    {
                        "error": f"Fabrary deck database not found at {_FABRARY_DECKS_PATH}",
                        "side": side,
                    },
                    indent=2,
                )

            data = json.loads(_FABRARY_DECKS_PATH.read_text(encoding="utf-8"))
            raw_decks = list(data.get("decks", []))
            deck_entry = next((d for d in raw_decks if str(d.get("id", "")).lower() == deck_key), None)

            if not deck_entry:
                known_ids = [d.get("id", "") for d in raw_decks]
                return json.dumps(
                    {
                        "error": f"Deck {deck_key!r} not found in fabrary database",
                        "deck_id_from_url": deck_id,
                        "available_decks": known_ids[:10],
                        "side": side,
                    },
                    indent=2,
                )

            # Check format match
            deck_format = str(deck_entry.get("format", ""))
            if deck_format != normalized_format:
                return json.dumps(
                    {
                        "error": f"Deck is {deck_format!r} but requested format is {normalized_format!r}",
                        "deck_id": deck_key,
                        "deck_name": deck_entry.get("name", deck_key),
                        "side": side,
                    },
                    indent=2,
                )

            # Resolve deck to card IDs by name lookup in the card database
            _cards_db_path = _FAB_DB_DIR / "cards.json"
            try:
                _all_cards = json.loads(_cards_db_path.read_text(encoding="utf-8"))
                _name_to_id = {
                    c["name"].lower(): c["id"]
                    for c in _all_cards
                    if "name" in c and "id" in c
                }
            except Exception:
                _name_to_id = {}
            card_ids: list[str] = []
            for _card_entry in deck_entry.get("cards", []):
                _cname = str(_card_entry.get("name", "")).lower()
                _count = int(_card_entry.get("count", 1))
                _cid = _name_to_id.get(_cname)
                if _cid:
                    card_ids.extend([_cid] * _count)

            if not card_ids:
                return json.dumps(
                    {
                        "error": f"Deck {deck_key!r} resolved to 0 legal cards for format {normalized_format!r}",
                        "deck_id": deck_key,
                        "deck_name": deck_entry.get("name", deck_key),
                        "side": side,
                    },
                    indent=2,
                )

            # Return as a deck option dict
            return json.dumps(
                {
                    "deck_id": deck_key,
                    "deck_name": deck_entry.get("name", deck_key),
                    "hero_id": str(deck_entry.get("hero_id", "")),
                    "format": deck_format,
                    "style": str(deck_entry.get("style", "balanced")),
                    "deck_size": len(card_ids),
                    "_card_ids": card_ids,
                    "source": "fabrary",
                    "source_url": fabrary_url,
                    "description": deck_entry.get("description", ""),
                    "side": side,
                },
                indent=2,
            )

        except Exception as exc:
            log.exception("fab_resolve_deck_from_url error")
            return json.dumps(
                {
                    "error": f"Failed to resolve deck: {exc}",
                    "side": side,
                    "deck_id_from_url": deck_id if match else None,
                },
                indent=2,
            )

    @mcp.tool()
    def fab_update_cards_db_from_fabtcg(
        legality_scope: str = "all",
        dry_run: bool = True,
        detail_workers: int = 8,
        cards_path: Optional[str] = None,
        decks_path: Optional[str] = None,
    ) -> str:
        """Run the Card Vault sync script for Flesh and Blood cards DB.

        This MCP tool wraps ``card_db/update_cards_db_from_fabtcg.py`` so callers
        can trigger card legality refreshes and missing-card imports without
        shell access.
        """
        if legality_scope not in {"all", "decks"}:
            return json.dumps(
                {"error": "legality_scope must be one of: all, decks"},
                indent=2,
            )

        if int(detail_workers) < 1:
            return json.dumps(
                {"error": "detail_workers must be >= 1"},
                indent=2,
            )

        if not _UPDATE_CARDS_SCRIPT_PATH.exists():
            return json.dumps(
                {"error": f"Update script not found at {_UPDATE_CARDS_SCRIPT_PATH}"},
                indent=2,
            )

        cmd: list[str] = [
            sys.executable,
            str(_UPDATE_CARDS_SCRIPT_PATH),
            "--legality-scope",
            legality_scope,
            "--detail-workers",
            str(int(detail_workers)),
        ]
        if dry_run:
            cmd.append("--dry-run")
        if cards_path:
            cmd.extend(["--cards", str(cards_path)])
        if decks_path:
            cmd.extend(["--decks", str(decks_path)])

        try:
            run = subprocess.run(  # noqa: S603
                cmd,
                cwd=str(_FAB_DB_DIR),
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            log.exception("fab_update_cards_db_from_fabtcg error")
            return json.dumps(
                {"error": f"Failed to execute card DB updater: {exc}", "command": cmd},
                indent=2,
            )

        return json.dumps(
            {
                "success": run.returncode == 0,
                "exit_code": run.returncode,
                "command": cmd,
                "dry_run": bool(dry_run),
                "legality_scope": legality_scope,
                "detail_workers": int(detail_workers),
                "stdout": run.stdout.strip(),
                "stderr": run.stderr.strip(),
            },
            indent=2,
        )

    @mcp.tool()
    def fab_cached_opponent_agents(
        deck_key: Optional[str] = None,
        matchup_key: Optional[str] = None,
        format_name: str = "silver_age",
        inner_agent_type: str = "ppo",
        use_if_available: bool = True,
        limit: int = 50,
    ) -> str:
        """List cached matchup opponent agents and optionally load one for immediate use."""
        entries = _list_cached_agent_entries()

        filtered = [
            e
            for e in entries
            if (deck_key is None or str(e.get("deck_key", "")) == str(deck_key))
            and (matchup_key is None or str(e.get("matchup_key", "")) == str(matchup_key))
            and (format_name is None or str(e.get("format_name", "")) == str(format_name))
            and (inner_agent_type is None or str(e.get("inner_agent_type", "")) == str(inner_agent_type))
        ]

        filtered.sort(
            key=lambda e: (
                bool(e.get("converged", False)),
                float(e.get("last_win_rate", 0.0)),
                int(e.get("total_train_episodes", 0)),
            ),
            reverse=True,
        )
        if int(limit) > 0:
            filtered = filtered[: int(limit)]

        loaded_cache_key: Optional[str] = None
        loaded_successfully = False
        if use_if_available and filtered:
            target_key = str(filtered[0].get("cache_key", ""))
            if target_key:
                _, loaded_agent, used_cached = _load_cached_matchup_entry(target_key)
                loaded_cache_key = target_key
                loaded_successfully = bool(used_cached and loaded_agent is not None)

        result = {
            "cache_dir": str(_FAB_AGENT_CACHE_DIR),
            "total_cached_entries": len(entries),
            "matched_entries": len(filtered),
            "loaded_cache_key": loaded_cache_key,
            "loaded_successfully": loaded_successfully,
            "entries": filtered,
        }
        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_full_matchup_win_summary(
        format_names_csv: Optional[str] = None,
        inner_agent_type: str = "ppo",
        inner_train_episodes: int = 25,
        inner_eval_episodes: int = 5,
        inner_max_steps: int = 200,
        max_decks_per_format: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Return full directed deck-vs-deck win% summaries for each supported format."""
        formats = _resolve_supported_formats(format_names_csv)
        if not formats:
            return json.dumps(
                {
                    "error": "No supported formats were found for summary generation.",
                    "requested_formats": format_names_csv,
                },
                indent=2,
            )

        all_formats: list[dict[str, Any]] = []
        for fmt in formats:
            try:
                options = _get_deck_options(fmt, seed)
            except Exception as exc:
                all_formats.append(
                    {
                        "format": fmt,
                        "error": f"Failed to load deck options: {exc}",
                        "decks_count": 0,
                        "directed_matchups": 0,
                        "matchups": [],
                        "deck_summaries": [],
                    }
                )
                continue

            if max_decks_per_format is not None and int(max_decks_per_format) > 0:
                options = options[: int(max_decks_per_format)]

            matchups: list[dict[str, Any]] = []
            deck_summaries: list[dict[str, Any]] = []

            for i, deck_option in enumerate(options):
                deck_key = str(deck_option.get("key", ""))
                deck_label = str(deck_option.get("label", deck_key))

                row_scores: list[float] = []
                row_errors = 0
                for j, matchup_option in enumerate(options):
                    matchup_key = str(matchup_option.get("key", ""))
                    if i == j or deck_key == matchup_key:
                        continue

                    try:
                        stats = _evaluate_deck_vs_matchup(
                            deck_option=deck_option,
                            matchup_option=matchup_option,
                            deck_key=deck_key,
                            matchup_key=matchup_key,
                            format_name=fmt,
                            inner_agent_type=inner_agent_type,
                            inner_train_episodes=inner_train_episodes,
                            inner_eval_episodes=inner_eval_episodes,
                            inner_max_steps=inner_max_steps,
                            seed=seed,
                        )
                        win_rate = float(stats.get("win_rate", 0.5))
                        row_scores.append(win_rate)
                        matchups.append(
                            {
                                "deck_key": deck_key,
                                "deck_label": deck_label,
                                "matchup_key": matchup_key,
                                "matchup_label": str(matchup_option.get("label", matchup_key)),
                                "win_rate": round(win_rate, 4),
                                "converged": bool(stats.get("converged", False)),
                                "total_train_episodes": int(stats.get("total_train_episodes", 0)),
                                "used_cached_agent": bool(stats.get("used_cached_agent", False)),
                                "error": None,
                            }
                        )
                    except Exception as exc:
                        row_errors += 1
                        matchups.append(
                            {
                                "deck_key": deck_key,
                                "deck_label": deck_label,
                                "matchup_key": matchup_key,
                                "matchup_label": str(matchup_option.get("label", matchup_key)),
                                "win_rate": 0.5,
                                "converged": False,
                                "total_train_episodes": 0,
                                "used_cached_agent": False,
                                "error": str(exc),
                            }
                        )

                deck_summaries.append(
                    {
                        "deck_key": deck_key,
                        "deck_label": deck_label,
                        "matchups_evaluated": len(row_scores) + row_errors,
                        "mean_win_rate": round(sum(row_scores) / len(row_scores), 4) if row_scores else 0.5,
                        "errors": row_errors,
                    }
                )

            all_formats.append(
                {
                    "format": fmt,
                    "error": None,
                    "decks_count": len(options),
                    "directed_matchups": len(matchups),
                    "matchups": matchups,
                    "deck_summaries": deck_summaries,
                }
            )

        result = {
            "requested_formats": format_names_csv,
            "resolved_formats": formats,
            "inner_agent_type": inner_agent_type,
            "inner_train_episodes": int(inner_train_episodes),
            "inner_eval_episodes": int(inner_eval_episodes),
            "inner_max_steps": int(inner_max_steps),
            "max_decks_per_format": max_decks_per_format,
            "formats": all_formats,
        }
        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_list_dual_agent_training_programs() -> str:
        """List Talishar dual-agent PPO training scripts available in this repo.

        Programs:
        - ``sage_precons`` — SAGE precon Assets decks (blitz)
        - ``silver_age`` — fabrary Silver Age decks
        - ``classic_constructed`` — fabrary Classic Constructed decks

        Use ``fab_run_dual_agent_training`` to execute one.
        """
        return json.dumps(_dual_agent_training_programs_payload(), indent=2)

    @mcp.tool()
    def fab_run_dual_agent_training(
        training_program: str,
        matchup: str = "all",
        episodes: int = 300,
        max_steps: Optional[int] = None,
        game_format: Optional[str] = None,
        seed: Optional[int] = None,
        out_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        talishar_url: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """Run a repo dual-agent self-play training script against Talishar.

        Parameters
        ----------
        training_program:
            One of ``sage_precons``, ``silver_age``, ``classic_constructed``.
            Call ``fab_list_dual_agent_training_programs`` for defaults and examples.
        matchup:
            ``all`` or a matchup slug (e.g. ``briar-vs-dorinthea``).
        episodes, max_steps, game_format, seed, out_dir, cache_dir:
            Forwarded to the underlying script when set.
        talishar_url:
            Sets ``TALISHAR_URL`` for the subprocess (default: server default in script).
        timeout_seconds:
            Subprocess timeout; omit for no limit (long runs).
        """
        program_key = str(training_program).strip().lower().replace("-", "_")
        if program_key == "classic_constructed" or program_key == "cc":
            program_key = "classic_constructed"
        if program_key not in _FAB_DUAL_AGENT_TRAINING_PROGRAMS:
            return json.dumps(
                {
                    "error": (
                        f"Unknown training_program {training_program!r}. "
                        f"Choose from: {list(_FAB_DUAL_AGENT_TRAINING_PROGRAMS)}"
                    ),
                    "available_programs": _dual_agent_training_programs_payload(),
                },
                indent=2,
            )

        try:
            cmd, meta = _build_dual_agent_training_command(
                program_key,
                matchup=matchup,
                episodes=episodes,
                max_steps=max_steps,
                game_format=game_format,
                seed=seed,
                out_dir=out_dir,
                cache_dir=cache_dir,
            )
        except FileNotFoundError as exc:
            return json.dumps({"error": str(exc), "training_program": program_key}, indent=2)

        env = os.environ.copy()
        if talishar_url:
            env["TALISHAR_URL"] = str(talishar_url)

        try:
            run = subprocess.run(  # noqa: S603
                cmd,
                cwd=str(_FAB_REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                env=env,
                timeout=int(timeout_seconds) if timeout_seconds is not None else None,
            )
        except subprocess.TimeoutExpired as exc:
            log.exception("fab_run_dual_agent_training timeout")
            return json.dumps(
                {
                    "error": f"Training subprocess timed out after {timeout_seconds}s",
                    "command": cmd,
                    "meta": meta,
                    "partial_stdout": (exc.stdout or "")[-4000:] if exc.stdout else "",
                    "partial_stderr": (exc.stderr or "")[-4000:] if exc.stderr else "",
                },
                indent=2,
            )
        except Exception as exc:
            log.exception("fab_run_dual_agent_training error")
            return json.dumps(
                {"error": f"Failed to run training script: {exc}", "command": cmd, "meta": meta},
                indent=2,
            )

        return json.dumps(
            {
                "success": run.returncode == 0,
                "exit_code": run.returncode,
                "command": cmd,
                "meta": meta,
                "talishar_url": env.get("TALISHAR_URL"),
                "stdout": run.stdout.strip()[-8000:],
                "stderr": run.stderr.strip()[-8000:],
            },
            indent=2,
        )

    _FAB_CUSTOM_TOOLS_REGISTERED = True
    return 10
