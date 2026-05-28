from __future__ import annotations

import importlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .environment_factory import FleshAndBloodFactory

_FAB_DB_DIR = Path(__file__).with_name("card_db")
_FABRARY_DECKS_PATH = _FAB_DB_DIR / "fabrary_decks.json"
_UPDATE_CARDS_SCRIPT_PATH = _FAB_DB_DIR / "update_cards_db_from_fabtcg.py"

_FAB_CUSTOM_TOOLS_REGISTERED = False


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

    def _run_eval_episode(env: Any, agent: Any, max_steps: int, seed: Optional[int]) -> dict[str, Any]:
        reset_out = env.reset(seed=seed)
        obs = reset_out.observation if hasattr(reset_out, "observation") else reset_out.get("observation", reset_out)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        for step in range(1, max_steps + 1):
            if hasattr(agent, "act_greedy"):
                action = agent.act_greedy(obs)
            else:
                action = agent.act(obs)
            out = env.step(action)
            obs = out.observation if hasattr(out, "observation") else out.get("observation", obs)
            reward = float(out.reward if hasattr(out, "reward") else out.get("reward", 0.0))
            terminated = bool(out.terminated if hasattr(out, "terminated") else out.get("terminated", False))
            truncated = bool(out.truncated if hasattr(out, "truncated") else out.get("truncated", False))
            total_reward += reward
            steps = step
            if terminated or truncated:
                break

        return {
            "steps": steps,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "final_observation": obs,
        }

    def _fab_win_probabilities(obs: Any) -> tuple[float, float]:
        if not isinstance(obs, dict):
            return 0.5, 0.5

        agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
        opp = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}

        agent_life = float(agent.get("life", 0.0))
        opp_life = float(opp.get("life", 0.0))

        if opp_life <= 0 < agent_life:
            return 1.0, 0.0
        if agent_life <= 0 < opp_life:
            return 0.0, 1.0

        agent_hand_size = len(agent.get("hand", [])) if isinstance(agent.get("hand"), list) else 0
        opp_hand_size = int(opp.get("hand_size", 0) or 0)

        agent_resources = float(agent.get("resources", 0.0))
        opp_resources = float(opp.get("resources", 0.0))
        agent_ap = float(agent.get("action_points", 0.0))
        opp_ap = float(opp.get("action_points", 0.0))
        agent_deck = float(agent.get("deck", 0.0))
        opp_deck = float(opp.get("deck", 0.0))

        agent_score = (
            1.8 * agent_life
            + 1.0 * agent_hand_size
            + 0.6 * agent_resources
            + 0.8 * agent_ap
            + 0.05 * agent_deck
        )
        opp_score = (
            1.8 * opp_life
            + 1.0 * opp_hand_size
            + 0.6 * opp_resources
            + 0.8 * opp_ap
            + 0.05 * opp_deck
        )

        pending = obs.get("pending_combat")
        if isinstance(pending, dict):
            atk = float(pending.get("attack_power", 0.0) or 0.0)
            blk = float(pending.get("total_block", 0.0) or 0.0)
            net = max(0.0, atk - blk)
            attacker = int(pending.get("attacker", 0) or 0)
            if attacker == 0:
                agent_score += 1.5 * net
            else:
                opp_score += 1.5 * net

        active_player = int(obs.get("active_player", 0) or 0)
        if active_player == 0:
            agent_score += 0.4
        else:
            opp_score += 0.4

        diff = (agent_score - opp_score) / 8.0
        agent_p = 1.0 / (1.0 + math.exp(-diff))
        agent_p = max(0.0, min(1.0, agent_p))
        return agent_p, 1.0 - agent_p

    def _fab_outcome_score(obs: Any, *, terminated: bool) -> float:
        if isinstance(obs, dict):
            agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
            opp = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}
            agent_life = float(agent.get("life", 0.0) or 0.0)
            opp_life = float(opp.get("life", 0.0) or 0.0)
            if terminated:
                if opp_life <= 0 < agent_life:
                    return 1.0
                if agent_life <= 0 < opp_life:
                    return 0.0
                if agent_life == opp_life:
                    return 0.5
        p_agent, _ = _fab_win_probabilities(obs)
        return float(p_agent)

    def _get_deck_options(format_name: str, seed: Optional[int]) -> list[dict[str, Any]]:
        env = registry.create("FleshAndBlood-DeckBuild-v0", render_mode=None, format=format_name)
        try:
            reset_out = env.reset(seed=seed, options={"format": format_name, "two_phase_deckbuild": True})
            obs = reset_out.observation if hasattr(reset_out, "observation") else reset_out.get("observation", {})
            options = obs.get("deck_options") if isinstance(obs, dict) else None
            return list(options) if isinstance(options, list) else []
        finally:
            env.close()

    def _evaluate_deck_vs_matchup(
        *,
        deck_option: dict[str, Any],
        matchup_option: dict[str, Any],
        format_name: str,
        inner_agent_type: str,
        inner_train_episodes: int,
        inner_eval_episodes: int,
        inner_max_steps: int,
        seed: Optional[int],
    ) -> dict[str, Any]:
        env_kwargs: dict[str, Any] = {
            "render_mode": None,
            "format": format_name,
            "agent_hero_id": str(deck_option.get("hero_id")),
            "opponent_hero_id": str(matchup_option.get("hero_id")),
            "deck_size": int(deck_option.get("deck_size", 40) or 40),
            "agent_deck_style": str(deck_option.get("style", "balanced")),
            "opponent_deck_style": str(matchup_option.get("style", "balanced")),
        }

        agent = _build_agent(inner_agent_type, {})
        train_env = registry.create("FleshAndBlood-Talishar-v0", **env_kwargs)
        try:
            train_result = agent.train(
                train_env,
                n_episodes=inner_train_episodes,
                max_steps=inner_max_steps,
                seed=seed,
            )
        finally:
            train_env.close()

        eval_env = registry.create("FleshAndBlood-Talishar-v0", **env_kwargs)
        eval_scores: list[float] = []
        try:
            base_seed = 0 if seed is None else int(seed)
            for ep in range(inner_eval_episodes):
                ep_seed = base_seed + 10_000 + ep
                out = _run_eval_episode(eval_env, agent, max_steps=inner_max_steps, seed=ep_seed)
                eval_scores.append(
                    _fab_outcome_score(
                        out.get("final_observation"),
                        terminated=bool(out.get("terminated", False)),
                    )
                )
        finally:
            eval_env.close()

        win_rate = (sum(eval_scores) / len(eval_scores)) if eval_scores else 0.5
        return {
            "win_rate": float(win_rate),
            "train_mean_reward": float(train_result.mean_reward),
            "train_best_reward": float(train_result.best_reward),
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

        agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
        opp = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}

        result = {
            "agent_win_probability": round(agent_p, 4),
            "opponent_win_probability": round(opp_p, 4),
            "inputs": {
                "agent_life": agent.get("life"),
                "opponent_life": opp.get("life"),
                "agent_hand_size": len(agent.get("hand", [])) if isinstance(agent.get("hand"), list) else agent.get("hand_size"),
                "opponent_hand_size": opp.get("hand_size"),
                "active_player": obs.get("active_player"),
            },
            "reasoning": (
                "Logistic model over life totals (x1.8), hand size (x1.0), "
                "resources (x0.6), action points (x0.8), deck size (x0.05), "
                "pending combat net damage (x1.5), and initiative bonus (+/-0.4)."
            ),
        }
        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_evaluate_deck_matchup(
        deck_key: str,
        matchup_key: str,
        format_name: str = "silver_age",
        inner_agent_type: str = "tabular_q",
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
            import uuid as _uuid  # noqa: PLC0415

            _agent_id = _uuid.uuid4().hex[:12]
            _registered_env_id = f"FleshAndBlood-matchup-{_agent_id}"

            # Register a factory baked with this matchup's hero / format so
            # rl_render_policy can recreate the exact environment.
            matchup_factory = FleshAndBloodFactory(
                _registered_env_id,
                agent_hero_id=str(deck_option.get("hero_id", "hero_dorinthea_ironsong")),
                opponent_hero_id=str(matchup_option.get("hero_id", "hero_rhinar_reckless_rampage")),
                deck_size=int(deck_option.get("deck_size", 40) or 40),
                format=format_name,
            )
            registry.register(matchup_factory)

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
                },
            }

            result["agent_id"] = _agent_id
            result["registered_env_id"] = _registered_env_id

        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_meta_reward_for_deck(
        deck_key: str,
        format_name: str = "silver_age",
        inner_agent_type: str = "tabular_q",
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
        try:
            # Create temp env to access format normalization
            temp_env = registry.create("FleshAndBlood-Talishar-v0", render_mode=None, format=format_name)
            normalized_format = temp_env._format
            temp_env.close()
        except Exception as exc:
            return json.dumps(
                {
                    "error": f"Invalid format {format_name!r}: {exc}",
                    "side": side,
                    "url": fabrary_url,
                },
                indent=2,
            )

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

            # Resolve the deck to card IDs using the environment's logic
            temp_env = registry.create("FleshAndBlood-Talishar-v0", render_mode=None, format=normalized_format)
            card_ids = temp_env._resolve_fabrary_deck(deck_entry)
            temp_env.close()

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

    _FAB_CUSTOM_TOOLS_REGISTERED = True
    return 6
