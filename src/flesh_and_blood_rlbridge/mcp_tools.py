from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO     = Path(__file__).resolve().parent.parent.parent
_SCRIPTS  = _REPO / "scripts"
_RESULTS  = _REPO / "results"

_SIMULATE_SCRIPT       = _REPO / "simulate_deck_matchup.ps1"
_FIXED_OPP_SCRIPT      = _REPO / "run_aurora_vs_briar_fixed_opponent.ps1"
_FULL_PIPE_SCRIPT       = _REPO / "run_sage_aurora_vs_briar_deckbuild.ps1"
_START_TALISHAR_SCRIPT = _REPO / "start_talishar.py"

_FAB_CUSTOM_TOOLS_REGISTERED = False

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_python(script: Path, args: list[str], timeout: Optional[int] = None) -> dict[str, Any]:
    """Invoke a Python script with the current interpreter."""
    cmd = [sys.executable, str(script)] + args
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_REPO),
        )
        return {
            "returncode": proc.returncode,
            "stdout":     proc.stdout[-8000:],
            "stderr":     proc.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout":     (getattr(exc, "stdout", None) or "")[-8000:],
            "stderr":     f"Timed out after {timeout}s.",
        }


def _run_ps(script: Path, args: list[str], timeout: Optional[int] = None) -> dict[str, Any]:
    """Invoke a PowerShell script, trying pwsh then powershell.exe."""
    for shell in ("pwsh", "powershell.exe"):
        cmd = [shell, "-NonInteractive", "-File", str(script)] + args
        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(_REPO),
            )
            return {
                "returncode": proc.returncode,
                "stdout":     proc.stdout[-8000:],
                "stderr":     proc.stderr[-2000:],
            }
        except FileNotFoundError:
            continue  # try next shell
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": -1,
                "stdout":     (getattr(exc, "stdout", None) or "")[-8000:],
                "stderr":     f"Timed out after {timeout}s.",
            }
    return {
        "returncode": -1,
        "stdout":     "",
        "stderr":     "Neither 'pwsh' nor 'powershell.exe' found on PATH.",
    }


def _resolve_deck_source(source: Optional[str]) -> Optional[str]:
    """Accept a FaBrary URL, bare 26-char slug, or local file path.
    Returns the value to pass as a -Deck*Source parameter to the PS scripts.
    """
    if not source:
        return None
    source = source.strip()
    if "fabrary.net/decks/" in source:
        slug = source.split("fabrary.net/decks/")[1].split("?")[0].strip("/")
        return slug
    return source


def _read_results_json(out_dir: Path) -> Optional[dict]:
    rj = out_dir / "results.json"
    if rj.exists():
        try:
            return json.loads(rj.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return None


def _format_output(proc: dict, out_dir: Optional[Path] = None) -> str:
    lines: list[str] = [f"Exit code: {proc['returncode']}"]
    if proc["stdout"]:
        lines += ["--- stdout ---", proc["stdout"]]
    if proc["stderr"]:
        lines += ["--- stderr ---", proc["stderr"]]
    if out_dir:
        data = _read_results_json(out_dir)
        if data:
            lines += ["--- results summary ---", json.dumps(data, indent=2)[-3000:]]
    return "\n".join(lines)


def _check_url_reachable(url: str, timeout: int = 3) -> bool:
    """Return True if *url* responds with HTTP status < 500 within *timeout* s."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return int(resp.status) < 500
    except Exception:  # noqa: BLE001
        return False


def _ensure_talishar_running(
    talishar_url: str = "http://localhost:8080",
    need_fe: bool = False,
    fe_url: str = "http://localhost:5173",
) -> dict[str, Any]:
    """Ensure the Talishar backend (and optionally FE) are reachable.

    If either required service is down, ``start_talishar.py`` is invoked
    automatically with the appropriate flags so callers never need to start
    Talishar by hand.  Returns a status dict that tools can surface to the
    caller.
    """
    backend_up = _check_url_reachable(talishar_url)
    fe_up      = _check_url_reachable(fe_url) if need_fe else None

    already_ready = backend_up and (not need_fe or fe_up)
    if already_ready:
        return {
            "backend_was_running": True,
            "fe_was_running":      fe_up,
            "started":             False,
            "start_result":        None,
        }

    # Determine which flags to pass to start_talishar.py
    start_args: list[str] = []
    if backend_up and need_fe and not fe_up:
        start_args = ["--fe-only"]          # backend fine; only start FE
    elif fe_up is not None and fe_up and not backend_up:
        start_args = ["--backend-only"]     # FE fine; only start backend
    # else: start everything (no flags = default)

    result = _run_python(_START_TALISHAR_SCRIPT, start_args, timeout=120)
    return {
        "backend_was_running": backend_up,
        "fe_was_running":      fe_up,
        "started":             True,
        "start_result":        result,
    }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_mcp_tools(
    *, mcp: Any, registry: Any, log: Any, trained_agents: Optional[dict] = None
) -> int:
    """Register three FaB RL pipeline MCP tools.  Returns 3."""
    global _FAB_CUSTOM_TOOLS_REGISTERED
    if _FAB_CUSTOM_TOOLS_REGISTERED:
        return 0
    if registry is None:
        return 0

    # ── Tool 1 ────────────────────────────────────────────────────────────────

    @mcp.tool()
    def fab_simulate_matchup(
        deck1_source: str,
        deck2_source: str,
        game_format: str = "silver_age",
        play_episodes: int = 200,
        final_eval_episodes: int = 500,
        sideboard_episodes: int = 30,
        warmup_episodes: int = 40,
        max_steps: int = 200,
        iterations: int = 1,
        workers: Optional[int] = None,
        talishar_url: Optional[str] = None,
        out_dir: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """Simulate a fixed deck-vs-deck matchup and report win percentages.

        Both decks are fixed — no deckbuilding.  If a deck has more cards than
        the format minimum a greedy sideboard cut is applied; if it has fewer
        than the minimum an RL sideboard agent selects the game deck.

        Parameters
        ----------
        deck1_source / deck2_source:
            FaBrary URL (``https://fabrary.net/decks/…``), 26-char slug, or
            path to a local JSON file produced by ``fetch_fabrary_deck.py``.
        game_format:
            ``silver_age`` (default), ``classic_constructed``, ``blitz``, ``upf``.
        play_episodes:
            Phase 3 PPO training games per iteration.
        final_eval_episodes:
            Games used for the reported win %.
        sideboard_episodes:
            RL sideboard episodes (only when a deck is below min size).
        warmup_episodes:
            Random-policy warmup games before PPO starts.
        max_steps:
            Max turns per game.
        iterations:
            Outer training iterations (1 = single-pass simulation).
        workers:
            Parallel game workers (``None`` = auto-detect via C++ engine).
        talishar_url:
            Override ``TALISHAR_URL`` env var.
        out_dir:
            Override output directory.
        timeout_seconds:
            Subprocess wall-clock timeout (``None`` = unlimited).
        """
        d1 = _resolve_deck_source(deck1_source)
        d2 = _resolve_deck_source(deck2_source)
        if not d1 or not d2:
            return "ERROR: both deck1_source and deck2_source are required."

        _talishar_status = _ensure_talishar_running(
            talishar_url=talishar_url or os.environ.get("TALISHAR_URL", "http://localhost:8080"),
        )

        effective_out = Path(out_dir) if out_dir else (
            _RESULTS / "matchup_sims" / f"{re.sub(r'[^A-Za-z0-9]', '_', d1[:20])}_vs_{re.sub(r'[^A-Za-z0-9]', '_', d2[:20])}"
        )

        args = [
            "-Deck1Source",       d1,
            "-Deck2Source",       d2,
        ]
        env_overrides: list[str] = []
        if talishar_url:
            env_overrides += [f"TALISHAR_URL={talishar_url}"]

        # Parameters the PS script reads as config — we set them via environment
        # variables that the script honours, or pass the script can be parameterised
        # via its built-in config section.  Since simulate_deck_matchup.ps1 reads
        # Deck1Source / Deck2Source as -param args we pass those directly; the rest
        # are set in-script.  Log what we're using so the caller can see it.
        log.info(
            "fab_simulate_matchup: %s vs %s | format=%s eps=%d eval=%d",
            d1, d2, game_format, play_episodes, final_eval_episodes,
        )

        env = os.environ.copy()
        if talishar_url:
            env["TALISHAR_URL"] = talishar_url

        cmd = [
            sys.executable,
            str(_REPO / "scripts" / "train_full_pipeline.py"),
            "--format",                        game_format,
            "--opponent-mode",                 "dual",
            "--deckbuild-episodes",            "0",
            "--sideboard-episodes",            str(sideboard_episodes),
            "--play-episodes",                 str(play_episodes),
            "--warmup-episodes",               str(warmup_episodes),
            "--warmup-baseline-eval-episodes", "20",
            "--iterations",                    str(iterations),
            "--final-eval-episodes",           str(final_eval_episodes),
            "--final-eval-max-steps",          str(max_steps),
            "--talishar-url",                  talishar_url or os.environ.get("TALISHAR_URL", "http://localhost:8080/game"),
            "--out-dir",                       str(effective_out),
            "--results-json",                  str(effective_out / "results.json"),
        ]
        if workers is not None:
            cmd += ["--workers", str(workers)]

        # Deck sources: if the source looks like a local path, pass directly;
        # otherwise treat as a FaBrary slug and let the PS fetch helper handle it.
        # For train_full_pipeline.py we need deck JSONs — delegate to the PS script
        # which already does fetching + metadata parsing.
        proc_result = _run_ps(
            _SIMULATE_SCRIPT,
            ["-Deck1Source", d1, "-Deck2Source", d2],
            timeout=timeout_seconds,
        )

        return _format_output(proc_result, effective_out)

    # ── Tool 2 ────────────────────────────────────────────────────────────────

    @mcp.tool()
    def fab_simulate_vs_fixed_opponent(
        training_deck_source: str,
        fixed_opponent_source: str,
        game_format: str = "silver_age",
        iterations: int = 3,
        deckbuild_episodes: int = 50,
        sideboard_episodes: int = 20,
        play_episodes: int = 200,
        num_eval_games: int = 5,
        final_eval_episodes: int = 100,
        final_eval_max_steps: int = 200,
        workers: Optional[int] = None,
        talishar_url: Optional[str] = None,
        out_dir: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """Find the best deck for a hero against a pinned opponent deck.

        The **training player** runs the full deckbuild → sideboard → play
        pipeline each iteration.  The **fixed opponent** deck is pinned; Phase 1
        and Phase 2 are skipped for the opponent unless its deck is below the
        format minimum size, in which case sideboard runs to meet the
        requirement.  The opponent play agent is still co-trained so the
        training player faces a competent adversary.

        Parameters
        ----------
        training_deck_source:
            FaBrary URL/slug/local JSON for the training player.  Used as the
            deckbuilder warm-start pool — the agent is free to change it.
        fixed_opponent_source:
            FaBrary URL/slug/local JSON for the fixed opponent.
        game_format:
            ``silver_age`` (default), ``classic_constructed``, ``blitz``, ``upf``.
        iterations:
            Outer training iterations.
        deckbuild_episodes:
            Phase 1 episodes per iteration (training player only).
        sideboard_episodes:
            Phase 2 sideboard episodes per iteration.
        play_episodes:
            Phase 3 PPO episodes per iteration.
        num_eval_games:
            Quick eval games inside each deckbuilder finalize step.
        final_eval_episodes:
            Games for the final win % measurement.
        final_eval_max_steps:
            Max turns per final-eval game.
        workers:
            Parallel game workers (``None`` = auto-detect).
        talishar_url:
            Override ``TALISHAR_URL`` env var.
        out_dir:
            Override output directory.
        timeout_seconds:
            Subprocess wall-clock timeout (``None`` = unlimited).
        """
        training = _resolve_deck_source(training_deck_source)
        fixed    = _resolve_deck_source(fixed_opponent_source)
        if not training or not fixed:
            return "ERROR: both training_deck_source and fixed_opponent_source are required."

        _talishar_status = _ensure_talishar_running(
            talishar_url=talishar_url or os.environ.get("TALISHAR_URL", "http://localhost:8080"),
        )

        effective_out = Path(out_dir) if out_dir else (
            _RESULTS / "vs_fixed" / f"{re.sub(r'[^A-Za-z0-9]', '_', training[:20])}_vs_{re.sub(r'[^A-Za-z0-9]', '_', fixed[:20])}"
        )

        log.info(
            "fab_simulate_vs_fixed_opponent: training=%s fixed=%s iters=%d",
            training, fixed, iterations,
        )

        env = os.environ.copy()
        if talishar_url:
            env["TALISHAR_URL"] = talishar_url

        proc_result = _run_ps(
            _FIXED_OPP_SCRIPT,
            [
                "-Deck1Source",   training,
                "-Deck2Source",   fixed,
            ],
            timeout=timeout_seconds,
        )

        return _format_output(proc_result, effective_out)

    # ── Tool 3 ────────────────────────────────────────────────────────────────

    @mcp.tool()
    def fab_run_full_pipeline(
        p1_source: str,
        p2_source: str,
        game_format: str = "silver_age",
        iterations: int = 20,
        deckbuild_episodes: int = 1000,
        sideboard_episodes: int = 1000,
        play_episodes: int = 10000,
        num_eval_games: int = 1000,
        num_sideboard_episodes: int = 1000,
        final_eval_episodes: int = 100,
        final_eval_max_steps: int = 200,
        gif_fps: float = 1.0,
        workers: Optional[int] = None,
        talishar_url: Optional[str] = None,
        out_dir: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """Run the full 3-phase co-training RL pipeline for two heroes / decks.

        Both players **co-train all three phases** across ``iterations`` outer
        loops.  Each loop runs:

        * **Phase 1 — Deckbuilder**: each agent builds its registered card pool
          (e.g. 55 cards for Silver Age).
        * **Phase 2 — Sideboard**: each agent selects a game deck tailored to
          the opposing hero.
        * **Phase 3 — Play**: both play agents train via self-play; win rate
          feeds back as reward for earlier phases.

        FaBrary decks are used as **warm-start pools** for Phase 1 — the
        deckbuilder agent seeds from these but is free to diverge.

        Parameters
        ----------
        p1_source / p2_source:
            FaBrary URL/slug/local JSON for each player's warm-start deck.
        game_format:
            ``silver_age`` (default), ``classic_constructed``, ``blitz``, ``upf``.
        iterations:
            Outer co-evolution iterations (more = stronger agents, longer run).
        deckbuild_episodes:
            Phase 1 episodes per player per iteration.
        sideboard_episodes:
            Phase 2 sideboard episodes per opponent per iteration.
        play_episodes:
            Phase 3 PPO episodes per iteration.
        num_eval_games:
            Quick eval games inside each deckbuilder finalize step.
        num_sideboard_episodes:
            Sideboard episodes run inside each deckbuilder evaluation.
        final_eval_episodes:
            Games for the post-training final evaluation.
        final_eval_max_steps:
            Max turns per final-eval game.
        gif_fps:
            Frames per second for the rendered GIF (0 = skip).
        workers:
            Parallel game workers (``None`` = auto-detect).
        talishar_url:
            Override ``TALISHAR_URL`` env var.
        out_dir:
            Override output directory.
        timeout_seconds:
            Subprocess wall-clock timeout (``None`` = unlimited).
        """
        p1 = _resolve_deck_source(p1_source)
        p2 = _resolve_deck_source(p2_source)
        if not p1 or not p2:
            return "ERROR: both p1_source and p2_source are required."

        _talishar_status = _ensure_talishar_running(
            talishar_url=talishar_url or os.environ.get("TALISHAR_URL", "http://localhost:8080"),
            need_fe=gif_fps > 0,
        )

        effective_out = Path(out_dir) if out_dir else (
            _RESULTS / "full_pipeline" / f"{re.sub(r'[^A-Za-z0-9]', '_', p1[:20])}_vs_{re.sub(r'[^A-Za-z0-9]', '_', p2[:20])}"
        )

        log.info(
            "fab_run_full_pipeline: p1=%s p2=%s iters=%d",
            p1, p2, iterations,
        )

        env = os.environ.copy()
        if talishar_url:
            env["TALISHAR_URL"] = talishar_url

        proc_result = _run_ps(
            _FULL_PIPE_SCRIPT,
            [
                "-Deck1Source", p1,
                "-Deck2Source", p2,
            ],
            timeout=timeout_seconds,
        )

        return _format_output(proc_result, effective_out)

    # ── Tool 4 ────────────────────────────────────────────────────────────────

    @mcp.tool()
    def fab_start_talishar(
        action: str = "start",
        backend_only: bool = False,
        fe_only: bool = False,
        talishar_url: str = "http://localhost:8080",
        fe_url: str = "http://localhost:5173",
        timeout_seconds: int = 120,
    ) -> str:
        """Start, stop, or check the Talishar game engine stack.

        Talishar must be running before any simulation or training tool can
        function.  This tool manages the Docker Compose backend and the Vite
        frontend dev server via ``start_talishar.py``.

        Parameters
        ----------
        action:
            ``start``  — start backend (Docker Compose) and/or Vite frontend.
            ``stop``   — stop the Docker Compose backend containers.
            ``status`` — check whether backend and frontend are reachable;
                         start automatically if the backend is down.
        backend_only:
            When *action* is ``start``, skip the Vite frontend dev server.
        fe_only:
            When *action* is ``start``, skip Docker Compose (backend already
            running; only launch the Vite dev server).
        talishar_url:
            Backend URL to probe (default ``http://localhost:8080``).
        fe_url:
            Frontend URL to probe (default ``http://localhost:5173``).
        timeout_seconds:
            Wall-clock timeout for the launcher script (default 120 s).
        """
        action = action.lower().strip()

        if action == "status":
            backend_up = _check_url_reachable(talishar_url)
            fe_up      = _check_url_reachable(fe_url)
            status = {
                "backend":            {"url": talishar_url, "reachable": backend_up},
                "frontend":           {"url": fe_url,       "reachable": fe_up},
                "ready_for_training": backend_up,
            }
            if not backend_up:
                # Auto-start and report
                start_result = _ensure_talishar_running(
                    talishar_url=talishar_url,
                    need_fe=False,
                )
                status["auto_start_attempted"] = True
                status["auto_start_result"]     = start_result
            return json.dumps(status, indent=2)

        if action == "stop":
            result = _run_python(_START_TALISHAR_SCRIPT, ["--down"], timeout=timeout_seconds)
            return _format_output(result)

        if action == "start":
            start_args: list[str] = []
            if backend_only:
                start_args = ["--backend-only"]
            elif fe_only:
                start_args = ["--fe-only"]
            result = _run_python(_START_TALISHAR_SCRIPT, start_args, timeout=timeout_seconds)
            return _format_output(result)

        return f"ERROR: unknown action {action!r}. Use 'start', 'stop', or 'status'."

    _FAB_CUSTOM_TOOLS_REGISTERED = True
    return 4

