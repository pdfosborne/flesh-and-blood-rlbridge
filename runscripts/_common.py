"""Shared helpers for runscripts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPTS_TRAINING = SCRIPTS_DIR / "training"
SCRIPTS_EVAL = SCRIPTS_DIR / "eval"
SCRIPTS_CPP = SCRIPTS_DIR / "cpp"
SCRIPTS_DECK = SCRIPTS_DIR / "deck"
PYTHON = sys.executable

MIN_DECK_SIZES: dict[str, int] = {
    "silver_age": 40,
    "blitz": 40,
    "classic_constructed": 60,
    "upf": 60,
}


def env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def talishar_url() -> str:
    return env_or_default("TALISHAR_URL", "http://localhost:8080/game")


def assets_path() -> Path:
    return Path(env_or_default("TALISHAR_ASSETS_PATH", str(REPO_ROOT / "Talishar" / "Assets")))


def run_command(cmd: Sequence[str], *, cwd: Path | None = None) -> int:
    print(f"  $ {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT))
    return int(completed.returncode)


def run_python(script: Path | str, *args: str, cwd: Path | None = None) -> int:
    return run_command([PYTHON, str(script), *args], cwd=cwd)


def title_case_token(token: str) -> str:
    token = token.strip()
    if not token:
        return token
    return token[:1].upper() + token[1:].lower()


def matchup_label_from_dir_name(dir_name: str) -> str | None:
    match = re.match(r"^(.+)_vs_(.+)$", dir_name)
    if not match:
        return None
    left = title_case_token(match.group(1).lower())
    right = title_case_token(match.group(2).lower())
    return f"{left}_vs_{right}"


def find_cpp_engine_dir(matchup_label: str, cache_root: Path | None = None) -> Path | None:
    root = cache_root or (REPO_ROOT / "results" / "cpp_engines")
    if not root.is_dir():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path.name == matchup_label or path.name.startswith(f"{matchup_label}-"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_cpp_engine_for_matchup(
    *,
    deck1: str,
    deck2: str,
    talishar_url_value: str | None = None,
    deck1_json: Path | None = None,
    deck2_json: Path | None = None,
    no_server: bool = True,
) -> int:
    cmd: list[str] = [
        PYTHON,
        str(SCRIPTS_CPP / "build_cpp_engine_for_matchup.py"),
        "--deck1",
        deck1,
        "--deck2",
        deck2,
        "--talishar-src",
        str(REPO_ROOT / "Talishar"),
    ]
    url = talishar_url_value or talishar_url()
    if url:
        cmd.extend(["--talishar-url", url])
    if no_server:
        cmd.append("--no-server")
    if deck1_json and deck1_json.is_file():
        cmd.extend(["--deck1-json", str(deck1_json)])
    if deck2_json and deck2_json.is_file():
        cmd.extend(["--deck2-json", str(deck2_json)])
    return run_command(cmd)


@dataclass(frozen=True)
class DeckMeta:
    hero_id: str
    hero_class: str
    equipment_header: str
    fmt: str
    short_name: str
    total_cards: int
    name: str


def read_deck_meta(json_path: Path, default_format: str = "silver_age") -> DeckMeta:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    total_cards = 0
    deck = data.get("deck") or {}
    if isinstance(deck, Mapping):
        for value in deck.values():
            total_cards += int(value)
    hero_id = str(data.get("hero_id") or "")
    short_name = hero_id.split("_")[0] if hero_id else ""
    return DeckMeta(
        hero_id=hero_id,
        hero_class=str(data.get("hero_class") or ""),
        equipment_header=str(data.get("equipment_header") or ""),
        fmt=str(data.get("format") or default_format),
        short_name=short_name,
        total_cards=total_cards,
        name=str(data.get("name") or hero_id),
    )


@dataclass(frozen=True)
class ResolvedDeckSource:
    slug: str | None
    local_path: Path


def resolve_deck_source(source: str, cache_dir: Path) -> ResolvedDeckSource:
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(source)
    if candidate.is_file():
        return ResolvedDeckSource(slug=None, local_path=candidate.resolve())

    slug = source
    match = re.search(r"fabrary\.net/decks/([A-Z0-9]+)", source, re.IGNORECASE)
    if match:
        slug = match.group(1)
    out_file = cache_dir / f"{slug.lower()}_deck.json"
    return ResolvedDeckSource(slug=slug, local_path=out_file)


def fetch_fabrary_deck(slug: str, out_file: Path, label: str) -> bool:
    if out_file.is_file():
        print(f"  [{label}] Deck already fetched -> {out_file}")
        return True
    print(f"  [{label}] Fetching FaBrary deck {slug} ...")
    rc = run_python(
        SCRIPTS_DECK / "fetch_fabrary_deck.py",
        f"https://fabrary.net/decks/{slug}",
        "--out",
        str(out_file),
        "--pretty",
    )
    if rc == 0:
        print(f"  [{label}] Saved -> {out_file}")
        return True
    print(f"  [{label}] WARNING: fetch failed (exit {rc})")
    return False


def print_banner(title: str, *, width: int = 64) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_final_eval(label: str, player: Mapping[str, Any]) -> None:
    if label:
        print(f"  {label}")
    win_rates = player.get("win_rates") or []
    if win_rates:
        joined = ", ".join(str(rate) for rate in win_rates)
        print(f"    Training win rates : {joined}")

    final_eval = player.get("final_eval")
    if isinstance(final_eval, Mapping):
        win_rate = float(final_eval.get("win_rate", 0.0))
        wins = int(final_eval.get("wins", 0))
        losses = int(final_eval.get("losses", 0))
        draws = int(final_eval.get("draws", 0))
        pct = round(win_rate * 100, 1)
        record = f"{wins}W/{losses}L/{draws}D"
        print(f"    Final eval win%    : {pct}%  ({record})")

    gif_path = player.get("final_eval_gif")
    if gif_path:
        print(f"    Render GIF         : {gif_path}")


def print_matchup_player_result(
    label: str,
    deck_name: str,
    player: Mapping[str, Any],
    *,
    final_eval_episodes: int | None = None,
) -> None:
    print()
    print(f"  {label} ({deck_name})")
    final_eval = player.get("final_eval")
    if isinstance(final_eval, Mapping):
        win_rate = float(final_eval.get("win_rate", 0.0))
        wins = int(final_eval.get("wins", 0))
        losses = int(final_eval.get("losses", 0))
        draws = int(final_eval.get("draws", 0))
        pct = round(win_rate * 100, 1)
        record = f"{wins}W / {losses}L"
        if draws > 0:
            record += f" / {draws}D"
        games = final_eval_episodes if final_eval_episodes is not None else "?"
        print(f"    Win rate (final eval) : {pct}%  ({record} over {games} games)")
    else:
        win_rates = player.get("win_rates") or []
        if win_rates:
            last = float(win_rates[-1])
            pct = round(last * 100, 1)
            print(f"    Win rate (last iter)  : {pct}%  (from training)")
        else:
            print("    Win rate              : N/A")

    active_decks = player.get("active_decks")
    if isinstance(active_decks, Mapping) and active_decks:
        sizes = [f"{key}: {value} cards" for key, value in active_decks.items()]
        print(f"    Game deck(s)          : {' | '.join(sizes)}")

    gif_path = None
    if isinstance(final_eval, Mapping):
        render = final_eval.get("render")
        if isinstance(render, Mapping):
            gif_path = render.get("gif")
    if gif_path:
        print(f"    Render GIF            : {gif_path}")


def describe_deck_size(meta: DeckMeta, min_size: int) -> None:
    if meta.total_cards >= min_size:
        if meta.total_cards > min_size:
            print(
                f"           -> Pool ({meta.total_cards} cards): greedy-cut to {min_size} for play"
            )
        else:
            print(
                f"           -> Game-ready ({meta.total_cards} cards): sideboard skipped"
            )
    else:
        print(
            f"           -> Below minimum ({meta.total_cards} < {min_size}): sideboard RL will run"
        )


def optional_train_workers_arg(play_workers: int | None) -> list[str]:
    if play_workers is None:
        return []
    return ["--workers", str(play_workers)]


def optional_cpp_engine_arg(engine_dir: Path | None) -> list[str]:
    if engine_dir is None:
        return []
    return ["--cpp-engine-dir", str(engine_dir)]


def load_results_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extend_if_present(args: list[str], flag: str, value: str | None) -> None:
    if value:
        args.extend([flag, value])
