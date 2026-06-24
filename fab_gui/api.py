"""API helpers backing the FAB RL Bridge web GUI."""

from __future__ import annotations

import json
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fab_tui.card_search import CardHit, CardSearchIndex
from fab_tui.config import (
    REPO_ROOT,
    SCRIPTS_EVAL,
    EnvironmentSettings,
    SideboardCompareSpec,
    slugify,
)
from fab_tui.decks import (
    DECK_CACHE as TUI_DECK_CACHE,
    export_precon_deck_json,
    list_precon_options,
    read_deck_format,
    read_deck_hero_info,
    resolve_deck_link,
)
from fab_tui.equipment import (
    EquipmentSearchIndex,
    parse_equipment_header,
    slot_display_name,
)
from fab_tui.runner import fetch_fabrary_deck, run_sideboard_compare
from fab_tui.saved_decks import list_saved_user_decks, save_user_deck
from fab_tui.sideboard_picker import (
    ManualSwapVariant,
    apply_manual_swap,
    compute_guide_policy_deck,
    write_candidates_manifest,
)

_SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
if str(SCRIPTS_EVAL) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_EVAL))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_dashboard_once(out_dir: Path) -> None:
    try:
        from sideboard_compare_dashboard import write_sideboard_compare_dashboard  # noqa: PLC0415

        write_sideboard_compare_dashboard(out_dir, auto_refresh_seconds=None)
    except Exception:
        pass


TALISHAR_CARD_IMAGES_CDN = "https://images.talishar.net/public/cardimages/english"
_HTTP_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; flesh-and-blood-rlbridge/1.0; +fab-gui)"
    ),
    "Accept": "image/webp,image/*,*/*;q=0.8",
}


def card_image_url(talishar_url: str, card_id: str) -> str:
    base = talishar_url.rstrip("/")
    return f"{base}/WebpImages/{card_id}.webp"


def card_image_cdn_url(card_id: str) -> str:
    from urllib.parse import quote

    token = str(card_id or "").strip()
    return f"{TALISHAR_CARD_IMAGES_CDN}/{quote(token, safe='')}.webp"


def card_image_display_url(card_id: str) -> str:
    """Browser-facing card art URL (Talishar public CDN)."""
    return card_image_cdn_url(card_id)


def card_image_proxy_path(card_id: str) -> str:
    """Same-origin path served by the GUI for Talishar card art."""
    from urllib.parse import quote

    token = str(card_id or "").strip()
    return f"/api/card-image/{quote(token, safe='')}"


def _card_image_id_variants(card_id: str) -> list[str]:
    token = str(card_id or "").strip()
    if not token:
        return []
    variants = [token]
    hyphen = token.replace("_", "-")
    if hyphen not in variants:
        variants.append(hyphen)
    underscore = token.replace("-", "_")
    if underscore not in variants:
        variants.append(underscore)
    return variants


def _card_image_fetch_urls(env: EnvironmentSettings, card_id: str) -> list[str]:
    """Ordered image sources: public CDN first, then local Talishar WebpImages."""
    base = env.talishar_url.rstrip("/")
    urls: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        if url not in seen:
            seen.add(url)
            urls.append(url)

    for cid in _card_image_id_variants(card_id):
        add(card_image_cdn_url(cid))
        add(f"{base}/WebpImages/{cid}.webp")
        add(f"{base}/WebpImages/en/{cid}.webp")
    return urls


def _fetch_webp_url(url: str) -> tuple[bytes, str] | None:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers=_HTTP_FETCH_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
            if data:
                return data, "image/webp"
    except (OSError, urllib.error.URLError, ValueError):
        return None
    return None


def fetch_card_image(env: EnvironmentSettings, card_id: str) -> tuple[bytes, str] | None:
    """Fetch WebP bytes from local Talishar or the Talishar public CDN."""
    if not str(card_id or "").strip():
        return None
    for url in _card_image_fetch_urls(env, card_id):
        payload = _fetch_webp_url(url)
        if payload is not None:
            return payload
    return None


def card_hit_to_dict(hit: CardHit, *, talishar_url: str) -> dict[str, Any]:
    classification = hit.classification or hit.type_line
    return {
        "card_id": hit.card_id,
        "name": hit.name,
        "pitch": hit.pitch,
        "cost": hit.cost,
        "power": hit.power,
        "defense": hit.defense,
        "card_class": hit.card_class,
        "type_line": hit.type_line,
        "classification": classification,
        "image_url": card_image_display_url(hit.card_id),
    }


def deck_counts_to_entries(
    counts: dict[str, int],
    *,
    game_format: str,
    talishar_url: str,
) -> list[dict[str, Any]]:
    index = CardSearchIndex(game_format)
    entries: list[dict[str, Any]] = []
    for card_id, count in sorted(counts.items(), key=lambda item: index.display_name(item[0])):
        if int(count) <= 0:
            continue
        hit = index.lookup(card_id)
        entries.append(
            {
                "card_id": card_id,
                "count": int(count),
                "name": hit.name if hit else card_id.replace("_", " ").title(),
                "pitch": hit.pitch if hit else None,
                "type_line": hit.type_line if hit else "",
                "classification": hit.classification if hit else "",
                "image_url": card_image_display_url(card_id),
            }
        )
    return entries


def load_deck_payload(path: Path, env: EnvironmentSettings) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    game_format = str(data.get("format") or "silver_age")
    deck = {str(k): int(v) for k, v in (data.get("deck") or {}).items() if int(v) > 0}
    sideboard = {
        str(k): int(v) for k, v in (data.get("sideboard") or {}).items() if int(v) > 0
    }
    pool = dict(deck)
    for cid, count in sideboard.items():
        pool[cid] = pool.get(cid, 0) + count
    info = read_deck_hero_info(path)
    return {
        "path": str(path.resolve()),
        "name": str(data.get("name") or (info.name if info else path.stem)),
        "hero_id": info.hero_id if info else str(data.get("hero_id") or ""),
        "hero_class": info.hero_class if info else str(data.get("hero_class") or ""),
        "equipment_header": info.equipment_header if info else str(data.get("equipment_header") or ""),
        "game_format": game_format,
        "deck": deck,
        "sideboard": sideboard,
        "card_pool": pool,
        "deck_entries": deck_counts_to_entries(deck, game_format=game_format, talishar_url=env.talishar_url),
        "sideboard_entries": deck_counts_to_entries(
            sideboard, game_format=game_format, talishar_url=env.talishar_url
        ),
        "pool_entries": deck_counts_to_entries(pool, game_format=game_format, talishar_url=env.talishar_url),
    }


def list_precons(env: EnvironmentSettings) -> list[dict[str, str]]:
    assets = Path(env.assets_path)
    return [
        {"label": label, "deck_name": deck_name}
        for label, deck_name in list_precon_options(assets)
    ]


def list_saved_decks_api() -> list[dict[str, Any]]:
    return [
        {
            "deck_id": entry.deck_id,
            "label": entry.label,
            "path": str(entry.path),
            "hero_id": entry.hero_id,
            "opponent_hero_id": entry.opponent_hero_id,
            "game_format": entry.game_format,
            "saved_at": entry.saved_at,
        }
        for entry in list_saved_user_decks()
    ]


def import_precon(deck_name: str, env: EnvironmentSettings) -> Path:
    cache_dir = TUI_DECK_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"precon_{slugify(deck_name)}.json"
    export_precon_deck_json(deck_name, Path(env.assets_path), out, game_format="sage")
    return out


def import_fabrary(url_or_slug: str) -> Path:
    cache_dir = TUI_DECK_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = resolve_deck_link(
        url_or_slug,
        label="gui",
        cache_dir=cache_dir,
        fetch_fn=fetch_fabrary_deck,
    )
    if path is None:
        raise ValueError(f"Could not resolve deck: {url_or_slug}")
    return path


def _load_opponent_pool(env: EnvironmentSettings, opponent: dict[str, Any]) -> dict[str, Any]:
    deck_path = str(opponent.get("opponent_deck_path") or "").strip()
    if deck_path:
        return load_deck_payload(Path(deck_path), env)

    deck_name = str(opponent.get("opponent_deck") or opponent.get("label") or "").strip()
    if not deck_name:
        raise ValueError("Opponent deck is not configured")

    cache_dir = TUI_DECK_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"precon_{slugify(deck_name)}.json"
    if not out.is_file():
        export_precon_deck_json(deck_name, Path(env.assets_path), out, game_format="sage")
    return load_deck_payload(out, env)


def opponent_from_precon(deck_name: str, env: EnvironmentSettings) -> dict[str, Any]:
    opp = {"opponent_deck": deck_name, "source": "precon", "label": deck_name}
    payload = _load_opponent_pool(env, opp)
    hero_id = str(payload.get("hero_id") or "").strip()
    if not hero_id:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from flesh_and_blood_rlbridge.opponent_deck import read_talishar_asset_hero_info

        info = read_talishar_asset_hero_info(env.assets_path, deck_name)
        hero_id = info.hero_id if info else deck_name.split("SAGE")[0].lower()
    equipment_header = str(payload.get("equipment_header") or hero_id).strip()
    return {
        "opponent_hero_id": hero_id,
        "opponent_deck": deck_name,
        "source": "precon",
        "label": deck_name,
        "equipment_header": equipment_header,
        "hero_class": str(payload.get("hero_class") or ""),
        "game_format": str(payload.get("game_format") or "silver_age"),
    }


def opponent_from_fabrary(url_or_slug: str, env: EnvironmentSettings) -> dict[str, str]:
    """Import a FaBrary deck and register it as a Talishar Assets opponent list."""
    deck_path = import_fabrary(url_or_slug)
    payload = load_deck_payload(deck_path, env)
    deck = payload.get("deck") or {}
    if not deck:
        raise ValueError("FaBrary deck has no main-deck cards")

    hero_id = str(payload.get("hero_id") or "").strip()
    equipment_header = str(payload.get("equipment_header") or hero_id).strip()
    if not hero_id:
        raise ValueError("Could not determine opponent hero from FaBrary deck")

    training_root = REPO_ROOT / "scripts" / "training"
    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))
    from train_pipeline_common import _write_deck_file  # noqa: PLC0415

    stem = f"rl_gui_{uuid.uuid4().hex[:10]}"
    _write_deck_file(
        {str(k): int(v) for k, v in deck.items() if int(v) > 0},
        equipment_header,
        stem,
        env.assets_path,
    )
    label = str(payload.get("name") or url_or_slug).strip()
    return {
        "opponent_hero_id": hero_id,
        "opponent_deck": stem,
        "opponent_deck_path": str(deck_path.resolve()),
        "source": "fabrary",
        "label": label,
        "equipment_header": equipment_header,
        "hero_class": str(payload.get("hero_class") or ""),
        "game_format": str(payload.get("game_format") or "silver_age"),
    }


def search_cards(
    query: str,
    *,
    game_format: str,
    talishar_url: str,
    limit: int = 24,
) -> list[dict[str, Any]]:
    index = CardSearchIndex(game_format)
    return [card_hit_to_dict(hit, talishar_url=talishar_url) for hit in index.search(query, limit=limit)]


def search_equipment(
    query: str,
    *,
    game_format: str,
    hero_id: str,
    hero_class: str = "",
    talishar_url: str,
    slot: str | None = None,
    limit: int = 24,
) -> list[dict[str, Any]]:
    index = EquipmentSearchIndex(game_format, hero_id=hero_id, hero_class=hero_class)
    return [
        card_hit_to_dict(hit, talishar_url=talishar_url)
        for hit in index.search(query, slot=slot, limit=limit)
    ]


def equipment_alternatives_for_slot(
    *,
    game_format: str,
    hero_id: str,
    hero_class: str = "",
    talishar_url: str,
    slot: str,
) -> list[dict[str, Any]]:
    """All legal equipment options for a loadout slot."""
    from fab_tui.equipment import equipment_slot

    index = EquipmentSearchIndex(game_format, hero_id=hero_id, hero_class=hero_class)
    slot_key = str(slot or "").strip().lower()
    hits = [
        hit
        for hit in index.all_hits()
        if equipment_slot(hit.card_id, hero_id=hero_id) == slot_key
    ]
    return [card_hit_to_dict(hit, talishar_url=talishar_url) for hit in hits]


def replace_equipment_slot(
    equipment_header: str,
    *,
    slot_index: int,
    replacement_card_id: str,
    hero_id: str,
    hero_class: str = "",
    game_format: str,
) -> str | None:
    equip_index = EquipmentSearchIndex(game_format, hero_id=hero_id, hero_class=hero_class)
    entries = parse_equipment_header(
        equipment_header,
        hero_id=hero_id,
        display_name=equip_index.display_name,
    )
    entry = next((row for row in entries if row.index == slot_index), None)
    if entry is None or entry.slot == "hero":
        return None
    parts = equipment_header.split()
    if not (1 <= slot_index <= len(parts)):
        return None
    parts[slot_index - 1] = replacement_card_id
    return " ".join(parts)


def equipment_loadout(
    equipment_header: str,
    *,
    hero_id: str,
    hero_class: str = "",
    game_format: str,
    talishar_url: str,
) -> list[dict[str, Any]]:
    equip_index = EquipmentSearchIndex(game_format, hero_id=hero_id, hero_class=hero_class)
    entries = parse_equipment_header(
        equipment_header,
        hero_id=hero_id,
        display_name=equip_index.display_name,
    )
    return [
        {
            "index": entry.index,
            "slot": entry.slot,
            "slot_label": slot_display_name(entry.slot),
            "card_id": entry.card_id,
            "name": entry.label,
            "classification": (
                hit.classification if (hit := equip_index.lookup(entry.card_id)) else ""
            ),
            "image_url": card_image_display_url(entry.card_id),
        }
        for entry in entries
    ]


def compute_guide_baseline(
    *,
    card_pool: dict[str, int],
    opponent_hero_id: str,
    hero_id: str,
    hero_class: str,
    game_format: str,
    equipment_header: str,
) -> dict[str, Any]:
    guide_deck = compute_guide_policy_deck(
        card_pool,
        opponent_hero_id=opponent_hero_id,
        hero_id=hero_id,
        game_format=game_format,
        hero_class=hero_class,
    )
    return {
        "baseline_deck": guide_deck,
        "equipment_header": equipment_header,
        "baseline_label": "Guide policy baseline",
    }


def apply_opponent_guide_sideboard(
    env: EnvironmentSettings,
    opponent: dict[str, Any],
    *,
    player_hero_id: str,
) -> dict[str, Any]:
    """Sideboard the opponent list vs the player hero and sync Talishar Assets."""
    opp_payload = _load_opponent_pool(env, opponent)
    guide_deck = compute_guide_policy_deck(
        {str(k): int(v) for k, v in (opp_payload.get("card_pool") or {}).items()},
        opponent_hero_id=player_hero_id,
        hero_id=str(opp_payload.get("hero_id") or opponent.get("opponent_hero_id") or ""),
        game_format=str(opp_payload.get("game_format") or "silver_age"),
        hero_class=str(opp_payload.get("hero_class") or ""),
    )
    asset_stem = str(opponent.get("opponent_deck") or "").strip()
    if not asset_stem:
        raise ValueError("Opponent Talishar asset name is missing")

    training_root = REPO_ROOT / "scripts" / "training"
    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))
    from train_pipeline_common import _write_deck_file  # noqa: PLC0415

    equipment_header = str(
        opp_payload.get("equipment_header") or opponent.get("opponent_hero_id") or ""
    ).strip()
    _write_deck_file(
        {str(k): int(v) for k, v in guide_deck.items() if int(v) > 0},
        equipment_header,
        asset_stem,
        env.assets_path,
    )
    game_format = str(opp_payload.get("game_format") or "silver_age")
    deck_entries = deck_counts_to_entries(
        guide_deck,
        game_format=game_format,
        talishar_url=env.talishar_url,
    )
    return {
        "deck": guide_deck,
        "deck_entries": deck_entries,
        "deck_size": sum(int(v) for v in guide_deck.values()),
        "baseline_label": "Guide policy sideboard",
        "equipment_header": equipment_header,
        "hero_id": opp_payload.get("hero_id"),
        "hero_class": opp_payload.get("hero_class") or "",
        "game_format": game_format,
    }


def try_swap(
    game_deck: dict[str, int],
    card_pool: dict[str, int],
    out_card: str,
    in_card: str,
) -> dict[str, int] | None:
    result = apply_manual_swap(game_deck, card_pool, out_card, in_card)
    if result is None:
        return None
    deck, pool = result
    return {"deck": deck, "card_pool": pool}


@dataclass
class TrainingRun:
    run_id: str
    out_dir: Path
    status: str = "pending"
    exit_code: int | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    _thread: threading.Thread | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "out_dir": str(self.out_dir),
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "dashboard_url": f"/api/runs/{self.run_id}/dashboard",
            "results_url": f"/api/runs/{self.run_id}/results",
        }


class RunRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, TrainingRun] = {}
        self._lock = threading.Lock()
        self._replay_jobs: dict[str, dict[str, Any]] = {}

    def get(self, run_id: str) -> TrainingRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def add(self, run: TrainingRun) -> None:
        with self._lock:
            self._runs[run.run_id] = run

    def set_replay_job(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._replay_jobs[run_id] = payload

    def get_replay_job(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._replay_jobs.get(run_id)


RUNS = RunRegistry()


def resolve_run_out_dir(run_id: str) -> Path | None:
    return _resolve_run_dir(run_id)


def replay_gif_path(out_dir: Path) -> Path | None:
    direct = out_dir / "winner_replay.gif"
    if direct.is_file():
        return direct

    summary_path = out_dir / "sideboard_compare_results.json"
    if summary_path.is_file():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            raw = str(data.get("replay_gif") or (data.get("winner") or {}).get("replay_gif") or "")
            if raw:
                path = Path(raw)
                if path.is_file():
                    return path
            winner_id = str((data.get("winner") or {}).get("candidate_id") or "")
            if winner_id:
                eval_gif = (
                    out_dir / "candidates" / winner_id / "final_eval" / "p1_optimal_policy.gif"
                )
                if eval_gif.is_file():
                    return eval_gif
        except (OSError, json.JSONDecodeError):
            pass
    return None


def replay_render_status(run_id: str) -> dict[str, Any] | None:
    out_dir = _resolve_run_dir(run_id)
    if out_dir is None:
        return None
    job = RUNS.get_replay_job(run_id)
    gif = replay_gif_path(out_dir)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "ready": gif is not None,
        "replay_gif_url": f"/api/runs/{run_id}/replay.gif" if gif is not None else None,
    }
    if job:
        payload.update(job)
    elif gif is not None:
        payload["status"] = "completed"
    else:
        payload["status"] = "missing"
    return payload


def start_replay_render(run_id: str, env: EnvironmentSettings) -> dict[str, Any]:
    out_dir = _resolve_run_dir(run_id)
    if out_dir is None:
        raise ValueError("Run not found")
    existing = replay_gif_path(out_dir)
    if existing is not None:
        return replay_render_status(run_id) or {"run_id": run_id, "status": "completed"}

    job = RUNS.get_replay_job(run_id)
    if job and job.get("status") == "running":
        return job

    RUNS.set_replay_job(run_id, {"run_id": run_id, "status": "running"})

    def _worker() -> None:
        try:
            training_root = REPO_ROOT / "scripts" / "training"
            if str(training_root) not in sys.path:
                sys.path.insert(0, str(training_root))
            from train_sideboard_compare import render_winner_replay_gif_for_run  # noqa: PLC0415

            result = render_winner_replay_gif_for_run(
                out_dir,
                talishar_url=env.talishar_url,
                talishar_fe_url=env.talishar_fe_url,
                assets_path=env.assets_path,
            )
            status = "completed" if result.get("gif") else "failed"
            RUNS.set_replay_job(
                run_id,
                {
                    "run_id": run_id,
                    "status": status,
                    "error": result.get("error"),
                    "outcome": result.get("outcome"),
                },
            )
        except Exception as exc:  # noqa: BLE001
            RUNS.set_replay_job(
                run_id,
                {"run_id": run_id, "status": "failed", "error": str(exc)},
            )

    thread = threading.Thread(target=_worker, name=f"replay-render-{run_id}", daemon=True)
    thread.start()
    return {"run_id": run_id, "status": "running"}


def start_training(
    *,
    env: EnvironmentSettings,
    starting_deck_path: Path,
    opponent_hero_id: str,
    opponent_deck: str,
    baseline_deck: dict[str, int],
    card_pool: dict[str, int],
    equipment_header: str,
    baseline_label: str,
    variants: list[dict[str, Any]],
    spec_kwargs: dict[str, Any],
) -> TrainingRun:
    player_info = read_deck_hero_info(starting_deck_path)
    game_format = read_deck_format(starting_deck_path)
    render_replay_gif = bool(spec_kwargs.pop("render_replay_gif", True))
    spec_kwargs.pop("no_render_gif", None)
    spec = SideboardCompareSpec(
        starting_deck=str(starting_deck_path),
        opponent_hero_id=opponent_hero_id,
        opponent_deck=opponent_deck,
        hero_id=player_info.hero_id if player_info else "",
        hero_class=player_info.hero_class if player_info else "",
        equipment_header=equipment_header,
        game_format=game_format,  # type: ignore[arg-type]
        no_render_gif=True,
        **spec_kwargs,
    )
    out_dir = spec.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_variants: list[ManualSwapVariant] = []
    for index, row in enumerate(variants, start=1):
        swaps = [tuple(pair) for pair in row.get("swaps") or []]
        manual_variants.append(
            ManualSwapVariant(
                candidate_id=str(row.get("candidate_id") or f"manual_{index:02d}"),
                label=str(row.get("label") or f"Manual variant {index}"),
                game_deck={str(k): int(v) for k, v in (row.get("game_deck") or {}).items()},
                swaps=swaps,  # type: ignore[arg-type]
            )
        )

    candidates_path = write_candidates_manifest(
        out_dir / "candidates_manifest.json",
        baseline_deck=baseline_deck,
        card_pool=card_pool,
        variants=manual_variants,
        baseline_label=baseline_label,
        equipment_header=equipment_header,
    )
    spec.candidates_json = str(candidates_path)
    spec.num_options = 1 + len(manual_variants)
    if equipment_header:
        spec.equipment_header = equipment_header

    run_id = uuid.uuid4().hex[:12]
    run = TrainingRun(run_id=run_id, out_dir=out_dir)
    RUNS.add(run)

    def _worker() -> None:
        run.status = "running"
        try:
            rc = run_sideboard_compare(
                spec,
                env,
                starting_deck=starting_deck_path,
                candidates_json=candidates_path,
            )
            run.exit_code = rc
            run.status = "completed" if rc == 0 else "failed"
            if rc == 0 and render_replay_gif:
                start_replay_render(run.run_id, env)
        except Exception:
            run.exit_code = 1
            run.status = "failed"
            raise
        finally:
            run.finished_at = datetime.now(timezone.utc).isoformat()
            _write_dashboard_once(out_dir)

    thread = threading.Thread(target=_worker, name=f"sideboard-train-{run_id}", daemon=True)
    run._thread = thread
    thread.start()
    return run


def run_status(run_id: str) -> dict[str, Any] | None:
    run = RUNS.get(run_id)
    if run is None:
        out_dir = _resolve_run_dir(run_id)
        if out_dir is None:
            return None
        return _status_from_disk(out_dir, run_id)

    payload = run.to_dict()
    payload.update(_status_from_disk(run.out_dir, run_id))
    return payload


def _resolve_run_dir(run_id: str) -> Path | None:
    root = REPO_ROOT / "results" / "sideboard_compare"
    if not root.is_dir():
        return None
    for path in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        marker = path / "candidates_manifest.json"
        if marker.is_file():
            try:
                data = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(data.get("gui_run_id") or "") == run_id:
                return path
    return None


def _status_from_disk(out_dir: Path, run_id: str) -> dict[str, Any]:
    summary_path = out_dir / "sideboard_compare_results.json"
    manifest_path = out_dir / "candidates_manifest.json"
    complete = summary_path.is_file()
    candidates: list[dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidates = manifest.get("candidates") or []
        except (OSError, json.JSONDecodeError):
            pass

    candidate_status: list[dict[str, Any]] = []
    for row in candidates:
        cid = str(row.get("candidate_id") or "")
        cand_dir = out_dir / "candidates" / cid
        result_path = cand_dir / "candidate_result.json"
        progress = 0.0
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                progress = float(result.get("progress_pct") or 100.0)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                progress = 100.0 if complete else 0.0
        candidate_status.append({"candidate_id": cid, "label": row.get("label"), "progress_pct": progress})

    return {
        "complete": complete,
        "candidate_status": candidate_status,
        "dashboard_ready": (out_dir / "sideboard_compare_dashboard.html").is_file(),
    }


def run_results(run_id: str) -> dict[str, Any] | None:
    run = RUNS.get(run_id)
    out_dir = run.out_dir if run else _resolve_run_dir(run_id)
    if out_dir is None:
        return None
    summary_path = out_dir / "sideboard_compare_results.json"
    if not summary_path.is_file():
        return {"complete": False, "out_dir": str(out_dir)}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"complete": False, "out_dir": str(out_dir)}
    data["complete"] = True
    data["out_dir"] = str(out_dir)
    data["run_id"] = run_id
    gif = replay_gif_path(out_dir)
    if gif is not None:
        data["replay_gif"] = str(gif)
        data["replay_gif_url"] = f"/api/runs/{run_id}/replay.gif"
    replay_job = RUNS.get_replay_job(run_id)
    if replay_job:
        data["replay_render_status"] = replay_job
    elif data.get("replay_render"):
        data["replay_render_status"] = data["replay_render"]
    return data


def dashboard_path(run_id: str) -> Path | None:
    run = RUNS.get(run_id)
    out_dir = run.out_dir if run else _resolve_run_dir(run_id)
    if out_dir is None:
        return None
    html_path = out_dir / "sideboard_compare_dashboard.html"
    if html_path.is_file():
        return html_path
    _write_dashboard_once(out_dir)
    if html_path.is_file():
        return html_path
    try:
        from sideboard_compare_dashboard import write_sideboard_compare_dashboard  # noqa: PLC0415

        return write_sideboard_compare_dashboard(out_dir, auto_refresh_seconds=5.0)
    except Exception:
        return None


def save_deck_api(
    *,
    baseline_deck: dict[str, int],
    card_pool: dict[str, int],
    equipment_header: str,
    hero_id: str,
    hero_class: str,
    game_format: str,
    label: str,
    opponent_hero_id: str,
    baseline_label: str,
) -> str:
    path = save_user_deck(
        baseline_deck=baseline_deck,
        card_pool=card_pool,
        equipment_header=equipment_header,
        hero_id=hero_id,
        hero_class=hero_class,
        game_format=game_format,
        label=label,
        opponent_hero_id=opponent_hero_id,
        baseline_label=baseline_label,
    )
    return str(path)


def persist_session_deck(payload: dict[str, Any]) -> Path:
    """Write an in-progress deck edit to the GUI session cache."""
    session_dir = TUI_DECK_CACHE / "gui_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    deck_id = str(payload.get("session_id") or uuid.uuid4().hex[:12])
    path = session_dir / f"{deck_id}.json"
    body = {
        "name": payload.get("name") or "GUI session deck",
        "hero_id": payload.get("hero_id") or "",
        "hero_class": payload.get("hero_class") or "",
        "format": payload.get("game_format") or "silver_age",
        "equipment_header": payload.get("equipment_header") or "",
        "deck": {str(k): int(v) for k, v in (payload.get("deck") or {}).items()},
        "sideboard": _sideboard_from_pool(payload.get("card_pool") or {}, payload.get("deck") or {}),
        "gui_session_id": deck_id,
    }
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def _sideboard_from_pool(card_pool: dict[str, int], game_deck: dict[str, int]) -> dict[str, int]:
    sideboard: dict[str, int] = {}
    for card_id in set(card_pool) | set(game_deck):
        remaining = int(card_pool.get(card_id, 0)) - int(game_deck.get(card_id, 0))
        if remaining > 0:
            sideboard[card_id] = remaining
    return sideboard
