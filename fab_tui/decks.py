"""Deck discovery, precon export, and FaBrary resolution for the TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fab_tui.config import REPO_ROOT, RESULTS_ROOT, slugify

# SAGE precon decks (hero slug, Talishar Assets deck name)
SAGE_PRECONS: list[tuple[str, str]] = [
    ("aurora", "AuroraSAGEPrecon"),
    ("briar", "BriarSAGEPrecon"),
    ("dorinthea", "DorintheSAGEPrecon"),
    ("kayo", "KayoSAGEPrecon"),
    ("viserai", "ViseraiSAGEPrecon"),
    ("iyslander", "IyslanderSAGEPrecon"),
    ("dash", "DashSAGEPrecon"),
    ("fai", "FaiSAGEPrecon"),
    ("azalea", "AzaleaSAGEPrecon"),
    ("boltyn", "BoltynSAGEPrecon"),
    ("enigma", "EnigmaSAGEPrecon"),
]

EXTRA_PRECONS: list[tuple[str, str]] = [
    ("ira", "Ira"),
]

HERO_CLASSES: dict[str, str] = {
    "briar": "Elementalist",
    "dorinthea": "Warrior",
    "dorinthea_ironsong": "Warrior",
    "kayo": "Brute",
    "viserai": "Runeblade",
    "iyslander": "Elementalist",
    "dash": "Mechanologist",
    "fai": "Ninja",
    "azalea": "Ranger",
    "boltyn": "Light",
    "enigma": "Illusionist",
    "ira": "Ninja",
    "aurora": "Runeblade",
    "briar_whetstone": "Elementalist",
}

DECK_CACHE = RESULTS_ROOT / "tui_decks"


@dataclass(frozen=True)
class DeckOption:
    label: str
    path: Path
    fmt: str = "silver_age"


def _hero_token(hero_id: str) -> str:
    return hero_id.replace("-", "_").split("_")[0].lower()


def hero_class_for(hero_id: str) -> str:
    token = _hero_token(hero_id)
    if hero_id in HERO_CLASSES:
        return HERO_CLASSES[hero_id]
    return HERO_CLASSES.get(token, "Warrior")


def resolve_deck_link(
    source: str,
    *,
    label: str,
    cache_dir: Path,
    fetch_fn,
) -> Path | None:
    """Resolve a FaBrary URL/slug or local JSON path to a deck file."""
    raw = source.strip()
    if not raw:
        return None
    local = Path(raw)
    if local.is_file():
        return local.resolve()
    slug = raw.rsplit("/", 1)[-1][:26]
    out = cache_dir / f"{slugify(label)}_{slug.lower()}.json"
    if fetch_fn(raw, out) != 0:
        return None
    return out


def list_precon_options(assets_path: Path) -> list[tuple[str, str]]:
    """Return ``(label, deck_asset_name)`` for available Talishar precons."""
    options: list[tuple[str, str]] = []
    for hero, deck_name in [*SAGE_PRECONS, *EXTRA_PRECONS]:
        if (assets_path / f"{deck_name}.txt").is_file():
            options.append((f"{hero.capitalize()} ({deck_name})", deck_name))
    return options


def export_precon_deck_json(
    deck_name: str,
    assets_path: Path,
    out_path: Path,
    *,
    game_format: str = "sage",
) -> Path:
    """Convert a Talishar ``Assets/<deck>.txt`` file to rlbridge deck JSON."""
    asset_file = assets_path / f"{deck_name}.txt"
    if not asset_file.is_file():
        raise FileNotFoundError(f"Precon deck not found: {asset_file}")

    lines = [
        line.strip()
        for line in asset_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    setup = lines[0].split() if lines else []
    hero_id = setup[0] if setup else deck_name
    equipment_header = " ".join(setup)
    resolver = None
    try:
        from flesh_and_blood_rlbridge.card_db.talishar_card_ids import (  # noqa: PLC0415
            TalisharCardIdResolver,
        )

        php = assets_path.parent / "GeneratedCode" / "GeneratedCardDictionaries.php"
        cards_path = REPO_ROOT / "src" / "flesh_and_blood_rlbridge" / "card_db" / "cards.json"
        resolver = TalisharCardIdResolver(talishar_php_path=php, cards_path=cards_path)
    except ImportError:
        resolver = None

    deck: dict[str, int] = {}
    for card in " ".join(lines[1:]).split():
        card_id = card.strip()
        if not card_id:
            continue
        if resolver is not None:
            card_id = resolver.resolve(card_id) or card_id
        deck[card_id] = deck.get(card_id, 0) + 1

    sideboard: dict[str, int] = {}
    fmt_norm = str(game_format or "sage").lower()
    min_game = 40 if fmt_norm in {"sage", "silver_age", "blitz"} else 60
    total = sum(deck.values())
    if total > min_game:
        game_deck: dict[str, int] = {}
        remaining = min_game
        for card_id, count in deck.items():
            if remaining <= 0:
                sideboard[card_id] = count
                continue
            take = min(count, remaining)
            if take:
                game_deck[card_id] = take
            leftover = count - take
            if leftover:
                sideboard[card_id] = sideboard.get(card_id, 0) + leftover
            remaining -= take
        deck = game_deck

    payload = {
        "name": deck_name,
        "hero_id": hero_id,
        "hero_class": hero_class_for(hero_id),
        "format": game_format,
        "equipment_header": equipment_header,
        "deck": deck,
        "sideboard": sideboard,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _checkpoint_to_json(meta_path: Path, cache_dir: Path) -> Path | None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("checkpoint_type") != "phase3_play":
        return None
    deck_spec = meta.get("deck_spec") or {}
    cards = deck_spec.get("cards") or {}
    if not cards:
        return None

    role = str(meta.get("role") or meta_path.parent.name)
    if role not in {"p1", "p2"}:
        role = "p1"
    hero_id = str(meta.get(f"{role}_hero") or meta.get("p1_hero") or "")
    equipment_header = str(deck_spec.get("equipment_header") or hero_id)
    if hero_id and not equipment_header.startswith(hero_id.replace("-", "_")):
        equipment_header = f"{hero_id.replace('-', '_')} {equipment_header}".strip()

    matchup = str(meta.get("matchup") or "run")
    episode = meta_path.parent.name
    out_path = cache_dir / f"checkpoint_{slugify(matchup)}_{role}_{episode}.json"
    if out_path.is_file():
        return out_path

    payload = {
        "name": f"{matchup} {role.upper()} {episode}",
        "hero_id": hero_id.replace("-", "_") or equipment_header.split()[0],
        "hero_class": hero_class_for(hero_id or equipment_header.split()[0]),
        "format": str(meta.get("game_format") or "silver_age"),
        "equipment_header": equipment_header,
        "deck": {str(k): int(v) for k, v in cards.items()},
        "sideboard": {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def discover_saved_decks() -> list[DeckOption]:
    """Find deck JSON from prior experiments, matchup fetches, and play checkpoints."""
    seen: set[Path] = set()
    options: list[DeckOption] = []

    def add(label: str, path: Path, fmt: str = "silver_age") -> None:
        resolved = path.resolve()
        if not resolved.is_file() or resolved in seen:
            return
        seen.add(resolved)
        options.append(DeckOption(label=label, path=resolved, fmt=fmt))

    for path in sorted((RESULTS_ROOT / "experiments").glob("*/decks/*.json")):
        exp = path.parent.parent.name
        add(f"[draft] {exp} / {path.name}", path)

    matchup_decks = RESULTS_ROOT / "matchup_sims" / "decks"
    if matchup_decks.is_dir():
        for path in sorted(matchup_decks.glob("*.json")):
            add(f"[fabrary cache] {path.stem}", path)

    for meta_path in sorted(RESULTS_ROOT.glob("**/metadata.json")):
        exported = _checkpoint_to_json(meta_path, DECK_CACHE)
        if exported is not None:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            matchup = str(meta.get("matchup") or "checkpoint")
            role = str(meta.get("role") or "p1")
            episode = meta_path.parent.name
            add(f"[checkpoint] {matchup} / {role} / {episode}", exported)

    saved_dir = DECK_CACHE / "saved"
    if saved_dir.is_dir():
        for path in sorted(saved_dir.glob("*.json")):
            if path.name == "index.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data.get("saved_meta"), dict):
                continue
            label = str(data.get("name") or data["saved_meta"].get("label") or path.stem)
            fmt = str(data.get("format") or "silver_age")
            add(f"[saved] {label}", path, fmt)

    options.sort(key=lambda item: item.label.lower())
    return options


def read_deck_format(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("format") or "silver_age")
    except (OSError, json.JSONDecodeError):
        return "silver_age"


@dataclass(frozen=True)
class DeckHeroInfo:
    hero_id: str
    hero_class: str
    equipment_header: str
    game_format: str
    name: str


def read_deck_hero_info(path: Path) -> DeckHeroInfo | None:
    """Read hero / equipment / format fields from an rlbridge deck JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    hero_id = str(data.get("hero_id") or "").strip()
    if not hero_id and data.get("equipment_header"):
        hero_id = str(data["equipment_header"]).split()[0]
    return DeckHeroInfo(
        hero_id=hero_id,
        hero_class=str(data.get("hero_class") or "").strip(),
        equipment_header=str(data.get("equipment_header") or "").strip(),
        game_format=str(data.get("format") or "").strip(),
        name=str(data.get("name") or hero_id or path.stem),
    )


def apply_player_from_deck(spec: "ExperimentSpec", deck_path: str | Path, *, player: str) -> bool:
    """Copy hero fields from a deck JSON onto P1 or P2. Returns True if applied."""
    from fab_tui.config import ExperimentSpec  # noqa: PLC0415

    info = read_deck_hero_info(Path(deck_path))
    if info is None or not info.hero_id:
        return False

    if player == "p1":
        spec.hero_id = info.hero_id
        if info.hero_class:
            spec.hero_class = info.hero_class
        if info.equipment_header:
            spec.equipment_header = info.equipment_header
        if info.game_format:
            spec.game_format = info.game_format  # type: ignore[assignment]
    elif player == "p2":
        spec.p2_hero_id = info.hero_id
        if info.hero_class:
            spec.p2_hero_class = info.hero_class
        if info.equipment_header:
            spec.p2_equipment_header = info.equipment_header
        if info.game_format and spec.opponent_mode == "dual":
            spec.game_format = info.game_format  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown player role: {player}")
    return True


def apply_mirror_opponent_from_p1(spec: "ExperimentSpec") -> None:
    """Mirror mode — opponent uses the same hero / equipment as P1."""
    from fab_tui.config import ExperimentSpec  # noqa: PLC0415

    spec.opponent_hero_id = spec.hero_id
    spec.p2_hero_id = spec.hero_id
    spec.p2_hero_class = spec.hero_class
    spec.p2_equipment_header = spec.equipment_header


def apply_opponent_from_asset(
    spec: "ExperimentSpec",
    assets_path: str | Path,
) -> bool:
    """Set opponent / P2 hero fields from a Talishar Assets deck file."""
    from fab_tui.config import ExperimentSpec  # noqa: PLC0415
    from flesh_and_blood_rlbridge.opponent_deck import read_talishar_asset_hero_info

    info = read_talishar_asset_hero_info(assets_path, spec.opponent_deck)
    if info is None:
        return False
    spec.opponent_deck = info.asset_stem
    spec.opponent_hero_id = info.hero_id
    spec.p2_hero_id = info.hero_id
    spec.p2_hero_class = info.hero_class
    spec.p2_equipment_header = info.equipment_header
    return True


def apply_heroes_from_decks(spec: "ExperimentSpec") -> None:
    """Populate hero settings from any attached deck JSON paths on the spec."""
    from fab_tui.config import ExperimentSpec  # noqa: PLC0415

    p1_path = spec.p1_fixed_deck or spec.p1_starting_deck
    if p1_path:
        apply_player_from_deck(spec, p1_path, player="p1")

    p2_path = spec.p2_fixed_deck or spec.p2_starting_deck
    if p2_path and spec.opponent_mode == "dual":
        apply_player_from_deck(spec, p2_path, player="p2")
    elif spec.opponent_mode != "preset" and p2_path:
        apply_player_from_deck(spec, p2_path, player="p2")
