"""Episode-level deck / hero context for player-fair observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .card_db.talishar_card_ids import TalisharCardIdResolver
from .card_vocab import card_index
from .talishar_deck_assets import resolve_talishar_deck_stem


@dataclass
class EpisodeContext:
    self_hero_id: str = ""
    opp_hero_id: str = ""
    format: str = "silver_age"
    self_deck_counts: dict[str, int] = field(default_factory=dict)
    first_player: int = 1

    def self_deck_indices(self) -> list[int]:
        """Expand deck multiset into up to 80 card indices (player-fair: own deck only)."""
        ordered: list[int] = []
        for cid in sorted(self.self_deck_counts.keys()):
            count = int(self.self_deck_counts[cid])
            idx = card_index(cid)
            if idx <= 0:
                continue
            ordered.extend([idx] * count)
        return ordered[:80]


def _read_asset_deck(
    assets_dir: Path,
    deck_name: str,
    *,
    talishar_php: Optional[Path] = None,
) -> tuple[str, dict[str, int]]:
    stem = resolve_talishar_deck_stem(assets_dir, deck_name)
    asset_file = assets_dir / f"{stem}.txt"
    if not asset_file.is_file():
        return "", {}
    lines = [
        line.strip()
        for line in asset_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    setup_cards = lines[0].split() if lines else []
    deck_cards = "\n".join(lines[1:]).split()
    resolver = TalisharCardIdResolver(
        talishar_php_path=talishar_php,
    )
    counts: dict[str, int] = {}
    for name in deck_cards:
        card_name = str(name or "").strip()
        if card_name:
            card_name = resolver.resolve(card_name) or card_name
            counts[card_name] = counts.get(card_name, 0) + 1
    hero_id = setup_cards[0] if setup_cards else ""
    return hero_id, counts


def load_episode_context(
    *,
    self_deck_name: str,
    opponent_deck_name: str,
    game_format: str = "silver_age",
    talishar_src: Optional[Path] = None,
    first_player: int = 1,
) -> EpisodeContext:
    """Load player-fair episode context from Talishar Assets deck files."""
    root = Path(__file__).resolve().parents[2]
    src = talishar_src or (root / "Talishar")
    assets_dir = src / "Assets"
    php = src / "GeneratedCode" / "GeneratedCardDictionaries.php"

    self_hero, self_counts = _read_asset_deck(
        assets_dir, self_deck_name, talishar_php=php if php.is_file() else None
    )
    opp_hero, _ = _read_asset_deck(
        assets_dir, opponent_deck_name, talishar_php=php if php.is_file() else None
    )
    return EpisodeContext(
        self_hero_id=self_hero,
        opp_hero_id=opp_hero,
        format=game_format,
        self_deck_counts=self_counts,
        first_player=2 if int(first_player) == 2 else 1,
    )


def hero_from_equipment(equipment: list) -> str:
    if not equipment:
        return ""
    first = equipment[0]
    if isinstance(first, dict):
        return str(first.get("cardNumber") or first.get("cardID") or "")
    return ""
