"""Resolve Talishar ``Assets/<stem>.txt`` deck names from hero slugs or precon labels."""

from __future__ import annotations

from pathlib import Path

# SAGE precon decks (hero slug → Assets file stem)
SAGE_PRECON_BY_HERO: dict[str, str] = {
    "aurora": "AuroraSAGEPrecon",
    "briar": "BriarSAGEPrecon",
    "dorinthea": "DorintheSAGEPrecon",
    "kayo": "KayoSAGEPrecon",
    "viserai": "ViseraiSAGEPrecon",
    "iyslander": "IyslanderSAGEPrecon",
    "dash": "DashSAGEPrecon",
    "fai": "FaiSAGEPrecon",
    "azalea": "AzaleaSAGEPrecon",
    "boltyn": "BoltynSAGEPrecon",
    "enigma": "EnigmaSAGEPrecon",
    "ira": "Ira",
}


def _asset_exists(assets_dir: Path, stem: str) -> bool:
    return bool(stem) and (assets_dir / f"{stem}.txt").is_file()


def resolve_talishar_deck_stem(assets_dir: str | Path, name: str) -> str:
    """Return an ``Assets`` file stem that exists for *name*.

    *name* may be a full asset stem (``BriarSAGEPrecon``,
    ``fab_precon_sage_ch1_kayo``), a hero slug (``briar``, ``dash``), or a
    title-cased hero token (``Briar``, ``Dash``).
    """
    assets = Path(assets_dir)
    raw = (name or "").strip()
    if not raw:
        return raw
    if _asset_exists(assets, raw):
        return raw

    token = raw.replace("-", "_")
    hero = token.split("_")[0].lower()
    title = hero[:1].upper() + hero[1:] if hero else raw

    candidates: list[str] = []

    def add(stem: str) -> None:
        if stem and stem not in candidates:
            candidates.append(stem)

    add(token)
    add(title)
    if hero in SAGE_PRECON_BY_HERO:
        add(SAGE_PRECON_BY_HERO[hero])
    add(f"{title}SAGEPrecon")

    if assets.is_dir():
        for pattern in (
            f"fab_precon_*_{hero}.txt",
            f"fab_precon_*_{token}.txt",
            f"*{title}*SAGEPrecon.txt",
            f"*{hero}*SAGEPrecon.txt",
        ):
            for path in sorted(assets.glob(pattern)):
                add(path.stem)

    for stem in candidates:
        if _asset_exists(assets, stem):
            return stem

    return raw
