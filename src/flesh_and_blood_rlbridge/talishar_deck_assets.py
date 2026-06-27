"""Resolve Talishar ``Assets/<stem>.txt`` deck names from hero slugs or precon labels."""

from __future__ import annotations

from pathlib import Path

# SAGE precon decks (hero slug → Assets file stem)
SAGE_PRECON_BY_HERO: dict[str, str] = {
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
    "arakni_web_of_deceit": "ArakniWebOfDeceitSAGEPrecon",
    "gravy": "GravyBonesSAGEPrecon",
    "gravy_bones": "GravyBonesSAGEPrecon",
    "lyath": "LyathGoldmaneSAGEPrecon",
    "lyath_goldmane": "LyathGoldmaneSAGEPrecon",
    "blaze": "BlazeSAGEPrecon",
    "blaze_firemind": "BlazeSAGEPrecon",
    "ira": "Ira",
}


def build_assets_equipment_headers(assets_dir: str | Path) -> dict[str, str]:
    """Map hero id -> fullest equipment header line found in ``Assets/*.txt``."""
    assets = Path(assets_dir)
    result: dict[str, str] = {}
    if not assets.is_dir():
        return result
    for txt_file in sorted(assets.glob("*.txt")):
        try:
            lines = txt_file.read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            first_line = lines[0].strip()
            if not first_line:
                continue
            hero_id = first_line.split()[0]
            prev = result.get(hero_id, "")
            if len(first_line.split()) > len(prev.split()):
                result[hero_id] = first_line
        except (OSError, IndexError):
            continue
    return result


def resolve_equipment_header_line(
    hero_id: str,
    assets_dir: str | Path,
    *,
    fallback: str = "",
) -> str:
    """Return the best Talishar equipment header line for *hero_id*."""
    token = hero_id.removeprefix("hero_").replace("-", "_").strip()
    headers = build_assets_equipment_headers(assets_dir)
    best = (fallback or "").strip()
    for key in (token, hero_id):
        line = headers.get(key, "").strip()
        if len(line.split()) > len(best.split()):
            best = line
    if best:
        return best
    return token or hero_id


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
    if token in SAGE_PRECON_BY_HERO:
        add(SAGE_PRECON_BY_HERO[token])
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
