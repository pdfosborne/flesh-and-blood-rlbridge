"""Tests for unified random matchup checkpoint evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "eval"))
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from eval_phase3_checkpoint import (  # noqa: E402
    CheckpointBundle,
    _latest_checkpoint,
    _resolve_eval_dashboard_dir,
)
from scripts.training.unified_checkpoint_eval import uniform_matchup_schedule  # noqa: E402


def _write_unified_checkpoint(
    run_dir: Path,
    matchup_name: str,
    episode: int,
) -> Path:
    matchup_dir = run_dir / matchup_name
    matchup_dir.mkdir(parents=True, exist_ok=True)
    (matchup_dir / "matchup_label.json").write_text(
        json.dumps({"name": matchup_name}),
        encoding="utf-8",
    )
    ckpt_dir = (
        matchup_dir
        / "unified_selfplay"
        / "p1"
        / f"episode_{episode:06d}"
    )
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "weights").mkdir(parents=True)
    (ckpt_dir / "weights" / "agent_weights.json").write_text("{}", encoding="utf-8")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_type": "unified_selfplay",
                "matchup": matchup_name,
                "episodes_completed": episode,
                "target_episodes": 1000,
                "game_format": "silver_age",
                "opponent_mode": "dual",
                "p1_hero": "hero_a",
                "p2_hero": "hero_b",
                "deck_spec": {
                    "equipment_header": "hero_a",
                    "cards": {"a_red": 40},
                },
            }
        ),
        encoding="utf-8",
    )
    p2_dir = ckpt_dir.parent.parent / "p2" / ckpt_dir.name
    p2_dir.mkdir(parents=True)
    (p2_dir / "weights").mkdir(parents=True)
    (p2_dir / "weights" / "agent_weights.json").write_text("{}", encoding="utf-8")
    (p2_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_type": "unified_selfplay",
                "matchup": matchup_name,
                "episodes_completed": episode,
                "role": "p2",
                "game_format": "silver_age",
                "opponent_mode": "dual",
                "p1_hero": "hero_a",
                "p2_hero": "hero_b",
                "deck_spec": {
                    "equipment_header": "hero_b",
                    "cards": {"b_red": 40},
                },
            }
        ),
        encoding="utf-8",
    )
    return ckpt_dir


def test_equipment_header_from_deck_stem_reads_asset_first_line(tmp_path: Path) -> None:
    from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: E402
        equipment_header_from_deck_stem,
    )

    assets = tmp_path / "Assets"
    assets.mkdir()
    (assets / "fab_oscilio_omn.txt").write_text(
        "oscilio_omn nullrune_hood nullrune_robe star_fall\n"
        "cosmic_flare_red nebula_duality_red\n",
        encoding="utf-8",
    )

    header = equipment_header_from_deck_stem(
        "fab_oscilio_omn",
        assets,
        fallback="oscilio",
    )
    assert header.startswith("oscilio_omn")
    assert "nullrune_hood" in header


def test_eval_equipment_header_uses_deck_stem(tmp_path: Path) -> None:
    from eval_phase3_checkpoint import _equipment_header  # noqa: E402

    assets = tmp_path / "Assets"
    assets.mkdir()
    (assets / "fab_oscilio_omn.txt").write_text(
        "oscilio_omn nullrune_hood nullrune_robe star_fall\n"
        "cosmic_flare_red\n",
        encoding="utf-8",
    )
    bundle = CheckpointBundle(
        role="p1",
        checkpoint_dir=tmp_path,
        metadata={
            "matchup": "test",
            "episodes_completed": 1,
            "game_format": "silver_age",
            "p1_hero": "oscilio",
            "p1_deck": "fab_oscilio_omn",
            "deck_spec": {"equipment_header": "oscilio", "cards": {"cosmic_flare_red": 1}},
        },
        weights_path=tmp_path / "weights.json",
    )

    header = _equipment_header(bundle, assets_path=str(assets))
    assert header.startswith("oscilio_omn")
    assert "nullrune_hood" in header


def test_resolve_fabrary_equipment_header_uses_richest_asset() -> None:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Talishar" / "Assets"
    arakni = assets / "ArakniWebOfDeceitSAGEPrecon.txt"
    if not arakni.is_file():
        return

    sys.path.insert(0, str(repo / "scripts" / "training"))
    from train_dual_agent_common import resolve_fabrary_equipment_header  # noqa: E402

    header = resolve_fabrary_equipment_header(
        {"hero_id": "hero_arakni_web_of_deceit", "id": "fab_precon_sage_ch2_arakni_web_of_deceit"},
        assets,
    )
    assert header.startswith("arakni_web_of_deceit")
    assert len(header.split()) > 1
    assert "blade_beckoner_boots" in header


def test_latest_checkpoint_from_unified_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260626_215125"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"format": "silver_age"}),
        encoding="utf-8",
    )
    (run_dir / "checkpoint_eval_scope.json").write_text(
        json.dumps({"matchup_dir": "match_a"}),
        encoding="utf-8",
    )
    _write_unified_checkpoint(run_dir, "match_a", 200)
    latest_dir = _write_unified_checkpoint(run_dir, "match_b", 700)

    bundle = _latest_checkpoint(run_dir, "p1")
    assert bundle is not None
    assert bundle.checkpoint_dir == latest_dir
    assert bundle.episodes_completed == 700


def test_eval_dashboard_dir_at_unified_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260626_215125"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    ckpt_dir = _write_unified_checkpoint(run_dir, "match_b", 700)
    bundle = CheckpointBundle(
        role="p1",
        checkpoint_dir=ckpt_dir,
        metadata=json.loads((ckpt_dir / "metadata.json").read_text(encoding="utf-8")),
        weights_path=ckpt_dir / "weights" / "agent_weights.json",
    )

    eval_dir = _resolve_eval_dashboard_dir(run_dir, bundle)
    assert eval_dir == run_dir / "eval_dashboard"


def test_uniform_matchup_schedule_is_reproducible() -> None:
    keys = ["a", "b", "c"]
    first = uniform_matchup_schedule(20, keys, seed=42)
    second = uniform_matchup_schedule(20, keys, seed=42)
    assert first == second
    assert len(first) == 20
    assert all(item in keys for item in first)
