"""Tests for real-time Talishar live play helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "eval"))

from talishar_live_play import (  # noqa: E402
    LivePlayContext,
    _outcome_from_human_perspective,
    configure_human_vs_agent,
    deck_labels_from_bundle,
    prepare_live_play_context,
    resolve_checkpoint_bundles,
)
from eval_phase3_checkpoint import CheckpointBundle  # noqa: E402


def _write_checkpoint(
    root: Path,
    *,
    matchup: str = "p3_aurora-vs-briar",
    episode: int = 25,
) -> Path:
    ckpt_dir = root / matchup / "p1" / f"episode_{episode:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "weights").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "weights" / "agent_weights.json").write_text("{}", encoding="utf-8")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "matchup": matchup,
                "episodes_completed": episode,
                "game_format": "silver_age",
                "p1_hero": "aurora",
                "p2_hero": "briar",
                "opponent_mode": "preset",
                "opponent_deck_name": "BriarSAGEPrecon",
                "deck_spec": {
                    "equipment_header": "aurora",
                    "cards": {"a_red": 40},
                },
            }
        ),
        encoding="utf-8",
    )
    return ckpt_dir


def test_resolve_checkpoint_bundles_finds_latest(tmp_path: Path) -> None:
    results_dir = tmp_path / "run"
    results_dir.mkdir()
    _write_checkpoint(results_dir, episode=10)
    later = _write_checkpoint(results_dir, episode=20)

    p1_bundle, p2_bundle = resolve_checkpoint_bundles(results_dir)
    assert p1_bundle is not None
    assert p1_bundle.checkpoint_dir == later
    assert p2_bundle is None


def test_prepare_live_play_context_writes_deck(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "run"
    results_dir.mkdir()
    _write_checkpoint(results_dir, episode=15)
    assets = tmp_path / "Assets"
    assets.mkdir()

    written: list[tuple[dict[str, int], str, str, str]] = []

    def _fake_write_deck(
        deck: dict[str, int],
        equipment_header: str,
        deck_name: str,
        assets_path: str,
    ) -> Path:
        written.append((deck, equipment_header, deck_name, assets_path))
        out = assets / f"{deck_name}.txt"
        out.write_text("deck", encoding="utf-8")
        return out

    monkeypatch.setattr(
        "talishar_live_play._write_deck_file",
        _fake_write_deck,
    )
    monkeypatch.setattr(
        "talishar_live_play._load_agent",
        lambda _path: object(),
    )

    ctx = prepare_live_play_context(
        results_dir,
        assets_path=str(assets),
    )
    assert ctx.p1_bundle.episodes_completed == 15
    assert ctx.p2_deck_name == "BriarSAGEPrecon"
    assert ctx.opponent_label.startswith("preset deck")
    assert len(written) == 1
    assert written[0][0] == {"a_red": 40}


def test_configure_human_vs_agent_opponent_deck_uses_p2_seat() -> None:
    base = LivePlayContext(
        p1_bundle=object(),
        p2_bundle=None,
        p1_agent="agent",
        p2_agent=None,
        p1_deck_name="trained_deck",
        p2_deck_name="BriarSAGEPrecon",
        game_format="silver_age",
        opponent_label="preset",
        cleanup_files=[],
        trained_deck_label="aurora",
        opponent_deck_label="BriarSAGEPrecon",
    )
    hva = configure_human_vs_agent(base, human_deck="opponent")
    assert hva.human_vs_agent
    assert hva.human_deck == "opponent"
    assert hva.human_player_id == 2
    assert hva.agent_player_id == 1
    assert hva.p1_deck_name == "trained_deck"
    assert hva.p2_deck_name == "BriarSAGEPrecon"
    assert hva.p1_agent == "agent"
    assert hva.p2_agent is None


def test_configure_human_vs_agent_trained_deck_uses_p1_seat() -> None:
    base = LivePlayContext(
        p1_bundle=object(),
        p2_bundle=None,
        p1_agent="agent",
        p2_agent=None,
        p1_deck_name="trained_deck",
        p2_deck_name="BriarSAGEPrecon",
        game_format="silver_age",
        opponent_label="preset",
        cleanup_files=[],
        trained_deck_label="aurora",
        opponent_deck_label="BriarSAGEPrecon",
    )
    hva = configure_human_vs_agent(base, human_deck="trained")
    assert hva.human_vs_agent
    assert hva.human_deck == "trained"
    assert hva.human_player_id == 1
    assert hva.agent_player_id == 2
    assert hva.p1_deck_name == "trained_deck"
    assert hva.p2_deck_name == "BriarSAGEPrecon"
    assert hva.p1_agent is None
    assert hva.p2_agent == "agent"


def test_outcome_from_human_perspective_flips_for_p2() -> None:
    assert _outcome_from_human_perspective("win", 1) == "win"
    assert _outcome_from_human_perspective("loss", 1) == "loss"
    assert _outcome_from_human_perspective("win", 2) == "loss"
    assert _outcome_from_human_perspective("loss", 2) == "win"
    assert _outcome_from_human_perspective("draw", 2) == "draw"


def test_deck_labels_dual_mode_ignores_stale_preset_name() -> None:
    p1 = CheckpointBundle(
        role="p1",
        checkpoint_dir=Path("/tmp/ep"),
        metadata={
            "p1_hero": "briar",
            "p2_hero": "briar",
            "opponent_mode": "dual",
            "opponent_deck_name": "Ira",
        },
        weights_path=Path("/tmp/ep/weights/agent_weights.json"),
    )
    p2 = CheckpointBundle(
        role="p2",
        checkpoint_dir=Path("/tmp/ep2"),
        metadata={"p2_hero": "briar"},
        weights_path=Path("/tmp/ep2/weights/agent_weights.json"),
    )
    trained, opponent = deck_labels_from_bundle(p1, p2)
    assert trained == "briar"
    assert opponent == "briar"
