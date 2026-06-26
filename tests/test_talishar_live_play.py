"""Tests for real-time Talishar live play helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "eval"))

from fab_bridge.agents import unified_agent_cache_format, unified_agent_weights_path  # noqa: E402
from talishar_live_play import (  # noqa: E402
    LivePlayContext,
    _gui_human_deck_to_trained_opponent,
    _outcome_from_human_perspective,
    _refresh_human_action_coach,
    configure_human_vs_agent,
    deck_labels_from_bundle,
    prepare_live_play_context,
    prepare_unified_live_play_context,
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


def test_gui_human_deck_mapping() -> None:
    assert _gui_human_deck_to_trained_opponent("player") == "trained"
    assert _gui_human_deck_to_trained_opponent("opponent") == "opponent"


def test_unified_agent_weights_path(tmp_path: Path) -> None:
    path = unified_agent_weights_path(tmp_path / "cache", "silver_age")
    assert path.parent.name == "silver_age"
    assert path.name.startswith("unified_agent_v")


def test_unified_agent_weights_path_maps_sage_to_silver_age(tmp_path: Path) -> None:
    assert unified_agent_cache_format("sage") == "silver_age"
    path = unified_agent_weights_path(tmp_path / "cache", "sage")
    assert path.parent.name == "silver_age"


def test_prepare_unified_live_play_context(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    fmt_dir = cache_dir / "silver_age"
    fmt_dir.mkdir(parents=True)
    weights = fmt_dir / unified_agent_weights_path(cache_dir, "silver_age").name
    weights.write_text("{}", encoding="utf-8")
    assets = tmp_path / "Assets"
    assets.mkdir()

    class _FakeEnv:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "talishar_live_play.TalisharEngineEnvironment",
        lambda **kwargs: _FakeEnv(),
    )
    monkeypatch.setattr(
        "eval_sideboard_compare._load_unified_agent",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "agent_cache.clone_agent_weights",
        lambda _src, _dst: None,
    )

    written: list[str] = []

    def _fake_write_deck(
        deck: dict[str, int],
        equipment_header: str,
        deck_name: str,
        assets_path: str,
    ) -> Path:
        written.append(deck_name)
        out = assets / f"{deck_name}.txt"
        out.write_text("deck", encoding="utf-8")
        return out

    monkeypatch.setattr("talishar_live_play._write_deck_file", _fake_write_deck)

    ctx = prepare_unified_live_play_context(
        player_deck={"a_red": 40},
        opponent_asset_stem="OppPrecon",
        player_equipment_header="aurora",
        game_format="silver_age",
        assets_path=str(assets),
        cache_dir=cache_dir,
        base_url="http://localhost:8080/game",
        fe_url="http://localhost:5173",
        human_deck="opponent",
        player_deck_label="My deck",
        opponent_deck_label="Opp deck",
    )
    assert ctx.human_vs_agent
    assert ctx.human_player_id == 2
    assert ctx.p2_deck_name == "OppPrecon"
    assert ctx.p1_bundle is None
    assert ctx.cpp_engine_deck1 == "aurora"
    assert ctx.cpp_engine_deck2 == "OppPrecon"
    assert len(written) == 1


def test_prepare_unified_live_play_context_watch_mode(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    fmt_dir = cache_dir / "silver_age"
    fmt_dir.mkdir(parents=True)
    weights = fmt_dir / unified_agent_weights_path(cache_dir, "silver_age").name
    weights.write_text("{}", encoding="utf-8")
    assets = tmp_path / "Assets"
    assets.mkdir()

    class _FakeEnv:
        def close(self) -> None:
            pass

    monkeypatch.setattr("talishar_live_play.TalisharEngineEnvironment", lambda **kwargs: _FakeEnv())
    monkeypatch.setattr("eval_sideboard_compare._load_unified_agent", lambda *_a, **_k: object())
    monkeypatch.setattr("agent_cache.clone_agent_weights", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "talishar_live_play._write_deck_file",
        lambda *_a, **_k: assets / "deck.txt",
    )

    ctx = prepare_unified_live_play_context(
        player_deck={"a_red": 40},
        opponent_asset_stem="OppPrecon",
        player_equipment_header="aurora",
        game_format="silver_age",
        assets_path=str(assets),
        cache_dir=cache_dir,
        base_url="http://localhost:8080/game",
        fe_url="http://localhost:5173",
        human_deck="watch",
    )
    assert not ctx.human_vs_agent
    assert ctx.p1_agent is not None
    assert ctx.p2_agent is not None


def test_prepare_unified_live_play_context_resolves_sage_cpp_stems(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "cache"
    fmt_dir = cache_dir / "silver_age"
    fmt_dir.mkdir(parents=True)
    weights = fmt_dir / unified_agent_weights_path(cache_dir, "silver_age").name
    weights.write_text("{}", encoding="utf-8")
    assets = tmp_path / "Assets"
    assets.mkdir()
    (assets / "BriarSAGEPrecon.txt").write_text("deck", encoding="utf-8")

    class _FakeEnv:
        def close(self) -> None:
            pass

    monkeypatch.setattr("talishar_live_play.TalisharEngineEnvironment", lambda **kwargs: _FakeEnv())
    monkeypatch.setattr("eval_sideboard_compare._load_unified_agent", lambda *_a, **_k: object())
    monkeypatch.setattr("agent_cache.clone_agent_weights", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "talishar_live_play._write_deck_file",
        lambda *_a, **_k: assets / "deck.txt",
    )

    ctx = prepare_unified_live_play_context(
        player_deck={"a_red": 40},
        opponent_asset_stem="BriarSAGEPrecon",
        player_equipment_header="briar",
        game_format="sage",
        assets_path=str(assets),
        cache_dir=cache_dir,
        base_url="http://localhost:8080/game",
        fe_url="http://localhost:5173",
        human_deck="opponent",
    )
    assert ctx.cpp_engine_deck1 == "BriarSAGEPrecon"
    assert ctx.cpp_engine_deck2 == "BriarSAGEPrecon"


def test_prepare_unified_live_play_context_logic_opponent(tmp_path: Path, monkeypatch) -> None:
    from train_play import LOGIC_POLICY  # noqa: PLC0415

    cache_dir = tmp_path / "cache"
    fmt_dir = cache_dir / "silver_age"
    fmt_dir.mkdir(parents=True)
    weights = fmt_dir / unified_agent_weights_path(cache_dir, "silver_age").name
    weights.write_text("{}", encoding="utf-8")
    assets = tmp_path / "Assets"
    assets.mkdir()

    class _FakeEnv:
        def close(self) -> None:
            pass

    monkeypatch.setattr("talishar_live_play.TalisharEngineEnvironment", lambda **kwargs: _FakeEnv())
    monkeypatch.setattr("eval_sideboard_compare._load_unified_agent", lambda *_a, **_k: object())
    monkeypatch.setattr("agent_cache.clone_agent_weights", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "talishar_live_play._write_deck_file",
        lambda *_a, **_k: assets / "deck.txt",
    )

    ctx = prepare_unified_live_play_context(
        player_deck={"a_red": 40},
        opponent_asset_stem="OppPrecon",
        player_equipment_header="aurora",
        game_format="silver_age",
        assets_path=str(assets),
        cache_dir=cache_dir,
        base_url="http://localhost:8080/game",
        fe_url="http://localhost:5173",
        human_deck="opponent",
        opponent_policy="logic",
    )
    assert ctx.opponent_policy == "logic"
    assert ctx.p1_agent is LOGIC_POLICY
    assert ctx.p2_agent is None
    assert ctx.trained_agent is not None


def test_refresh_human_action_coach_uses_callback() -> None:
    hints_received: list[list[dict]] = []

    class _Coach:
        def build_hints(self, *_args, **_kwargs) -> list:
            from flesh_and_blood_rlbridge.frontend_action_overlay import ActionCoachHint

            return [
                ActionCoachHint(
                    index=0,
                    label="Attack",
                    policy_pct=0.8,
                    is_best=True,
                )
            ]

    class _Env:
        _last_state = {}
        _acting_player_id = 1
        _last_action_overlay_key = None

        def _legal_actions(self, _state: dict) -> list:
            return [{"index": 0, "label": "Attack", "zone": "hand"}]

        def _encode_observation(self, _state: dict, _legal: list) -> str:
            return "{}"

        def update_frontend_action_overlay(self, *_args, **_kwargs) -> None:
            raise AssertionError("overlay should not be used with callback")

    ctx = LivePlayContext(
        p1_agent=None,
        p2_agent="agent",
        p1_deck_name="p1",
        p2_deck_name="p2",
        game_format="silver_age",
        opponent_label="",
        cleanup_files=[],
        human_vs_agent=True,
        human_player_id=2,
    )
    _refresh_human_action_coach(
        _Env(),
        ctx,
        _Coach(),
        {"turnNo": 1, "playerHand": [], "turnPhase": {}, "playerPrompt": {}, "playerInputPopUp": {}},
        on_hints=lambda rows: hints_received.append(rows),
        last_hint_key=[""],
    )
    assert len(hints_received) == 1
    assert hints_received[0][0]["label"] == "Attack"
    assert hints_received[0][0]["isBest"] is True
