"""Tests for Talishar multi-backend URL pool."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from flesh_and_blood_rlbridge.talishar_backend_pool import (
    TalisharBackendPool,
    is_shard_connection_error,
    normalize_talishar_url,
    parse_talishar_urls_string,
    probe_backend_health,
    resolve_eval_backend_url,
    resolve_talishar_backend_urls,
    shard_eviction_threshold,
)


def test_normalize_talishar_url_adds_game_suffix() -> None:
    assert normalize_talishar_url("http://localhost:8080") == "http://localhost:8080/game"
    assert normalize_talishar_url("http://localhost:8080/game") == "http://localhost:8080/game"
    assert normalize_talishar_url("http://localhost:8081/game/") == "http://localhost:8081/game"


def test_parse_talishar_urls_string() -> None:
    parsed = parse_talishar_urls_string(
        "http://localhost:8080/game; http://localhost:8081/game,"
    )
    assert parsed == (
        "http://localhost:8080/game",
        "http://localhost:8081/game",
    )


def test_resolve_prefers_talishar_urls_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TALISHAR_EVAL_URL", raising=False)
    monkeypatch.setenv(
        "TALISHAR_URLS",
        "http://localhost:8081/game,http://localhost:8082/game",
    )
    monkeypatch.setenv("TALISHAR_URL", "http://localhost:8080/game")
    urls = resolve_talishar_backend_urls()
    assert urls == ("http://localhost:8081/game", "http://localhost:8082/game")


def test_resolve_falls_back_to_talishar_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TALISHAR_URLS", raising=False)
    monkeypatch.delenv("TALISHAR_EVAL_URL", raising=False)
    monkeypatch.setenv("TALISHAR_URL", "http://localhost:9090/game")
    urls = resolve_talishar_backend_urls()
    assert urls == ("http://localhost:9090/game",)


def test_url_for_worker_round_robin() -> None:
    pool = TalisharBackendPool(
        urls=("http://localhost:8080/game", "http://localhost:8081/game"),
    )
    assert pool.url_for_worker(0) == "http://localhost:8080/game"
    assert pool.url_for_worker(1) == "http://localhost:8081/game"
    assert pool.url_for_worker(2) == "http://localhost:8080/game"


def test_allocate_url_thread_safe() -> None:
    pool = TalisharBackendPool(
        urls=("http://localhost:8080/game", "http://localhost:8081/game"),
    )
    first = pool.allocate_url()
    second = pool.allocate_url()
    third = pool.allocate_url()
    assert (first, second, third) == (
        "http://localhost:8080/game",
        "http://localhost:8081/game",
        "http://localhost:8080/game",
    )


def test_health_check_skipped_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAB_SKIP_TALISHAR_HEALTH_CHECK", "1")
    ok, reason = probe_backend_health("http://localhost:8080/game")
    assert ok is True
    assert reason == "skipped"


def test_health_check_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAB_SKIP_TALISHAR_HEALTH_CHECK", raising=False)

    def _fake_probe(url: str, *, timeout: float = 3.0) -> tuple[bool, str]:
        if url.endswith("8081/game"):
            return False, "unreachable (connection refused)"
        return True, "rlstep"

    with patch(
        "flesh_and_blood_rlbridge.talishar_backend_pool.probe_backend_health",
        side_effect=_fake_probe,
    ):
        pool = TalisharBackendPool(
            urls=("http://localhost:8080/game", "http://localhost:8081/game"),
        )
        failed = pool.health_check()
    assert failed == ["http://localhost:8081/game"]


def test_filter_healthy_uses_reachable_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAB_SKIP_TALISHAR_HEALTH_CHECK", raising=False)

    def _fake_probe(url: str, *, timeout: float = 3.0) -> tuple[bool, str]:
        if "8082" in url or "8083" in url:
            return False, "unreachable (connection refused)"
        return True, "rlstep"

    with patch(
        "flesh_and_blood_rlbridge.talishar_backend_pool.probe_backend_health",
        side_effect=_fake_probe,
    ):
        pool = TalisharBackendPool(
            urls=(
                "http://localhost:8080/game",
                "http://localhost:8081/game",
                "http://localhost:8082/game",
                "http://localhost:8083/game",
            ),
        )
        filtered = pool.filter_healthy()
    assert filtered.urls == (
        "http://localhost:8080/game",
        "http://localhost:8081/game",
    )


def test_resolve_excludes_eval_shard_from_training(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "TALISHAR_URLS",
        "http://localhost:8080/game,"
        "http://localhost:8081/game,"
        "http://localhost:8082/game",
    )
    monkeypatch.setenv("TALISHAR_EVAL_URL", "http://localhost:8082/game")
    urls = resolve_talishar_backend_urls()
    assert urls == (
        "http://localhost:8080/game",
        "http://localhost:8081/game",
    )


def test_resolve_eval_backend_url_prefers_eval_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TALISHAR_EVAL_URL", "http://localhost:8083/game")
    monkeypatch.setenv("TALISHAR_URL", "http://localhost:8080/game")
    assert resolve_eval_backend_url() == "http://localhost:8083/game"


def test_resolve_eval_backend_url_falls_back_to_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TALISHAR_EVAL_URL", raising=False)
    monkeypatch.setenv("TALISHAR_URL", "http://localhost:8080/game")
    assert resolve_eval_backend_url() == "http://localhost:8080/game"


def test_resolve_render_backend_url_prefers_render_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from flesh_and_blood_rlbridge.talishar_backend_pool import resolve_render_backend_url

    monkeypatch.setenv("TALISHAR_RENDER_URL", "http://localhost:8085/game")
    monkeypatch.setenv("TALISHAR_EVAL_URL", "http://localhost:8084/game")
    assert resolve_render_backend_url() == "http://localhost:8085/game"


def test_resolve_excludes_render_shard_from_training(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "TALISHAR_URLS",
        "http://localhost:8080/game,"
        "http://localhost:8081/game,"
        "http://localhost:8082/game,"
        "http://localhost:8083/game",
    )
    monkeypatch.setenv("TALISHAR_EVAL_URL", "http://localhost:8082/game")
    monkeypatch.setenv("TALISHAR_RENDER_URL", "http://localhost:8083/game")
    urls = resolve_talishar_backend_urls()
    assert urls == (
        "http://localhost:8080/game",
        "http://localhost:8081/game",
    )


def test_is_shard_connection_error_detects_transport_failures() -> None:
    assert is_shard_connection_error(ConnectionError("boom"))
    assert is_shard_connection_error(
        ConnectionError('MaxRetryError("HTTPConnectionPool: RemoteDisconnected")')
    )
    assert not is_shard_connection_error(ValueError("bad action"))


def test_note_shard_failure_evicts_after_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAB_SHARD_EVICTION_THRESHOLD", "2")
    pool = TalisharBackendPool(
        urls=(
            "http://localhost:8080/game",
            "http://localhost:8081/game",
            "http://localhost:8082/game",
        ),
    )
    assert not pool.note_shard_failure("http://localhost:8081/game")
    assert pool.note_shard_failure("http://localhost:8081/game")
    assert pool.urls == (
        "http://localhost:8080/game",
        "http://localhost:8082/game",
    )


def test_note_shard_failure_does_not_evict_last_backend() -> None:
    pool = TalisharBackendPool(urls=("http://localhost:8080/game",))
    pool.note_shard_failure("http://localhost:8080/game")
    pool.note_shard_failure("http://localhost:8080/game")
    pool.note_shard_failure("http://localhost:8080/game")
    assert pool.urls == ("http://localhost:8080/game",)


def test_pick_replacement_skips_failed_backend() -> None:
    pool = TalisharBackendPool(
        urls=(
            "http://localhost:8080/game",
            "http://localhost:8081/game",
        ),
    )
    assert pool.pick_replacement(
        "http://localhost:8081/game",
        worker_index=1,
    ) == "http://localhost:8080/game"


def test_note_shard_success_resets_failure_streak(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAB_SHARD_EVICTION_THRESHOLD", "2")
    pool = TalisharBackendPool(
        urls=(
            "http://localhost:8080/game",
            "http://localhost:8081/game",
        ),
    )
    pool.note_shard_failure("http://localhost:8081/game")
    pool.note_shard_success("http://localhost:8081/game")
    assert not pool.note_shard_failure("http://localhost:8081/game")
    assert pool.urls == (
        "http://localhost:8080/game",
        "http://localhost:8081/game",
    )


def test_shard_eviction_threshold_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAB_SHARD_EVICTION_THRESHOLD", "5")
    assert shard_eviction_threshold() == 5
