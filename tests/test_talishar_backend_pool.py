"""Tests for Talishar multi-backend URL pool."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from flesh_and_blood_rlbridge.talishar_backend_pool import (
    TalisharBackendPool,
    normalize_talishar_url,
    parse_talishar_urls_string,
    probe_backend_health,
    resolve_talishar_backend_urls,
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
    monkeypatch.setenv(
        "TALISHAR_URLS",
        "http://localhost:8081/game,http://localhost:8082/game",
    )
    monkeypatch.setenv("TALISHAR_URL", "http://localhost:8080/game")
    urls = resolve_talishar_backend_urls()
    assert urls == ("http://localhost:8081/game", "http://localhost:8082/game")


def test_resolve_falls_back_to_talishar_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TALISHAR_URLS", raising=False)
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
