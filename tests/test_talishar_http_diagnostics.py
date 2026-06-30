"""Tests for Talishar HTTP failure diagnostics."""

from __future__ import annotations

from flesh_and_blood_rlbridge.talishar_fast_client import (
    _should_retry_talishar_http,
    diagnose_talishar_http_failure,
)


def test_diagnose_disk_full() -> None:
    body = (
        "<br /><b>Notice</b>: copy(): Write failed with errno=28 "
        "No space left on device in CreateLocalGame.php"
    )
    err = diagnose_talishar_http_failure(
        "http://localhost:8080/game/APIs/CreateLocalGame.php",
        body,
        method="POST",
    )
    assert "out of disk space" in str(err)
    assert "errno 28" in str(err)


def test_diagnose_empty_body() -> None:
    err = diagnose_talishar_http_failure(
        "http://localhost:8080/game/GetNextTurn.php",
        "",
    )
    assert "empty body" in str(err)


def test_should_not_retry_disk_full() -> None:
    body = "errno=28 No space left on device"
    assert _should_retry_talishar_http(body) is False


def test_should_retry_empty_body() -> None:
    assert _should_retry_talishar_http("") is True
    assert _should_retry_talishar_http("   ") is True
