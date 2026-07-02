"""Targeted tests for Talishar fast HTTP client retry semantics."""

from __future__ import annotations

import pytest
import requests

from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment
from flesh_and_blood_rlbridge.talishar_fast_client import TalisharFastClient, make_talishar_session


def test_fast_client_session_retries_only_get() -> None:
    session = make_talishar_session()
    adapter = session.adapters["http://"]
    retry = adapter.max_retries
    assert "GET" in retry.allowed_methods
    assert "POST" not in retry.allowed_methods


def test_engine_session_retries_only_get() -> None:
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    session = env._make_session()
    adapter = session.adapters["http://"]
    retry = adapter.max_retries
    assert "GET" in retry.allowed_methods
    assert "POST" not in retry.allowed_methods


def test_post_rlstep_does_not_loop_request_retries() -> None:
    client = TalisharFastClient("http://localhost:8080/game", keep_alive=False)

    call_count = {"n": 0}

    def _fail_post(*args, **kwargs):
        call_count["n"] += 1
        raise requests.ConnectionError("simulated transport failure")

    client.session.post = _fail_post  # type: ignore[method-assign]

    with pytest.raises(requests.ConnectionError):
        client.post_rlstep(
            {
                "gameName": "g1",
                "playerID": 1,
                "authKey": "k1",
                "mode": 99,
                "trainingMode": True,
            }
        )

    assert call_count["n"] == 1
