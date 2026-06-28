"""Optimized HTTP client for Talishar RL training (fast backend).

Uses keep-alive connections, optional RLStep overlay (single round-trip per action),
and tight priority polling without the legacy 350ms post-ProcessInput sleep.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TALISHAR_URL = "http://localhost:8080/game"

_FAST_HTTP_RETRIES = 3


def _parse_json_body(body_text: str, *, allow_empty: bool = False) -> dict[str, Any]:
    text = body_text.strip()
    if allow_empty and not text:
        return {}
    obj_start = text.find("{")
    arr_start = text.find("[")
    starts = [i for i in (obj_start, arr_start) if i >= 0]
    if starts:
        text = text[min(starts):]
    elif allow_empty:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = body_text[:500].replace("\n", " ")
        raise RuntimeError(
            f"Talishar returned non-JSON response: {preview!r}"
        ) from exc
    return data if isinstance(data, dict) else {"_raw": data}


def make_talishar_session(*, keep_alive: bool = True, pool_size: int | None = None) -> requests.Session:
    if pool_size is None:
        pool_env = os.environ.get("FAB_TALISHAR_HTTP_POOL_SIZE", "").strip()
        if pool_env:
            try:
                pool_size = max(4, int(pool_env))
            except ValueError:
                pool_size = 32
        else:
            pool_size = 32
    pool_size = max(4, int(pool_size))
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.2,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods={"GET", "POST"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        pool_connections=min(8, pool_size),
        pool_maxsize=pool_size,
        max_retries=retry,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    headers = {"User-Agent": "TalisharRLEnv/2.0-fast"}
    if keep_alive:
        headers["Connection"] = "keep-alive"
    else:
        headers["Connection"] = "close"
    session.headers.update(headers)
    return session


class TalisharFastClient:
    """Low-latency Talishar HTTP helper for training rollouts."""

    RLSTEP_PATH = "/APIs/RLStep.php"

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float = 30.0,
        keep_alive: bool = True,
        pool_size: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout = float(request_timeout)
        self.session = make_talishar_session(keep_alive=keep_alive, pool_size=pool_size)
        self._rlstep_available: Optional[bool] = None

    def probe_rlstep(self, *, force: bool = False) -> bool:
        if self._rlstep_available is not None and not force:
            return self._rlstep_available
        from .talishar_backend_pool import probe_backend_health  # noqa: PLC0415

        ok, _reason = probe_backend_health(self.base_url, timeout=min(5.0, self.request_timeout))
        self._rlstep_available = ok
        return ok

    def http_get(
        self,
        path: str,
        params: dict[str, str],
        *,
        allow_empty_body: bool = False,
    ) -> dict[str, Any]:
        url = self.base_url + path
        resp = self.session.get(url, params=params, timeout=self.request_timeout)
        body_text = resp.text
        if allow_empty_body and body_text.strip() == "":
            return {}
        return _parse_json_body(body_text, allow_empty=allow_empty_body)

    def http_post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + path
        resp = self.session.post(
            url,
            json=payload,
            timeout=self.request_timeout,
        )
        return _parse_json_body(resp.text)

    def post_rlstep(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.http_post_json(self.RLSTEP_PATH, payload)

    def fetch_both_player_states(
        self,
        *,
        game_name: str,
        p1_auth: str,
        p2_auth: str,
        parallel: bool = True,
    ) -> dict[int, dict[str, Any]]:
        def fetch(pid: int, auth: str) -> dict[str, Any]:
            return self.http_get(
                "/GetNextTurn.php",
                {
                    "gameName": game_name,
                    "playerID": str(pid),
                    "authKey": auth,
                    "lastUpdate": "0",
                },
            )

        if not parallel:
            return {1: fetch(1, p1_auth), 2: fetch(2, p2_auth)}

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(fetch, 1, p1_auth)
            f2 = pool.submit(fetch, 2, p2_auth)
            return {1: f1.result(), 2: f2.result()}

    def wait_for_priority(
        self,
        *,
        game_name: str,
        p1_auth: str,
        p2_auth: str,
        is_game_over: Callable[[dict[str, Any]], bool],
        adopt_state: Callable[[dict[str, Any], int], dict[str, Any]],
        infer_priority: Callable[[dict[int, dict[str, Any]]], Optional[int]],
        poll_interval: float,
        max_polls: int,
        deadlock_polls: int,
    ) -> dict[str, Any]:
        """Poll P1/P2 snapshots until one seat reports priority (fast polling)."""
        import time

        for i in range(max_polls):
            states = self.fetch_both_player_states(
                game_name=game_name,
                p1_auth=p1_auth,
                p2_auth=p2_auth,
                parallel=True,
            )
            error_count = 0
            for pid in (1, 2):
                state = states[pid]
                err_msg = state.get("error", "")
                is_transient = (
                    isinstance(err_msg, str) and "too short" in err_msg and i < 6
                )
                if (state.get("havePriority", False) or is_game_over(state)) and (
                    not is_transient
                ):
                    return adopt_state(state, pid)
                if err_msg and not is_transient:
                    error_count += 1
            if error_count == 2:
                return {"error": "game_crashed"}
            if i >= deadlock_polls and i % 2 == 0:
                inferred = infer_priority(states)
                if inferred is not None:
                    inferred_state = states[inferred]
                    if inferred_state.get("havePriority", False) or is_game_over(
                        inferred_state
                    ):
                        return adopt_state(inferred_state, inferred)
            if i + 1 < max_polls:
                time.sleep(poll_interval)
        inferred = infer_priority(states)
        if inferred is not None:
            return adopt_state(states[inferred], inferred)
        return adopt_state(states.get(1, {}), 1)
