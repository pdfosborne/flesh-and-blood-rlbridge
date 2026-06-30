"""Round-robin pool of Talishar HTTP backends for parallel RL rollouts."""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

from .talishar_fast_client import DEFAULT_TALISHAR_URL, TalisharFastClient

_URL_SPLIT_RE = re.compile(r"[,;]+")
_HEALTH_PROBE_BODY = json.dumps(
    {"gameName": "0", "playerID": 1, "authKey": "", "mode": 99}
).encode("utf-8")


def _rlstep_response_ready(
    *,
    status: int,
    body: str,
    content_type: str = "",
) -> bool:
    """True when RLStep.php is present and responding like the rl-bridge overlay."""
    if status == 404:
        return False
    text = body.strip()
    if "{" in text or "[" in text:
        return True
    # RLStep sets application/json even when the POST body is empty (e.g. probe game).
    if status == 200 and not text and "application/json" in content_type.lower():
        return True
    return False


def normalize_talishar_url(url: str) -> str:
    """Normalize a Talishar game API base URL."""
    from urllib.parse import urlsplit, urlunsplit

    text = str(url).strip().rstrip("/")
    if not text:
        return DEFAULT_TALISHAR_URL
    if text.endswith("/game"):
        return text
    parts = urlsplit(text)
    path = parts.path or ""
    if path in ("", "/"):
        return urlunsplit((parts.scheme, parts.netloc, "/game", "", ""))
    return text


def parse_talishar_urls_string(raw: str) -> tuple[str, ...]:
    """Parse comma- or semicolon-separated backend URLs."""
    parts = [normalize_talishar_url(p) for p in _URL_SPLIT_RE.split(raw) if p.strip()]
    return tuple(parts)


def resolve_talishar_backend_urls(
    *,
    configured_backends: Iterable[str] = (),
    fallback_url: str | None = None,
) -> tuple[str, ...]:
    """Resolve Talishar backend URLs for training rollouts (excludes eval shard)."""
    env_urls = os.environ.get("TALISHAR_URLS", "").strip()
    if env_urls:
        parsed = parse_talishar_urls_string(env_urls)
        if parsed:
            return _exclude_eval_backend(parsed)

    configured = tuple(
        normalize_talishar_url(u)
        for u in configured_backends
        if str(u).strip()
    )
    if configured:
        return _exclude_eval_backend(configured)

    single = (
        fallback_url
        or os.environ.get("TALISHAR_URL", "").strip()
        or DEFAULT_TALISHAR_URL
    )
    return _exclude_eval_backend((normalize_talishar_url(single),))


def resolve_eval_backend_url(*, fallback_url: str | None = None) -> str:
    """Return the dedicated eval Talishar backend URL when configured."""
    eval_url = os.environ.get("TALISHAR_EVAL_URL", "").strip()
    if eval_url:
        return normalize_talishar_url(eval_url)
    return normalize_talishar_url(
        fallback_url
        or os.environ.get("TALISHAR_URL", "").strip()
        or DEFAULT_TALISHAR_URL
    )


def resolve_render_backend_url(*, fallback_url: str | None = None) -> str:
    """Return the dedicated live-render Talishar backend when configured."""
    render_url = os.environ.get("TALISHAR_RENDER_URL", "").strip()
    if render_url:
        return normalize_talishar_url(render_url)
    return resolve_eval_backend_url(fallback_url=fallback_url)


def _exclude_reserved_backends(urls: Iterable[str]) -> tuple[str, ...]:
    """Drop dedicated eval and render shards from a training backend URL list."""
    reserved: set[str] = set()
    for env_key in ("TALISHAR_EVAL_URL", "TALISHAR_RENDER_URL"):
        reserved_url = os.environ.get(env_key, "").strip()
        if reserved_url:
            reserved.add(normalize_talishar_url(reserved_url))
    if not reserved:
        return tuple(urls)
    training = tuple(
        normalize_talishar_url(url)
        for url in urls
        if normalize_talishar_url(url) not in reserved
    )
    if training:
        return training
    return tuple(normalize_talishar_url(url) for url in urls)


def _exclude_eval_backend(urls: Iterable[str]) -> tuple[str, ...]:
    """Drop the dedicated eval shard from a training backend URL list."""
    return _exclude_reserved_backends(urls)


def probe_backend_health(url: str, *, timeout: float = 3.0) -> tuple[bool, str]:
    """Return whether *url* is ready for fast training and a short status label."""
    if os.environ.get("FAB_SKIP_TALISHAR_HEALTH_CHECK", "").strip() in {
        "1",
        "true",
        "yes",
    }:
        return True, "skipped"

    normalized = normalize_talishar_url(url)
    rlstep_url = normalized + TalisharFastClient.RLSTEP_PATH
    headers = {"User-Agent": "TalisharRLEnv/health"}
    try:
        get_req = urllib.request.Request(  # noqa: S310
            rlstep_url,
            headers=headers,
            method="GET",
        )
        with urllib.request.urlopen(get_req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read(1024).decode("utf-8", errors="replace")
            content_type = resp.headers.get("Content-Type", "")
            if _rlstep_response_ready(
                status=int(resp.status),
                body=body,
                content_type=content_type,
            ):
                return True, "rlstep"
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            body = exc.read(1024).decode("utf-8", errors="replace")
            content_type = exc.headers.get("Content-Type", "")
            if _rlstep_response_ready(
                status=exc.code,
                body=body,
                content_type=content_type,
            ):
                return True, "rlstep"
        if exc.code == 404:
            return False, "RLStep.php not found (404)"
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return False, f"unreachable ({reason})"
    except TimeoutError:
        return False, "timeout"
    except OSError as exc:
        return False, f"unreachable ({exc})"

    try:
        post_req = urllib.request.Request(  # noqa: S310
            rlstep_url,
            data=_HEALTH_PROBE_BODY,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(post_req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read(1024).decode("utf-8", errors="replace")
            content_type = resp.headers.get("Content-Type", "")
            if _rlstep_response_ready(
                status=int(resp.status),
                body=body,
                content_type=content_type,
            ):
                return True, "rlstep"
            return False, f"RLStep non-JSON response ({resp.status})"
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False, "RLStep.php not found (404)"
        body = exc.read(1024).decode("utf-8", errors="replace")
        content_type = exc.headers.get("Content-Type", "")
        if _rlstep_response_ready(
            status=exc.code,
            body=body,
            content_type=content_type,
        ):
            return True, "rlstep"
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return False, f"unreachable ({reason})"
    except TimeoutError:
        return False, "timeout"
    except OSError as exc:
        return False, f"unreachable ({exc})"


@dataclass
class TalisharBackendPool:
    """Thread-safe router across one or more Talishar HTTP backends."""

    urls: tuple[str, ...]
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _next_index: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not self.urls:
            object.__setattr__(self, "urls", (DEFAULT_TALISHAR_URL,))

    @classmethod
    def from_runtime(cls, *, fallback_url: str | None = None) -> TalisharBackendPool:
        configured: tuple[str, ...] = ()
        try:
            from runtime_defaults import RUNTIME  # noqa: PLC0415

            configured = tuple(RUNTIME.meta.engine.talishar_backends)
        except Exception:
            pass
        urls = resolve_talishar_backend_urls(
            configured_backends=configured,
            fallback_url=fallback_url,
        )
        return cls(urls=urls)

    @classmethod
    def from_urls(cls, urls: Iterable[str]) -> TalisharBackendPool:
        resolved = resolve_talishar_backend_urls(configured_backends=urls)
        return cls(urls=resolved)

    @property
    def primary_url(self) -> str:
        return self.urls[0]

    def url_for_worker(self, worker_index: int) -> str:
        """Map a stable worker slot index to a backend."""
        return self.urls[int(worker_index) % len(self.urls)]

    def allocate_url(self) -> str:
        """Thread-safe round-robin assignment for new worker slots."""
        with self._lock:
            url = self.urls[self._next_index % len(self.urls)]
            self._next_index += 1
            return url

    def health_check(self, *, timeout: float = 3.0) -> list[str]:
        """Return URLs whose RLStep endpoint is unreachable."""
        failed: list[str] = []
        for url in self.urls:
            ok, _reason = probe_backend_health(url, timeout=timeout)
            if not ok:
                failed.append(url)
        return failed

    def health_status(self, *, timeout: float = 3.0) -> dict[str, str]:
        """Map each backend URL to a probe status (``ok`` or failure reason)."""
        status: dict[str, str] = {}

        def _probe(url: str) -> tuple[str, str]:
            ok, reason = probe_backend_health(url, timeout=timeout)
            return url, "ok" if ok else reason

        if len(self.urls) <= 1:
            url, label = _probe(self.urls[0])
            status[url] = label
            return status

        with ThreadPoolExecutor(max_workers=len(self.urls)) as pool:
            futures = [pool.submit(_probe, url) for url in self.urls]
            for fut in as_completed(futures):
                url, label = fut.result()
                status[url] = label
        return status

    def filter_healthy(
        self,
        *,
        timeout: float = 3.0,
        min_healthy: int = 1,
    ) -> TalisharBackendPool:
        """Drop unreachable backends; fail only when none are usable."""
        status = self.health_status(timeout=timeout)
        healthy = [url for url in self.urls if status.get(url) == "ok"]
        unhealthy = [
            f"{url} ({status[url]})"
            for url in self.urls
            if status.get(url) != "ok"
        ]
        if len(healthy) < max(1, int(min_healthy)):
            detail = "; ".join(unhealthy) if unhealthy else "no backends configured"
            raise RuntimeError(
                "No Talishar backends are reachable for fast training. "
                f"Probed: {detail}. "
                "Start the stack with: python start_talishar.py --backend-only --shards N"
            )
        if unhealthy:
            print(
                "  WARNING: skipping unreachable Talishar backend(s): "
                + "; ".join(unhealthy),
                flush=True,
            )
            print(
                f"  Using {len(healthy)}/{len(self.urls)} backend(s) for training.",
                flush=True,
            )
        if len(healthy) == len(self.urls):
            return self
        return replace(self, urls=tuple(healthy))

    def format_log_label(self) -> str:
        if len(self.urls) == 1:
            return self.urls[0]
        return f"{len(self.urls)} backends ({', '.join(self.urls)})"
