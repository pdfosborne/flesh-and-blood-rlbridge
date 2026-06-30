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
from pathlib import Path
from typing import Iterable, Optional

from .talishar_fast_client import DEFAULT_TALISHAR_URL, TalisharFastClient

_URL_SPLIT_RE = re.compile(r"[,;]+")
_HEALTH_PROBE_BODY = json.dumps(
    {"gameName": "0", "playerID": 1, "authKey": "", "mode": 99}
).encode("utf-8")

_DEFAULT_SHARD_EVICTION_THRESHOLD = 3


def shard_eviction_threshold() -> int:
    raw = os.environ.get("FAB_SHARD_EVICTION_THRESHOLD", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return _DEFAULT_SHARD_EVICTION_THRESHOLD


def is_shard_connection_error(exc: BaseException) -> bool:
    """True for transport-level Talishar backend failures that warrant eviction."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) is not None:
        return True
    text = repr(exc)
    markers = (
        "RemoteDisconnected",
        "Connection aborted",
        "Connection refused",
        "Max retries exceeded",
        "Connection reset",
    )
    return any(marker in text for marker in markers)


def is_shard_reset_error(exc: BaseException) -> bool:
    """True when a shard cannot start a game (Start.php / CreateGame failures)."""
    if is_shard_connection_error(exc):
        return True
    text = repr(exc)
    markers = (
        "Start.php returned non-JSON",
        "CreateGame failed",
        "CreateLocalGame",
        "Undefined array key 0",
        "non-JSON response",
        "TalisharConnectionError",
    )
    return any(marker in text for marker in markers)


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


def resolve_eval_backend_candidates(
    *,
    fallback_url: str | None = None,
    include_render_fallback: bool | None = None,
) -> tuple[str, ...]:
    """Ordered Talishar URLs to try for checkpoint eval (dedicated eval first)."""
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(url: str) -> None:
        norm = normalize_talishar_url(url)
        if norm and norm not in seen:
            seen.add(norm)
            candidates.append(norm)

    eval_url = os.environ.get("TALISHAR_EVAL_URL", "").strip()
    if eval_url:
        _add(eval_url)
    elif fallback_url:
        _add(fallback_url)

    for url in resolve_talishar_backend_urls(fallback_url=fallback_url):
        _add(url)

    if include_render_fallback is None:
        include_render_fallback = os.environ.get(
            "FAB_EVAL_FALLBACK_TO_RENDER", ""
        ).strip().lower() in {"1", "true", "yes"}
    render_url = os.environ.get("TALISHAR_RENDER_URL", "").strip()
    if not include_render_fallback and render_url:
        render_norm = normalize_talishar_url(render_url)
        candidates = [url for url in candidates if url != render_norm]

    if not candidates:
        _add(
            fallback_url
            or os.environ.get("TALISHAR_URL", "").strip()
            or DEFAULT_TALISHAR_URL
        )
    return tuple(candidates)


def _resolve_probe_deck_stems(assets_dir: Path) -> tuple[str, str]:
    """Pick two playable on-disk deck stems for a game-start health probe."""
    stems: list[str] = []
    for path in sorted(assets_dir.glob("fab_*.txt")):
        try:
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
        except OSError:
            continue
        if not lines:
            continue
        stems.append(path.stem)
        if len(stems) >= 2:
            break
    if not stems:
        return "", ""
    if len(stems) == 1:
        return stems[0], stems[0]
    return stems[0], stems[1]


def probe_backend_game_start(
    url: str,
    *,
    assets_path: str | Path | None = None,
    game_format: str = "silver_age",
    timeout: float = 15.0,
) -> tuple[bool, str]:
    """Probe *url* with CreateLocalGame + Start.php (fast_reset) using a real deck."""
    if os.environ.get("FAB_SKIP_TALISHAR_HEALTH_CHECK", "").strip() in {
        "1",
        "true",
        "yes",
    }:
        return True, "skipped"

    if assets_path is not None:
        assets_dir = Path(assets_path).expanduser().resolve()
    else:
        try:
            from fab_bridge.paths import talishar_assets_dir  # noqa: PLC0415

            assets_dir = talishar_assets_dir()
        except Exception:
            return False, "assets path unknown"

    p1_stem, p2_stem = _resolve_probe_deck_stems(assets_dir)
    if not p1_stem:
        return False, "no playable probe deck in Assets"

    from flesh_and_blood_rlbridge.talishar_engine_environment import (  # noqa: PLC0415
        TalisharEngineEnvironment,
    )

    env = None
    try:
        env = TalisharEngineEnvironment(
            base_url=normalize_talishar_url(url),
            local_deck_name=p1_stem,
            opponent_deck_name=p2_stem or p1_stem,
            game_format=game_format,
            self_play=True,
            max_turns=30,
            talishar_backend="fast",
            request_timeout=float(timeout),
        )
        env.fast_reset(seed=0)
        return True, "game_start"
    except Exception as exc:
        if is_shard_reset_error(exc):
            return False, str(exc)[:240]
        return False, f"probe failed ({exc!r})"[:240]
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def pick_healthy_eval_backend(
    *,
    fallback_url: str | None = None,
    assets_path: str | Path | None = None,
    timeout: float = 15.0,
    game_format: str = "silver_age",
    exclude_urls: Iterable[str] = (),
) -> tuple[str, dict[str, str]]:
    """Return the first backend that passes RLStep and game-start probes."""
    excluded = {
        normalize_talishar_url(url)
        for url in exclude_urls
        if str(url).strip()
    }
    candidates = [
        url
        for url in resolve_eval_backend_candidates(fallback_url=fallback_url)
        if normalize_talishar_url(url) not in excluded
    ]
    if not candidates:
        raise RuntimeError("No Talishar eval backend candidates configured")

    preferred = normalize_talishar_url(candidates[0])
    status: dict[str, str] = {}
    for url in candidates:
        ok_rl, reason_rl = probe_backend_health(url, timeout=min(float(timeout), 5.0))
        if not ok_rl:
            status[url] = reason_rl
            continue
        ok_start, reason_start = probe_backend_game_start(
            url,
            assets_path=assets_path,
            game_format=game_format,
            timeout=timeout,
        )
        if ok_start:
            status[url] = "ok"
            chosen = normalize_talishar_url(url)
            if chosen != preferred:
                print(
                    f"  Eval backend: using {chosen} "
                    f"(preferred {preferred} unavailable: {status.get(preferred, '?')})",
                    flush=True,
                )
            return chosen, status
        status[url] = reason_start

    detail = "; ".join(f"{url} ({status.get(url, '?')})" for url in candidates)
    raise RuntimeError(
        "No Talishar backend passed eval game-start probe. Tried: " + detail
    )


def build_eval_backend_pool(
    *,
    fallback_url: str | None = None,
    assets_path: str | Path | None = None,
    game_format: str = "silver_age",
) -> TalisharBackendPool:
    """Build an eval failover pool with a healthy primary URL first."""
    chosen, _status = pick_healthy_eval_backend(
        fallback_url=fallback_url,
        assets_path=assets_path,
        game_format=game_format,
    )
    candidates = resolve_eval_backend_candidates(fallback_url=fallback_url)
    ordered = [chosen] + [url for url in candidates if normalize_talishar_url(url) != chosen]
    return TalisharBackendPool(urls=tuple(ordered))


def reconcile_reserved_shard_urls(
    training_env: dict[str, str],
    *,
    healthy_training_urls: Iterable[str],
    assets_path: str | Path | None = None,
    game_format: str = "silver_age",
) -> dict[str, str]:
    """Repoint reserved eval/render URLs when game-start probes fail at stack startup."""
    updated = dict(training_env)
    training = [
        normalize_talishar_url(url)
        for url in healthy_training_urls
        if str(url).strip()
    ]

    for key in ("TALISHAR_EVAL_URL", "TALISHAR_RENDER_URL"):
        url = updated.get(key, "").strip()
        if not url:
            continue
        norm = normalize_talishar_url(url)
        ok, reason = probe_backend_game_start(
            norm,
            assets_path=assets_path,
            game_format=game_format,
        )
        if ok:
            print(f"  {key}: game-start probe ok ({norm})", flush=True)
            continue

        print(
            f"  WARNING: {key} failed game-start probe ({norm}): {reason}",
            flush=True,
        )
        if key != "TALISHAR_EVAL_URL":
            continue

        replacement: str | None = None
        for candidate in training:
            if candidate == norm:
                continue
            ok_candidate, _ = probe_backend_game_start(
                candidate,
                assets_path=assets_path,
                game_format=game_format,
            )
            if ok_candidate:
                replacement = candidate
                break

        if replacement is None:
            try:
                replacement, _ = pick_healthy_eval_backend(
                    fallback_url=training[0] if training else norm,
                    assets_path=assets_path,
                    game_format=game_format,
                    exclude_urls=(norm,),
                )
            except RuntimeError:
                replacement = None

        if replacement:
            updated[key] = replacement
            print(
                f"  Repointed {key} → {replacement} (game-start failover)",
                flush=True,
            )

    return updated


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
    _failure_streak: dict[str, int] = field(default_factory=dict, repr=False)

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
        with self._lock:
            if not self.urls:
                return DEFAULT_TALISHAR_URL
            return self.urls[int(worker_index) % len(self.urls)]

    def note_shard_success(self, url: str) -> None:
        """Reset the consecutive failure counter for a healthy backend."""
        normalized = normalize_talishar_url(url)
        with self._lock:
            self._failure_streak.pop(normalized, None)

    def note_shard_failure(self, url: str) -> bool:
        """Record a connection failure; evict the shard when the streak is exceeded.

        Returns ``True`` when the shard was evicted from the pool.
        """
        normalized = normalize_talishar_url(url)
        threshold = shard_eviction_threshold()
        with self._lock:
            if normalized not in {normalize_talishar_url(u) for u in self.urls}:
                return False
            streak = self._failure_streak.get(normalized, 0) + 1
            self._failure_streak[normalized] = streak
            if streak < threshold:
                return False
            self._evict_unlocked(
                normalized,
                reason=f"{streak} consecutive connection failure(s)",
            )
            return True

    def pick_replacement(
        self,
        failed_url: str,
        *,
        worker_index: int | None = None,
    ) -> str | None:
        """Return another healthy backend URL, or ``None`` when none remain."""
        failed = normalize_talishar_url(failed_url)
        with self._lock:
            candidates = [
                url
                for url in self.urls
                if normalize_talishar_url(url) != failed
            ]
            if not candidates:
                return None
            if worker_index is not None:
                return candidates[int(worker_index) % len(candidates)]
            return candidates[0]

    def _evict_unlocked(self, url: str, *, reason: str) -> None:
        normalized = normalize_talishar_url(url)
        remaining = tuple(
            backend
            for backend in self.urls
            if normalize_talishar_url(backend) != normalized
        )
        if len(remaining) == len(self.urls):
            return
        if not remaining:
            return
        object.__setattr__(self, "urls", remaining)
        self._failure_streak.pop(normalized, None)
        print(
            f"  WARNING: evicted unhealthy Talishar shard {normalized} ({reason}); "
            f"{len(remaining)} backend(s) remain",
            flush=True,
        )
        try:
            from fab_bridge.unified_training_debug import (  # noqa: PLC0415
                log_event,
                shard_label,
            )

            log_event(
                "connection",
                "Evicted unhealthy Talishar shard",
                url=normalized,
                shard=shard_label(normalized),
                reason=reason,
                remaining_shards=[shard_label(backend) for backend in remaining],
            )
        except Exception:
            pass

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
