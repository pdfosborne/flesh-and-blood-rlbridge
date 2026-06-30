"""Non-blocking checkpoint evaluation for play training."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

_eval_executor: Optional[ThreadPoolExecutor] = None
_eval_futures: list[Future[Any]] = []
_eval_futures_lock = threading.Lock()


def get_checkpoint_eval_executor() -> ThreadPoolExecutor:
    global _eval_executor
    if _eval_executor is None:
        _eval_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ckpt-eval",
        )
    return _eval_executor


def submit_checkpoint_eval(
    fn: Callable[[], Any],
    *,
    label: str = "",
    run_dir: Path | str | None = None,
) -> Future[Any]:
    """Run *fn* on the shared single-worker eval pool (training does not wait)."""
    lock_root = Path(run_dir).expanduser().resolve() if run_dir is not None else None

    def _wrapped() -> Any:
        try:
            if lock_root is not None:
                from fab_bridge.eval_shard_lock import hold_eval_shard  # noqa: PLC0415

                with hold_eval_shard(lock_root, "checkpoint_eval", label=label):
                    return fn()
            return fn()
        except Exception as exc:
            suffix = f" ({label})" if label else ""
            print(f"  [checkpoint-eval] async job failed{suffix}: {exc!r}")
            raise

    future = get_checkpoint_eval_executor().submit(_wrapped)
    with _eval_futures_lock:
        _eval_futures.append(future)
        # Drop completed futures so the list does not grow without bound.
        _eval_futures[:] = [f for f in _eval_futures if not f.done()]
    return future


def wait_for_checkpoint_evals(*, timeout: Optional[float] = None) -> None:
    """Block until all submitted checkpoint eval jobs finish."""
    with _eval_futures_lock:
        pending = list(_eval_futures)
    for future in pending:
        try:
            future.result(timeout=timeout)
        except Exception:
            pass
    with _eval_futures_lock:
        _eval_futures.clear()


def shutdown_checkpoint_eval_executor(*, wait: bool = True) -> None:
    global _eval_executor
    if wait:
        wait_for_checkpoint_evals()
    if _eval_executor is not None:
        _eval_executor.shutdown(wait=wait)
        _eval_executor = None
