"""Engine errors for unsupported card interactions."""

from __future__ import annotations

UNIMPLEMENTED_PREFIX = "UnimplementedError"


class UnimplementedError(Exception):
    """The engine recognized an interaction it cannot resolve faithfully."""


def unimplemented_message(detail: str) -> str:
    detail = detail.strip()
    if detail.startswith(UNIMPLEMENTED_PREFIX):
        return detail
    return f"{UNIMPLEMENTED_PREFIX}: {detail}"
