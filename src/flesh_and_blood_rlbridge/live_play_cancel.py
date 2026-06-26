"""Shared cancellation exception for embedded GUI live play."""


class LivePlayCancelled(Exception):
    """Raised when a live play session is stopped by the user."""
