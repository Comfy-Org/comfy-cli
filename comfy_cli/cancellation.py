"""Process-wide cancellation token tied to SIGINT.

Why: long-running ops (workflow execution, cloud polling, partner generation)
must cancel cleanly on Ctrl+C — close the WebSocket, post a cancel-job to the
cloud, emit a final ``error.code = "cancelled"`` envelope, and exit with 130
(POSIX SIGINT convention).

Usage:
    token = install_sigint_handler()
    while not token.is_set():
        ...
    if token.is_set():
        renderer.error("cancelled", "Cancelled by user", exit_code=130)

The token is process-wide. `install_sigint_handler()` is idempotent.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CancellationToken:
    _event: threading.Event = field(default_factory=threading.Event)
    _callbacks: list[Callable[[], None]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancelled or timeout. Returns True if cancelled."""
        return self._event.wait(timeout=timeout)

    def cancel(self) -> None:
        """Mark cancelled and fire all registered callbacks.

        Callbacks run on the signal-handler thread (i.e., the main thread for
        SIGINT). They should be quick and non-blocking — typically just
        setting a flag or closing a socket.
        """
        with self._lock:
            already = self._event.is_set()
            self._event.set()
            callbacks = list(self._callbacks)
            self._callbacks.clear()
        if already:
            return
        for cb in callbacks:
            try:
                cb()
            except Exception:  # noqa: BLE001
                # Cancellation must never raise — losing one cleanup is
                # better than masking the original exit reason.
                pass

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register a one-shot cancel callback. If already cancelled, runs immediately."""
        with self._lock:
            if self._event.is_set():
                fire = True
            else:
                self._callbacks.append(callback)
                fire = False
        if fire:
            try:
                callback()
            except Exception:  # noqa: BLE001
                pass


_TOKEN: CancellationToken | None = None
_INSTALLED = False
_TOKEN_LOCK = threading.Lock()
_PREVIOUS_SIGINT_HANDLER = None


def get_token() -> CancellationToken:
    global _TOKEN
    if _TOKEN is None:
        with _TOKEN_LOCK:
            if _TOKEN is None:
                _TOKEN = CancellationToken()
    return _TOKEN


def install_sigint_handler() -> CancellationToken:
    """Idempotently install a SIGINT handler that cancels the token.

    The previous handler is preserved and re-raised after the token fires, so
    Python's default ``KeyboardInterrupt`` propagation still works.
    """
    global _INSTALLED, _PREVIOUS_SIGINT_HANDLER
    token = get_token()
    if _INSTALLED:
        return token

    previous = signal.getsignal(signal.SIGINT)
    _PREVIOUS_SIGINT_HANDLER = previous

    def _handler(signum: int, frame: object) -> None:
        token.cancel()
        # Chain to the previous handler so KeyboardInterrupt still raises.
        if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
            previous(signum, frame)
        else:
            # Default: raise KeyboardInterrupt in the main thread.
            raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _handler)
    _INSTALLED = True
    return token


def reset_for_testing() -> None:
    global _TOKEN, _INSTALLED, _PREVIOUS_SIGINT_HANDLER
    _TOKEN = None
    _INSTALLED = False
    if _PREVIOUS_SIGINT_HANDLER is not None:
        signal.signal(signal.SIGINT, _PREVIOUS_SIGINT_HANDLER)
        _PREVIOUS_SIGINT_HANDLER = None
