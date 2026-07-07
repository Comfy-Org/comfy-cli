"""Cancellation token + SIGINT integration."""

import os
import signal
import threading
import time

import pytest

from comfy_cli import cancellation


@pytest.fixture(autouse=True)
def reset_token():
    cancellation.reset_for_testing()
    yield
    cancellation.reset_for_testing()


def test_token_starts_unset():
    t = cancellation.get_token()
    assert t.is_set() is False


def test_cancel_sets_event():
    t = cancellation.get_token()
    t.cancel()
    assert t.is_set() is True


def test_callbacks_fire_on_cancel():
    t = cancellation.get_token()
    seen = []
    t.on_cancel(lambda: seen.append("a"))
    t.on_cancel(lambda: seen.append("b"))
    t.cancel()
    assert seen == ["a", "b"]


def test_callback_registered_after_cancel_fires_immediately():
    t = cancellation.get_token()
    t.cancel()
    seen = []
    t.on_cancel(lambda: seen.append("late"))
    assert seen == ["late"]


def test_cancel_is_idempotent():
    t = cancellation.get_token()
    seen = []
    t.on_cancel(lambda: seen.append("x"))
    t.cancel()
    t.cancel()
    assert seen == ["x"]


def test_callback_exception_does_not_propagate():
    t = cancellation.get_token()

    def boom():
        raise RuntimeError("nope")

    t.on_cancel(boom)
    # Must not raise.
    t.cancel()
    assert t.is_set()


def test_wait_releases_on_cancel():
    t = cancellation.get_token()
    threading.Timer(0.05, t.cancel).start()
    got = t.wait(timeout=1.0)
    assert got is True


def test_sigint_handler_cancels_token():
    # Install the handler then send SIGINT to this process.
    t = cancellation.install_sigint_handler()
    raised = []
    try:
        os.kill(os.getpid(), signal.SIGINT)
        # Give the handler a moment to run.
        time.sleep(0.05)
    except KeyboardInterrupt:
        raised.append(True)
    assert t.is_set()
    assert raised == [True], "default chained handler should still raise"


def test_install_is_idempotent():
    t1 = cancellation.install_sigint_handler()
    t2 = cancellation.install_sigint_handler()
    assert t1 is t2
