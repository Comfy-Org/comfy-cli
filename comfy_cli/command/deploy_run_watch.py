"""Bound the synchronous deployment watcher without changing its frozen state machine."""

from __future__ import annotations

import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from threading import Thread

from comfy_cli.deploy_events import JobEventCallbacks, JobWatchRequest, JobWatchResult, watch_job


@dataclass(frozen=True, slots=True)
class DeployRunTimeoutError(Exception):
    seconds: float

    def __str__(self) -> str:
        return f"deploy job did not finish within {self.seconds:g} seconds"


def watch_with_timeout(request: JobWatchRequest, timeout: float, sleep_fn=time.sleep) -> JobWatchResult:
    result: Future[JobWatchResult] = Future()

    def worker() -> None:
        try:
            result.set_result(watch_job(request, JobEventCallbacks(), sleep_fn))
        except Exception as error:  # noqa: BLE001
            result.set_exception(error)

    Thread(target=worker, name="comfy-deploy-watch", daemon=True).start()
    try:
        return result.result(timeout=timeout)
    except FutureTimeoutError as error:
        raise DeployRunTimeoutError(timeout) from error
