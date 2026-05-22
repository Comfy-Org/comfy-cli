"""Loader exception types.

``CQLRuntimeError`` inherits from ``ComfygraphError`` so consumers can catch
a single base type across both the Python loader and the WASM engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from comfy_cli.comfygraph import ComfygraphError


@dataclass
class CQLRuntimeError(ComfygraphError):
    """Error from the Python-side CQL loader / normalizer.

    Inherits from ``ComfygraphError`` so callers can ``except ComfygraphError``
    to handle both WASM and Python failures uniformly.
    """

    runtime_message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize the Exception base with the message so str(e) works.
        Exception.__init__(self, self.runtime_message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.runtime_message

    def as_details(self) -> dict[str, Any]:
        return {"runtime_message": self.runtime_message, **self.details}
