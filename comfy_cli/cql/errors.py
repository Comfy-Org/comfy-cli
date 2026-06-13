"""Loader exception types.

``CQLRuntimeError`` is a standalone exception for the Python-side CQL
loader / normaliser.  It carries a ``details`` dict for structured error
envelopes, matching the interface that callers expect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CQLRuntimeError(Exception):
    """Error from the Python-side CQL loader / normalizer."""

    runtime_message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize the Exception base with the message so str(e) works.
        Exception.__init__(self, self.runtime_message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.runtime_message

    def as_details(self) -> dict[str, Any]:
        return {**self.details, "runtime_message": self.runtime_message}
