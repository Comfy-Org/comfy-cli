"""Thin accessor for the Router-backed ``comfy generate`` models API.

The ``comfy-sdk`` ``Comfy`` client pulls in pydantic, which loads the compiled
``pydantic_core`` extension. A ``comfy`` process that has loaded it holds the
.pyd/.so open for its whole lifetime, and on Windows an open DLL cannot be
replaced — so if it were loaded on a path ``comfy install`` walks, uv could not
upgrade pydantic_core when ComfyUI's ``requirements.txt`` resolves a different
version (os error 5). ``generate`` is a ``LazyCommand`` in ``comfy_cli.cmdline``,
so the import stays reachable only from the Router ``generate`` path.

**Lazy-import rule:** ``comfy_sdk`` must be imported *only* inside a function,
never at module scope — not here and not anywhere else reachable from
``comfy_cli.cmdline``. ``tests/comfy_cli/test_tracking_lazy_import.py`` guards
this: importing this module (and ``generate.app``) must not load ``comfy_sdk``
or ``pydantic_core``. Importing this module from ``app.py`` is fine — it stays
SDK-free at module scope. The decision is recorded in
``docs/adr/DEPENDENCIES-0019-load-comfy-sdk-only-inside-the-router-generate-path.md``.

**Auth:** the SDK sends the key as ``Authorization: Bearer`` to Router — that is
the SDK's own contract with Router and cannot be changed from here. It is *not*
what the partner-API proxy accepts: comfy-api's ``comfy_firebase_auth.go`` reads
``comfyui-`` keys from ``X-API-Key`` only and validates a Bearer value as a
Firebase JWT, which is why ``client._auth_headers`` splits the two. Confirm Router
accepts a Bearer ``comfyui-`` key end to end when the first caller lands. The SDK
client also does not send the ``Comfy-Env`` / ``Comfy-Usage-Source`` /
``User-Agent`` headers that ``client._auth_headers`` adds.
"""

from __future__ import annotations

import contextlib
import os
import threading

from comfy_cli.command.generate.client import ApiError

# The bridge mutates the process environment, so overlapping calls must not
# interleave their guard / override / construction / restore.
_BRIDGE_LOCK = threading.Lock()


@contextlib.contextmanager
def _router_base_url_bridge(router_var: str):
    """Point the SDK's Router variable at ``COMFY_API_BASE_URL`` for one construction.

    The SDK's ``Comfy()`` has no base-URL kwarg and resolves the Router host from
    its Router variable at construction only, so a custom ``COMFY_API_BASE_URL``
    deployment is bridged onto it while ``Comfy()`` runs — when the Router
    variable is unset — and the process environment is restored afterwards so
    later calls and child processes see what the user set. The whole lifecycle
    runs under ``_BRIDGE_LOCK`` so overlapping calls each construct against the
    configured target.
    """
    # TODO: drop once comfy-sdk exposes router_base_url=
    with _BRIDGE_LOCK:
        # Normalised like ``spec.base_url()``; blank counts as unset.
        api_base_url = (os.environ.get("COMFY_API_BASE_URL") or "").strip().rstrip("/")
        if (os.environ.get(router_var) or "").strip() or not api_base_url:
            yield
            return
        previous = os.environ.get(router_var)
        os.environ[router_var] = api_base_url
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(router_var, None)
            else:
                os.environ[router_var] = previous


def router_models(api_key: str, *, timeout: float | None = None):
    """Return the ``models`` API of a Router-backed ``comfy-sdk`` client.

    Imports ``comfy_sdk`` lazily (see the module docstring); a missing SDK is
    wrapped into an :class:`ApiError` telling the user how to recover rather
    than surfacing a bare ``ImportError``, and an SDK that is installed but fails
    to import (e.g. a broken pydantic) names the real cause.
    """
    try:
        from comfy_sdk import ROUTER_BASE_URL_ENV_VAR, Comfy
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "comfy_sdk":
            msg = "comfy generate needs comfy-sdk; run `pip install -U comfy-cli`"
        else:
            msg = (
                f"comfy-sdk is installed but failed to import ({exc}); its install or a "
                "dependency such as pydantic is broken — reinstall it with "
                "`pip install --force-reinstall comfy-sdk`"
            )
        # The message is also the body: callers that render ``status`` + ``body``
        # for HTTP errors still show the recovery hint.
        raise ApiError(0, msg, msg) from exc

    with _router_base_url_bridge(ROUTER_BASE_URL_ENV_VAR):
        # ``Comfy(timeout=None)`` disables the SDK's own 30s default, so only pass
        # an explicit timeout through.
        if timeout is None:
            return Comfy(api_key=api_key).models
        return Comfy(api_key=api_key, timeout=timeout).models
