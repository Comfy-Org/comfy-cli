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
SDK-free at module scope.

**Auth:** the SDK sends the key as ``Authorization: Bearer``. Cloud's
``comfy_firebase_auth.go`` accepts ``comfyui-`` keys in either header, so the
``X-API-Key`` split that ``client._auth_headers`` performs is not needed here.
"""

from __future__ import annotations

import os

from comfy_cli.command.generate.client import ApiError


def router_models(api_key: str, *, timeout: float | None = None):
    """Return the ``models`` API of a Router-backed ``comfy-sdk`` client.

    Imports ``comfy_sdk`` lazily (see the module docstring); a missing SDK is
    wrapped into an :class:`ApiError` telling the user how to recover rather
    than surfacing a bare ``ImportError``.
    """
    # The SDK's ``Comfy()`` has no base-URL kwarg and resolves the Router host
    # from ``COMFY_ROUTER_BASE_URL`` only, so bridge a custom ``COMFY_API_BASE_URL``
    # deployment onto it when the Router var is unset.
    # TODO: drop once comfy-sdk exposes router_base_url=
    if not os.environ.get("COMFY_ROUTER_BASE_URL"):
        api_base_url = os.environ.get("COMFY_API_BASE_URL")
        if api_base_url:
            os.environ["COMFY_ROUTER_BASE_URL"] = api_base_url

    try:
        from comfy_sdk import Comfy
    except ImportError as exc:
        raise ApiError(
            0,
            "",
            "comfy generate needs comfy-sdk; run `pip install -U comfy-cli`",
        ) from exc

    # ``Comfy(timeout=None)`` disables the SDK's own 30s default, so only pass an
    # explicit timeout through.
    if timeout is None:
        return Comfy(api_key=api_key).models
    return Comfy(api_key=api_key, timeout=timeout).models
