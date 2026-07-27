"""Partner-API credential resolution for the local exec path.

Cloud auto-injects credentials at submit time; local doesn't. This module
finds the first usable credential (OAuth session, env var, or stored API key)
so the local exec can hand it to ``submit_prompt`` for nodes that talk to
partner APIs.

The OAuth session is refreshed when possible but never cleared from this
best-effort path (``refresh=True, allow_clear=False``): access tokens are
short-lived, so a signed-in user's token routinely lapses between commands;
refreshing it here keeps local runs working, while ``allow_clear=False``
guarantees a fatal refresh error can't log the user out from under a
foreground command. A refresh that can't succeed falls through to the
env/stored-key tail exactly as before.
"""

from __future__ import annotations


def _resolve_partner_credential() -> tuple[str, str] | None:
    """Locate an ``api_key_comfy_org`` credential to inject into a local
    submit's ``extra_data`` so partner-API nodes can authenticate.

    Returns ``(extra_data_key, value)`` so the caller can write the
    correct field for the credential type. Delegates to the shared
    OAuth-first chain (``comfy_cli.credentials``), same precedence as the
    cloud target resolution:

    1. Active OAuth session → ``auth_token_comfy_org``
    2. ``COMFY_CLOUD_API_KEY`` env var → ``api_key_comfy_org``
    3. Stored ``comfy-cloud-api-key`` provider record → ``api_key_comfy_org``

    The session is refreshed when possible (``refresh=True``): access tokens
    are short-lived by design, so a signed-in user whose token has lapsed since
    their last command would otherwise be skipped here and hit
    ``partner_node_requires_credential`` on every local run. ``allow_clear=False``
    keeps this best-effort injector from ever destroying the shared login: a
    fatal refresh error does not clear the stored session, and a transient
    refresh failure returns the stale session — which, once it is past the
    resolver's own 30s expiry leeway, falls through to env/stored key. (A token
    inside the 30–60s window is still returned as live; it is refreshed when the
    network allows and otherwise good for the imminent submit.) Concurrent
    ``comfy run`` fan-outs are safe under the OAuth refresh lock.

    Truly best-effort: the refresh path does network I/O plus a file-locked
    persist, and ``ensure_fresh_session`` only swallows the transient/timeout
    cases — an unexpected error (e.g. an ``OSError`` acquiring the lock or
    saving the rotated token) would otherwise propagate and abort the run. We
    catch it and fall through to a network-free read of env/stored keys, so this
    injector never turns a refresh hiccup into a failed ``comfy run``. Returns
    ``None`` when nothing usable is configured.
    """
    from comfy_cli.credentials import resolve_cloud_credential

    try:
        cred = resolve_cloud_credential(purpose="cloud", refresh=True, allow_clear=False)
    except Exception:  # noqa: BLE001 — best-effort: never abort the run on a refresh hiccup
        # refresh=False reads the store as-is (no network, no lock): an expired
        # session fails its own expiry check and the resolver returns env/stored
        # key — exactly the pre-BE-3361 behavior on this path.
        cred = resolve_cloud_credential(purpose="cloud", refresh=False, allow_clear=False)
    if cred is None:
        return None
    if cred.kind == "oauth":
        return ("auth_token_comfy_org", cred.value)
    return ("api_key_comfy_org", cred.value)
