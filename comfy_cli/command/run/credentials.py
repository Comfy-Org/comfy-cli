"""Partner-API credential resolution for the local exec path.

Cloud auto-injects credentials at submit time; local doesn't. This module
finds the first usable credential (active OAuth session, env var, or stored
API key) so the local exec can hand it to ``submit_prompt`` for nodes that
talk to partner APIs.
"""

from __future__ import annotations


def _resolve_partner_credential() -> tuple[str, str] | None:
    """Locate an ``api_key_comfy_org`` credential to inject into a local
    submit's ``extra_data`` so partner-API nodes can authenticate.

    Returns ``(extra_data_key, value)`` so the caller can write the
    correct field for the credential type. Delegates to the shared
    OAuth-first chain (``comfy_cli.credentials``), same precedence as the
    cloud target resolution:

    1. Active (non-expired) OAuth session → ``auth_token_comfy_org``
    2. ``COMFY_CLOUD_API_KEY`` env var → ``api_key_comfy_org``
    3. Stored ``comfy-cloud-api-key`` provider record → ``api_key_comfy_org``

    ``refresh=False`` preserves this chain's historical behavior: the session
    is read as-is, never refreshed here. Returns ``None`` when nothing usable
    is configured.
    """
    from comfy_cli.credentials import resolve_cloud_credential

    cred = resolve_cloud_credential(purpose="cloud", refresh=False)
    if cred is None:
        return None
    if cred.kind == "oauth":
        return ("auth_token_comfy_org", cred.value)
    return ("api_key_comfy_org", cred.value)
