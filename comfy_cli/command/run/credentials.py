"""Partner-API credential resolution for the local exec path.

Cloud auto-injects credentials at submit time; local doesn't. This module
finds the first usable credential (env var, stored API key, or active
OAuth session) so the local exec can hand it to ``submit_prompt`` for
nodes that talk to partner APIs.
"""

from __future__ import annotations

import os


def _resolve_partner_credential() -> tuple[str, str] | None:
    """Locate an ``api_key_comfy_org`` credential to inject into a local
    submit's ``extra_data`` so partner-API nodes can authenticate.

    Returns ``(extra_data_key, value)`` so the caller can write the
    correct field for the credential type. OAuth-first, matching the cloud
    target resolution in ``target.resolve_target``:

    1. Active (non-expired) OAuth session → ``auth_token_comfy_org``
    2. ``COMFY_CLOUD_API_KEY`` env var → ``api_key_comfy_org``
    3. Stored ``comfy-cloud-api-key`` provider record → ``api_key_comfy_org``

    Returns ``None`` when nothing usable is configured.
    """
    from comfy_cli.auth import store as auth_store
    from comfy_cli.target import CLOUD_API_KEY_PROVIDER

    session = auth_store.get_cloud_session()
    if session is not None and not session.is_expired() and session.access_token:
        return ("auth_token_comfy_org", session.access_token)

    env_key = os.environ.get("COMFY_CLOUD_API_KEY")
    if env_key:
        return ("api_key_comfy_org", env_key)

    record = auth_store.get(CLOUD_API_KEY_PROVIDER)
    if record is not None and getattr(record, "key", None):
        return ("api_key_comfy_org", record.key)

    return None
