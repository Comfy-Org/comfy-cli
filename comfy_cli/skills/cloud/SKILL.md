---
name: comfy-cloud
description: Drive Comfy Cloud from the comfy CLI — sign-in via OAuth, pin a custom base URL, route commands with --where cloud, and submit / track / collect workflow runs against a hosted ComfyUI.
---

`comfy` talks to two backends: a local ComfyUI on `127.0.0.1:8188`, or **Comfy Cloud**. This skill is everything cloud-specific.

## The one-time setup

```bash
# (Optional) point at a non-prod env — a PR preview, staging, etc.
comfy cloud set-base-url https://fe-pr-12159.testenvs.comfy.org

# Sign in. Opens a browser; completes via OAuth 2.1 + PKCE.
comfy cloud login

# Confirm.
comfy --json cloud whoami
```

`cloud set-base-url` persists the URL in the CLI config so future `cloud login` and `--where cloud` calls use it. Override transiently with the `COMFY_CLOUD_BASE_URL` env var.

Default base URL (no config, no env): `https://cloud.comfy.org`.

## Routing

Routing follows the precedence documented in the core skill. Once you've
run `comfy cloud login` or set an API key, the CLI auto-detects cloud
credentials and routes there by default — no `--where cloud` needed.

## Submitting a workflow on cloud

```bash
# Async (default) — returns prompt_id immediately, track separately with jobs watch.
comfy --json run --workflow flux.json --where cloud

# Synchronous — blocks until done, returns outputs in the same envelope.
comfy --json run --workflow flux.json --where cloud --wait
```

Two cloud-specific behaviours vs. local:

1. The CLI auto-injects `extra_data.api_key_comfy_org` / `auth_token_comfy_org` into the submitted prompt. Partner-API nodes (BFL, Gemini, etc.) need this — the web UI sends it automatically.
2. The cloud path uses HTTP polling for status (no WebSocket exposed). `jobs watch` polls `/api/job/<id>/status` every `--poll-interval` seconds.

## Tracking jobs on cloud

```bash
comfy --json jobs ls --where cloud                 # /api/jobs
comfy --json jobs status <prompt_id> --where cloud # /api/job/<id>/status + /api/history_v2/<id>
comfy --json-stream jobs watch <prompt_id> --where cloud --poll-interval 1.5
```

`jobs watch` emits NDJSON state-transition events (one per line) and exits when the job hits `completed` or `error`. Backgrounding works as you'd expect:

```bash
PID=$(comfy --json run --workflow X.json --where cloud | jq -r .data.prompt_id)
comfy --json-stream jobs watch "$PID" --where cloud > "$PID.events" &
```

## Output URLs

`run` and `jobs status` return `outputs: [...]` of fetchable URLs. Cloud URLs require the bearer token:

```bash
curl -H "Authorization: Bearer $TOKEN" "<output-url>" -o image.png
```

`/view` and `/api/view` both work on cloud — both go to the same handler.

## What lives where

| Stored in | What |
|---|---|
| `~/.config/comfy-cli/config.ini` (or platformdirs equivalent) | `cloud_base_url` (set via `cloud set-base-url`) |
| `~/.config/comfy-cli/credentials.json` (OS keychain on supported platforms) | OAuth session: `access_token`, `refresh_token`, `expires_at`, `base_url` |
| `$COMFY_CLOUD_BASE_URL` (env) | Per-shell override of the base URL |
| `$COMFY_CLOUD_API_KEY` (env, alt path) | Hidden API-key auth bypass for service accounts / testing |

Precedence for base URL: env var > config > default.

## Common gotchas

**"Login takes me to the wrong domain."** `comfy cloud login` reads the base URL fresh on every invocation. If the browser opens at `cloud.comfy.org` when you wanted a PR env, run `comfy cloud set-base-url <url>` first.

**"`comfy run --where cloud` returns 401 with XML."** That XML is an object-storage error, not a ComfyUI error — the request hit a CDN catch-all because the wrong path was used. Cloud ComfyUI endpoints live under `/api/*`. If you see this from the comfy CLI itself, it's a bug; from your own scripts, prepend `/api`.

**"401 `invalid auth token`."** Your OAuth token's `aud` claim doesn't match what `/api/prompt` expects (`comfy-cloud`). Decode the token (`jwt.io` or python `base64.urlsafe_b64decode`) and check `aud`. Cause: the OAuth resource you logged in against doesn't grant `comfy-cloud` audience. Fix: ask the cloud team to register the `comfyui` resource, or change the OAuth resource your CLI requests.

**"Session expired mid-run."** Tokens are short-lived (~1 hour). The CLI auto-refreshes on 401 (retries once after refreshing the OAuth token), but if the refresh token itself has expired, re-run `comfy cloud login`.

**"Pinned an env, but the old session is still active."** `cloud set-base-url` clears an expired session but warns (and keeps) a still-valid one. Run `comfy cloud logout` then `cloud login` for a clean swap.

**"How do I know which cloud I'm talking to?"** Always available via `comfy --json cloud whoami`. The `data.base_url` field is always populated and canonical (it reflects the active session when you have one, otherwise the configured default). `data.session.base_url` is also present when you're signed in via OAuth.

**"`whoami` says `signed_in: false` but commands still work."** That meant "no OAuth session" — outdated. After the recent fix, `signed_in` is true if *any* auth path is configured. Inspect `auth_method` (`"oauth"`, `"api_key"`, or `null`) and `api_key_source` (`"env"`, `"store"`, or `null`) to know which.

## Debugging cloud-specific failures

Read [comfy-debug] for the full envelope playbook. The cloud-flavored short version:

- `cloud_not_configured` → `comfy cloud login`
- `cloud_unauthorized`   → session expired, re-login
- `cloud_http_error` HTTP 401 → audience / token mismatch (see "Common gotchas")
- `cloud_http_error` HTTP 404 with XML body → wrong path (`/api/*` is the real prefix)
- `cloud_timeout` → raise `--timeout`, or rely on the default async submit and tail with `jobs watch`

## What you can NOT do today on cloud (and the local fallback)

| Not available on cloud | Local alternative |
|---|---|
| Real-time WebSocket events for `jobs watch` | ✅ available locally |
| Auto-convert UI-format workflows (`/workflow/convert`) | ✅ available locally |
| Custom-node installation | ✅ `comfy node install` (local) |
| Long sessions past refresh-token expiry | ✅ no auth needed locally |

Cloud excels at: heavy models you don't want to host, multi-GPU jobs, and being reachable from a CI runner. Local excels at: fast iteration, custom nodes, debugging, full-control over the workflow graph.
