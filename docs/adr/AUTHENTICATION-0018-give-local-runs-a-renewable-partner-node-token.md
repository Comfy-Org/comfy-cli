# ADR-AUTHENTICATION-0018: Give Local Runs a Renewable Partner-Node Token

Date: 2026-11-20

## Status

Proposed

Relation: Builds on [ADR-AUTHENTICATION-0007](AUTHENTICATION-0007-recover-a-rejected-sign-in-once-and-keep-watches-alive.md).
Only a process holding the secrets lock writes the stored sign-in, and the
refresh rules there are unchanged.

## Context

A local `comfy run` put the sign-in's access token in
`extra_data.auth_token_comfy_org`. It lasts fifteen minutes and ComfyUI cannot
renew it, so a partner node still polling when it expired stopped after the
provider task was created: the user was charged and got no output.
Comfy Cloud mints a partner-node token from a live comfy-cli sign-in
(`POST /api/auth/token`, `{"resource":"partner-node"}`). It lasts about 90
minutes, ComfyUI renews it through comfy-api for its session's seven days, and
every mint creates a server-side session.

## Decision

- `comfy run` against a loopback ComfyUI, signed in with `comfy cloud login`,
  sends a partner-node token. One token is cached in the secrets store beside
  the sign-in, keyed by base URL, OAuth client, user, workspace and refresh
  family. A token cached for any other key is never sent. A refresh keeps it.
- Under the secrets lock, a run sends the cached token while it has 45 minutes
  left and renews it through comfy-api after that. A new one is minted when
  renewal answers 401/403, returns a token issued for less time than the one it
  renews (the session is ending), or fails with anything but a 429 for a token
  that no longer outlives a fresh sign-in token (15 minutes). A 401/403 drops
  the cached token whatever its expiry; a 429, a 5xx or a network failure sends
  it while it outlives a fresh sign-in token. A renewal stopped at the session's
  end is not cached, so the next renewal is judged against the full-lifetime
  token again, and is sent, if the mint fails, under the same 15-minute rule.
  No mint is made while the store cannot be written: the token would be lost,
  and every run would mint another.
- When no partner-node token can be sent (none is minted, a 429 leaves an
  expiring one, or anything fails), the run gets the sign-in token, refreshed
  first unless it is under a minute old. A sign-in that cannot refresh sends
  its access token as before. One resolution starts at most one refresh
  sequence: after it, whatever its outcome (a failure, or tokens that cannot be
  stored, `oauth.SessionNotSaved`, after one retry of the write), the rest of
  the resolution reads the store as it is.
- `comfy cloud logout` first takes the sign-in and its cached record into
  memory and clears them under the secrets lock; only then does it revoke, so
  an interrupted or slow revoke never leaves the sign-in stored. For an
  environment that issues partner-node tokens, the sign-in's access token,
  refreshed in memory when it has expired and never stored, revokes every
  session of that sign-in, whether or not a token was cached. That includes
  sessions a later mint replaced in the cache and the sessions of Cloud jobs
  submitted with that sign-in (ingest keys them by the same login), so logout
  also ends token renewal for those jobs; it matters only for a job whose API
  node runs more than 90 minutes after submit. A refused refresh, even
  `invalid_grant`, does not prove the sign-in ended, so it counts as an
  incomplete revocation. A cached token that the sign-in revoke did not cover
  (it failed or could not be made, or the record names another login) revokes
  its own session at its own environment. Logout reports which of these
  happened, and warns when only the cached token or nothing was revoked.
- A new `comfy cloud login`, the `comfy setup` sign-in and `comfy cloud
  set-base-url` clearing a stale sign-in save the new sign-in (or clear the old
  one) first, then revoke only the replaced sign-in's cached token, with that
  token, at its own environment. They do not revoke the whole sign-in, so the
  Cloud jobs it submitted keep renewing. With no cached token they make no
  request; a failed revoke warns.
- Only base URLs with a known comfy-api (prod, staging, test) get a token.

## What we reject

- Exchanging a token handed in through `COMFY_CLOUD_AUTH_TOKEN`. It is scoped
  to one request; a mint would turn it into a credential renewable for seven
  days.
- Sending the cached token to another machine. `comfy deploy run` keeps the
  sign-in token: the cached session is shared by every local run, and a
  deployment's worker runs the user's own code elsewhere. A ComfyUI on a
  non-loopback `--host` gets no partner-node credential at all, as on `main`:
  comfy-cli does not read object_info from such a host, so it sees no partner
  node. The loopback check at submit is kept as a guard, so the shared token
  does not depend on that object_info rule.
- Telling a loopback address from a tunnel. A loopback host behind an SSH
  tunnel, a port-forward or a proxy is the user's own setup and gets the
  partner-node token; `main` already sent the broader sign-in token there, and
  the partner-node token reaches only partner routes.
- Minting per run. Each mint writes a server-side session row.

## Consequences

- A local run gets at least 45 minutes of token in the normal case, and ComfyUI
  0.43.0 or later renews it during the run. An older ComfyUI still stops once
  that token expires.
- A new login revokes the replaced sign-in's cached token, so a local run still
  in flight with it cannot renew past its current token. Earlier sessions of
  that sign-in (a token a later mint replaced, or Cloud jobs) keep renewing
  until their sessions or the sign-in end. When the revoke fails, the login
  warns; it does not keep the replaced sign-in's credentials to retry, since
  holding them is a larger security trade-off.
- A process can wait on the secrets lock while another one mints or renews.

### Known limits

- Logout now makes a revoke request (and, for an expired access token, a
  refresh) for every sign-in to an environment that issues partner-node tokens,
  each with a 15-second timeout. The sign-in is already gone locally by then.
- A store that stays unwritable after a refresh keeps the spent refresh token;
  the next command sends it and the server ends the sign-in. Such a store cannot
  record any recovery state either.
- With the access token expired and its refresh refused or failing, only the
  cached token's own session is revoked, and logout says so; a session it
  replaced can renew until its sign-in ends.

## Validation

`tests/comfy_cli/cloud/test_partner_token.py` covers the cache key, reuse,
renewal, the refusals, logout, and the credentials that stay unchanged.

## Revisit triggers

- Comfy Cloud changes the partner-node token lifetime or the renew limit (30
  per hour per session).
- comfy-cli needs a partner-node token for an environment without a known
  comfy-api, or for `comfy deploy run`.
