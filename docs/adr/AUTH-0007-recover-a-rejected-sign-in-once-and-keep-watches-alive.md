# ADR-AUTH-0007: Recover a Rejected Sign-In Once, and Keep Watches Alive

Date: 2026-11-04

## Status

Proposed

Relation: Builds on [ADR-AUTOMATION-0003](AUTOMATION-0003-one-versioned-machine-output-dialect.md).
One envelope per command, append-only error codes and human diagnostics off
stdout all stay as they are. This ADR adds one error code, `watch_auth_lost`,
and fixes what it means.

## Context

A Comfy Cloud sign-in is one stored OAuth session: an access token that lasts
about fifteen minutes and a refresh token that rotates on every use. The
server tracks each refresh token as part of a family. Sending a refresh token
that was already used is treated as theft: the server kills the whole family,
and the user has to sign in again.

Several `comfy` processes can share that one stored session at once: a
`comfy build release create --watch` in one terminal, a `comfy deploy status
--watch` in another, an agent running `comfy deploy up` in a third. Each of
them can get a 401 when its access token expires mid-wait. Before this
decision, two failure modes followed from that:

- Two processes rejected on the same old access token could both refresh. The
  second one either replayed a consumed refresh token (and killed the family)
  or rotated the token the first had just stored, so every peer refreshed in
  turn.
- A watch over work the server had already accepted (a release that is
  building, a deployment that is coming up) ended with `build_not_signed_in`
  or `deploy_not_signed_in` and exit 1 when its recovery failed. That told the
  user to sign in again and suggested the work had failed, while the release
  or deployment was still running, and the stored sign-in was often fine.

Who owns the token family, when a process may refresh it, and what a watch
reports when it loses its sign-in are contracts that later auth or watch code
must keep. They belong in an ADR, not only in code comments.

## Decision

### One owner of the token family at a time

- Only a process holding the cross-process secrets lock
  (`auth_store.lock_path()`) may refresh or write the stored session. The lock
  wait is bounded at 90 seconds (`_REFRESH_LOCK_TIMEOUT_S`).
- Under the lock the process re-reads the store and decides from what is
  stored now, never from what it read before locking. It never sends a
  refresh token it observed before taking the lock if the store has since
  moved on.
- If the lock wait times out, the process does not refresh and does not
  re-read the store. It returns the session it held before locking and
  leaves the stored session untouched.

### What a process rejected with a 401 does

A request refused with a 401 passes the access token that was refused
(`rejected_access_token`) into the shared refresh. Under the lock:

1. **Adopt a usable peer session.** If the stored access token differs from
   the rejected one and has a known, unexpired `expires_at` (outside the usual
   leeway), a peer already refreshed. Use that session and do not refresh.
2. **Otherwise refresh the current family.** A stored token with no
   `expires_at`, an expired one, or the rejected token itself is not usable.
   The process refreshes with the refresh token stored now. A token whose expiry
   is unknown is never adopted, because the one retry would just get another 401.
3. **Proactive callers** (a process refreshing before expiry, with no 401) keep
   the earlier rule: any peer rotation since their pre-lock read means the
   family already moved, and they use the stored session.

### When the stored session is cleared

- Only an error that means the family is dead (`invalid_grant`,
  `invalid_token`, or a reuse-detection signal) clears the stored session, and
  only if the caller did not pass `allow_clear=False`. Background read-mostly
  callers, such as the detached `comfy run` watcher, pass that.
- Any other refresh failure (network, 5xx, timeout, `invalid_client`) never
  clears it. The old session stays and the caller's request surfaces its 401.

### One retry per rejected request

`BuilderClient._send` and the deploy API client retry a 401 at most once, and
only for a client built from the stored sign-in. The order is: 401, shared
recovery, one retry with the new token, and if that is refused too, the
command reports the refusal. A client handed a token (`COMFY_BUILDER_TOKEN`,
`COMFY_CLOUD_AUTH_TOKEN`) or a workspace API key never refreshes and never
retries. The server refuses a 401 before acting, so the retry is safe for
every method; a retried create carries the same idempotency key.

### What a watch reports when it loses its sign-in

A watch that reads work the server already accepted ends with `watch_auth_lost`
and exit code **75** (`EX_TEMPFAIL`, the same exit as `deploy_watch_lost`),
not 1, when all of these hold:

- the read got a 401 after shared recovery and the one retry;
- the client was built from the stored sign-in (it has a `sign_in_base_url`);
- a sign-in that can still recover remains stored: it has an access token, a
  refresh token, and the same `base_url` as the client's.

That re-read (`credentials.sign_in_still_stored`) takes the secrets lock with
`timeout=0`. If a peer holds the lock, the answer is "still stored": being
locked out does not prove the sign-in is gone, and the watch must not hang
for another lock budget after recovery already timed out. If the store
cannot be read at all (`OSError`), the answer is "not stored".

The envelope says only the watch ended: the message says the release may still
be building or the deployment may still be coming up. `details.releaseId`
(build) or `details.deployment_id` (deploy) names the work. The hint names the
command that reads it again (`comfy build release show <id>`, or `comfy deploy
status --deployment <id> --watch`) and says the stored sign-in was preserved.
The watch never clears the stored sign-in on this path.

Every other 401 keeps its sign-in code and its "sign in again" or "replace
the key" hint: a 401 before the server accepted anything, a 401 on an API key
or a handed token, or a 401 with no recoverable sign-in left.
`build_not_signed_in` and `deploy_not_signed_in` keep their meaning.

## Alternatives considered

- **Always refresh after a 401.** This is the simplest rule, and it is the bug:
  N processes rejected on the same old token rotate the family N times, and a
  late one replays a consumed token and kills the family.
- **Adopt any stored token that differs from the rejected one.** Rejected
  because a peer token with no recorded expiry, or one that has since expired,
  gets a second 401 on the only retry. This case was raised in review and is
  handled by the expiry check in rule 1.
- **Retry until a request succeeds.** Rejected because a server that keeps
  refusing a valid-looking token would loop forever, and each loop spends a
  refresh. One retry covers the real case, a token that expired mid-wait.
- **Wait on the lock again to re-read the store before classifying a watch
  401.** Rejected because after a 90 second timeout a second wait can hang the
  watch for as long again, or forever with no timeout. A non-blocking read
  answers the only question the classification needs.
- **Reuse `deploy_watch_lost`, or add a separate code for build.** Rejected
  because `deploy_watch_lost` means the service stopped answering, which needs a
  different next step. One `watch_auth_lost` covers both build and deploy
  watches, with the same exit code.
- **Keep `build_not_signed_in` / exit 1 for a lost watch.** Rejected because it
  makes callers and agents resubmit work that is still running, and sign in
  again when the stored sign-in is fine.

## What we reject

- Refreshing, or writing the stored session, outside the secrets lock.
- Sending a refresh token observed before the lock after the store has moved on.
- Treating a token with unknown expiry as usable for a reactive adoption.
- More than one retry per rejected request, or any retry for a key or a handed
  token.
- Clearing the stored session on a refresh failure that does not mean the
  family is dead, or from a watch that ends with `watch_auth_lost`.
- Waiting on the secrets lock with no timeout on any path that runs after a
  401.
- Exit 1 for a watch that ended while the server-side work may still be
  running.
- Reusing `watch_auth_lost` for a failure before the server accepted the work.

## Consequences

### Positive

- Parallel commands on one sign-in refresh once, not once each, and a late 401
  no longer kills the token family.
- A watch that loses its sign-in tells the truth: the work may still be
  running, here is how to read it, and the sign-in was kept. Exit 75 lets
  scripts tell "watch ended" from "work failed".
- Build and deploy watches report the same code and exit for the same
  situation.

### Negative

- `watch_auth_lost` is a new public error code. Under ADR-AUTOMATION-0003 it
  is append-only once released and cannot be renamed or repurposed.
- A watch can end with `watch_auth_lost` when a peer held the lock and the
  sign-in was in fact about to be lost. The next command then reports the
  sign-in code. That is the cost of not hanging.
- Callers that matched on `build_not_signed_in` / `deploy_not_signed_in` to
  detect a lost watch now see a different code and exit 75.

### Known limits

- The lock is a file lock on one machine. Two machines sharing one stored
  session (a synced home directory) are not coordinated.
- The tests cover real temporary credential storage and a real OS lock, with
  mocked HTTP. They do not cover a live OAuth server's reuse detection under
  concurrency.

## Validation

- `tests/comfy_cli/auth/test_oauth.py`: a rejected older token adopts a peer's
  unexpired token; it does not adopt an expired peer token or one with no
  expiry, and refreshes with the stored family instead.
- `tests/comfy_cli/test_session_refresh.py`: overlapping clients reuse the peer
  refresh before retrying; one retry, then a second 401 surfaces; handed tokens
  never swap; the retried create keeps its idempotency key; build and deploy
  watches end as `watch_auth_lost` with the sign-in kept; a watch does not wait
  again when a peer outlives the refresh lock; unrefreshable, signed-out and
  pre-accept 401s keep the sign-in code.
- `tests/comfy_cli/command/test_deploy_up.py` and `test_deploy_status.py`: the
  follow read's 401 with and without a stored sign-in, and a 401 on the create.
- The error-code registry tests keep `watch_auth_lost` registered and emitted
  from the same places.

## Revisit triggers

- The server starts returning a distinct status or error for "token expired"
  versus "token revoked". The adopt-or-refresh rule could then read it
  directly.
- Access-token lifetime moves well past fifteen minutes, or the server stops
  rotating refresh tokens. One retry and the family-ownership rule may then
  be more than is needed.
- Field reports of `watch_auth_lost` followed by a work item that had in fact
  failed, or of family-kill re-logins with this rule in place.
- Session storage moves off the local file store (for example to an OS
  keychain with its own locking).
