# ADR-TRANSFER-0012: Resume Model Uploads Only for the Same File and Account

Date: 2026-11-06

## Status

Proposed

Relation: Builds on [ADR-TRANSFER-0011](TRANSFER-0011-upload-only-the-named-model-and-prove-cloud-stored-its-bytes.md).
Approval, name-reuse checks, checking each hashing block before sending it, and
proving the stored hash all remain in effect. The success envelope is unchanged.

## Context

A user uploads a large checkpoint and their connection drops. Sending one PUT
again loses the bytes already sent. A storage session can continue that upload,
but its address is a credential and must not follow a later sign-in or changed
file into another upload.

## Decision

Both model upload commands request resumable uploads. A response without an
upload mode keeps the existing single-PUT path. The server decides the size
limit; the CLI does not keep a separate upper limit.

The CLI owns saved sessions in a private directory under its configuration
folder. An atomic replacement writes each session with user-only permissions.
At most 32 session records and 32 interrupted-save temporary files remain.
Sessions expire at the earlier of the server's expiry or seven days after
creation. Oversized, malformed, expired and insecure records are never reused.
On Windows, file access follows the user's configuration-directory ACL; POSIX
also checks ownership and mode 0600.

Reuse requires authenticated user and workspace ids from Cloud, the same server,
resolved path, folder, hashed size, modified time and freshly computed content
hash. Unknown identity disables persistence. A failure to save is reported and
still permits retries within that process.

Storage's status query decides where to resume. Each piece is at most 64 MiB.
A resumed offset inside a hashing block still checks that whole block before
sending any part of it. Only acknowledged bytes advance progress. Transport
errors keep no exception text because it can contain the session credential.
Expired sessions and permanent storage refusals discard the saved record and
restart once; thirty minutes without acknowledged progress pauses. Certificate
verification and certificate-bundle loading failures stop immediately with
credential-safe TLS guidance.
The legacy single-PUT path uses the same certificate classifier and guidance;
transient TLS failures there retain the existing bounded fresh-upload retry.
HTTP 408 and transient TLS failures query held bytes and retry within the same
session. TLS setup diagnostics use fixed text, including for normalized URLs.
Completion deletes the session after a valid task response, and the command
waits for the task without a total deadline. Repeated unanswered task polls
still have a bounded retry window. The task id is emitted before waiting, so
a caller can look it up if the command stops. HTTPS with a host and no userinfo
is required for resumable and saved sessions; legacy single-PUT addresses retain
the existing validation for compatibility with storage emulators.
Progress output failures, including Rich converting broken pipes to process-exit
exceptions, mute reporting without stopping task polling or the stored-hash check.
Later storage-wait notices use the same reporting boundary.
The shared reporter also keeps `comfy build push` running after a broken pipe
or closed progress stream; its upload protocol is unchanged.

## Alternatives considered

- Keep one PUT and raise only the size limit. A drop still restarts the file.
- Reuse state by path alone. This could send bytes for a different file,
  account, workspace or folder.
- Trust size and modified time alone. Files can change without either changing.
- Save session addresses in ordinary config or output. That exposes credentials.

## Consequences

Reruns hash the whole file again. Storage receives only newly needed pieces,
and no temporary copy of a multi-GB model is created. State is local to one
machine; simultaneous invocations are not coordinated and may start separate
uploads. Each new upload has a separate record, so completing one does not
delete another upload's saved state. Losing saved state loses the ability to
resume that session, but a new upload still works. A task may continue after
the command loses its sign-in.

## Validation

The model-upload tests use a fake bucket for mid-piece drops, process reruns,
final-piece recovery, refusal, expiry, transient errors and byte verification.
State tests cover each identity field, expiration, permissions and bounds.
Real Requests and OpenSSL check invalid local certificate bundles. Real Rich
rendering checks closed streams and broken pipes during storage notices and
polling. Tests isolate Rich's process-wide stdout redirection. Existing command
tests retain approval and stored-hash assertions. CI runs
these tests on Linux, macOS and Windows. Real multi-GB storage is not exercised
by the fake bucket.

## Revisit triggers

Revisit when storage changes its resumable protocol or lifetime, when users
need to resume across machines, or when concurrent uploads must share a session.
