# ADR-BUILD-0014: Prove local model bytes before reference reuse

Date: 2026-11-11

## Status

Proposed

## Context

Build push and pull reconcile local models before reusing a blobId or sourceUri.
A persisted path/size/mtime digest memo can answer with an old SHA-256 after a
same-size rewrite restores its mtime. Pull can then preserve the wrong local
identity, and push can publish a reference to bytes the file no longer holds.
Matching filesystem metadata does not establish content identity.

## Decision

Builds on [BUILD-VALIDATION-0008](BUILD-VALIDATION-0008-let-a-dry-run-check-public-pins-with-the-builder.md):
dry-run confirmation, public-pin validation, no-upload/no-save behavior and wire
contracts remain unchanged. Both build push and pull supply the existing streaming
SHA-256 hasher directly to local-model reconciliation on every attempt, including
dry runs and unchanged retries. Reuse a model reference only after a fresh hash
matches the digest recorded for it. Remove the private metadata-only memo; old
cache files are left untouched and no longer read or written by these commands.

The existing prepare_push copy semantics, stale-reference invalidation, upload
reuse, custom-node packaging, pull ownership and upload verification stay in place.
No caller-visible command, schema, hash algorithm or dependency is added.

## Alternatives considered

Additional stat fields such as ctime/inode can detect more changes but cannot
prove arbitrary byte equality across platforms and clock resolutions. Prefix,
suffix or sampled hashing can miss a middle edit. A trusted immutable-file version
could justify reuse without another read, but none exists in the current contract.

## What we reject

Do not let a persisted metadata-only digest, sampled fingerprint or cached blob
reference replace fresh local bytes as the authority for build reconciliation.
Do not turn a failed read into a cache hit or publish using a guessed identity.

## Consequences

Every push/pull attempt reads all referenced local model bytes again. This costs
I/O proportional to their total size; the existing chunked hasher bounds memory.
Unchanged content still reuses its reference without a duplicate upload. The
cache cannot preserve its previous cross-command no-read optimization safely.
This decision does not freeze a concurrently edited source file or strengthen
existing upload-time race guarantees; upload validation remains unchanged.

## Validation

Command-level tests cover preserved-mtime same-size rewrites, replacements,
changed middle chunks, stale blob/public-URL invalidation, pull and dry-run state,
forged legacy memo entries, read failure, and rehash-without-duplicate-upload.
Use fake services and local files; no production side effects are needed.

## Revisit triggers

Revisit no-read reuse only when a separately reviewed immutable-file/version
facility can prove that an identifier still names exactly the previously hashed
bytes. A faster metadata heuristic alone is insufficient.
