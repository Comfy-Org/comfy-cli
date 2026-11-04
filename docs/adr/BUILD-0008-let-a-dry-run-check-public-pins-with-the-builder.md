# ADR-BUILD-0008: Let a Dry Run Check Public Pins with the Builder

Date: 2026-11-04

## Status

Proposed

## Context

`comfy build push --dry-run` used to send no HTTP at all. It computed the
uploads from the local spec and stopped. The real push makes one check the
dry run did not: it sends the spec's public custom nodes (registry versions
and repositories) to the builder's snapshot importer, `POST
/v1/snapshots/resolve`, and refuses with `build_registry_pin_missing` when the
importer cannot place one of them. That is the only place that knows what the
Comfy Registry actually publishes and how a repository normalizes.

So a dry run could pass while the push of the same spec failed. The common
case was a registry version the registry has Flagged: the dry run said
"looks good", the push showed its plan, and was then refused. A dry run that
passes something the push will refuse is not a preview.

Making the dry run ask the builder changes what `--dry-run` promises: from "no
network, no sign-in" to "may make one authenticated read". That is a
command-level side-effect contract that scripts and agents rely on, so it is
recorded here.

## Decision

`comfy build push --dry-run` may make exactly one builder request, and only
for the pin check:

- **When.** Only if the spec has at least one public custom node, meaning an
  entry that is not `source: local` and names a `registryVersion` or a
  `repository` (`public_node_projection`). A spec whose packs are all local
  sends no request and needs no sign-in, as before.
- **What is sent.** Only the public-node projection, wrapped as a snapshot.
  Local packs, models, files, workflow content and upload plans are never sent.
- **Which call.** `resolve_snapshot`, the snapshot importer. It is a read: it
  returns a definition and a report, and writes nothing on the builder. The dry
  run makes no other builder call. It creates no blobs, uploads nothing, saves
  no build and cuts no release. It does not rewrite the local spec either.
- **Sign-in.** The request goes through the same authenticated builder client
  as a push (`_builder_client`): an injected token, a Cloud token or API key
  from the environment, or the stored sign-in. With none of these, a dry run
  with public pins fails with `build_not_signed_in`.
- **Parity with push.** Push and dry run share one function,
  `_require_public_pins`. They send the same payload, compare the same
  identities, and refuse with the same `build_registry_pin_missing` code,
  message and hint. A spec the dry run passes on this check is one the push
  passes on this check, against the same builder state.
- **Failure is terminal.** If the lookup itself fails (network, 5xx, refused
  credential), the dry run exits through `_builder_call` with the same error a
  push would get. It never treats an unanswered check as a pass.
- **No invented reasons.** The importer does not say why it could not place a
  pin. The CLI names the identities and does not claim "Flagged" or any other
  registry status.

The help text says "check public custom node pins with the builder; upload and
save nothing", and the `--release` / `--dry-run` conflict says the dry run
"saves nothing" instead of "sends nothing".

## Alternatives considered

- **Keep the dry run offline.** It keeps the old promise, but the dry run keeps
  passing specs the push refuses, which is the bug.
- **Check pins only with a flag (`--dry-run --check-pins`).** Rejected because
  the default dry run would still give the wrong answer, and the people and
  agents who need the check are the ones who will not know to ask for it.
- **Warn instead of refusing.** Rejected because the push refuses, and a
  preview that is softer than the real thing gets ignored.
- **Ask the public registry directly, without sign-in.** Rejected because it
  would be a second implementation of what the importer decides. Repository
  normalization and which versions the builder can place live in the builder,
  and the two would drift.
- **Fall back to passing when the lookup fails.** Rejected because "could not
  check" is not "checked and fine".

## What we reject

- Any dry-run builder call that writes: blob creation, uploads, saves,
  releases, or any endpoint not documented as read-only.
- Sending anything beyond the public-node projection on a dry run.
- A builder request, or a sign-in requirement, for a spec with no public pins.
- A second copy of the pin check that can drift from the push's.
- Treating a failed or unanswered lookup as a pass.

## Consequences

### Positive

- A dry run now fails on the pin the push would refuse, with the push's own
  code, before anyone acts on the plan.
- One function owns the check, so push and dry run cannot drift.
- All-local specs keep a zero-request, no-sign-in dry run.

### Negative

- A dry run on a spec with public pins now needs sign-in and network. A CI job
  that ran `--dry-run` with no credential will fail with `build_not_signed_in`
  until it sets `COMFY_CLOUD_API_KEY` or drops the public pins.
- The dry run now depends on the builder being up for those specs.
- `build_registry_pin_missing` can now come from a dry run. Its meaning is
  unchanged, so this stays within ADR-AUTOMATION-0003's rule that error codes
  are never repurposed.

### Known limits

- The answer is only as fresh as the builder's view when the dry run ran. A
  registry change between the dry run and the push can still make the push
  refuse.
- The dry run still reads no folder list from the builder, so it does not
  check model folder case. That stays as it was.

## Validation

- `tests/comfy_cli/command/test_build_push.py`: a dry run refuses a Flagged pin
  with the push's own error; it sends only the public pins to the importer and
  makes no other call; a failed lookup exits instead of passing; the pretty
  output says the pins were checked and nothing was uploaded or saved.
- `tests/comfy_cli/command/test_build_auth_matrix.py` and
  `build_auth_support.py`: a dry run with public pins needs a credential, and
  one with only local packs sends nothing.

## Revisit triggers

- The builder exposes a public, unauthenticated pin-check endpoint. The dry run
  could then drop the sign-in requirement.
- The importer starts returning a reason per pin it could not place. The
  refusal could then name it.
- Users report CI breakage from the new sign-in requirement often enough that
  an offline mode is worth adding back as an explicit option.
- The dry run needs a second builder read (for example the folder list). That
  widens this boundary and needs its own decision.
