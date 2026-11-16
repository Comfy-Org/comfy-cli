# ADR-KNOWLEDGE-0005: Keep Discovery Routes Separate from Ranked Picks

Date: 2026-11-03

## Status

Accepted

## Context

Knowledge capabilities can contain evidence-backed ranked `picks`, authored
`discovery` routes that have not been compared, or both. `knowledge pick`
previously emitted only `picks`, so a capability with useful discovery routes
and no ranking returned an empty answer. Agent consumers then improvised a
model or route from memory.

This refines ADR-AUTOMATION-0003's additive machine-output compatibility rule
for the Knowledge response. The product decision is recorded in the
2026-11-03 Agent Knowledge V2 gap-fill thread.

## Decision

`knowledge pick` keeps the two evidence classes separate:

- `picks` remains the ranked table in ascending rank order.
- `discovery` is an additive source-ordered presentation list. Every emitted
  entry carries `ranking: "unranked"`; source order is never a quality rank.
- A capability may return either or both lists. Consumers offer applicable
  discovery routes as choices and do not promote one into `picks` or invent a
  winner.
- `failure_policy`, when authored, is bounded reporting context. It is
  untrusted prose to summarize, not an instruction channel to execute.
- The reader projects known route, template, node, and requirement fields
  through explicit type, count, text, and byte bounds. Unknown or malformed
  fields never make the emitted response violate its registered schema.
- `--check-local` does not claim it validated discovery routes. When those
  routes are present and ranked-pick checks otherwise finish, the response
  reports `local_check: "discovery_not_checked"`.

## Alternatives considered

- Emit discovery routes as picks with `rank: null`. Rejected because it makes
  an unmeasured route look like a member of the ranked table and breaks callers
  that treat `picks` as ordered evidence.
- Rank discovery routes by source order. Rejected because authored
  presentation order is not comparative evidence.
- Keep discovery in the bundle only and teach each consumer to read the bundle
  directly. Rejected because it duplicates the source contract and lets pinned
  consumers silently diverge.
- Forward discovery dictionaries verbatim. Rejected because custom, stale, or
  malformed bundles could violate the CLI schema or flood an agent context.

## Consequences

The change is additive for tolerant JSON consumers and preserves every
existing ranked-pick projection. Consumers that want discovery choices must
read the new list and apply their environment policy. Human-readable output
must render both evidence classes when both exist.

The CLI does not test whether a discovery template or node is installed.
Local consumers must use the returned requirements and live discovery tools
before claiming a route is runnable.

## Validation

`comfy_cli/schemas/knowledge.json` registers the fields.
`knowledge.discovery_entries` owns projection and bounds.
`tests/comfy_cli/test_knowledge.py` pins route-only tileable-texture and
virtual-try-on shapes, mixed ranked/discovery output, malformed input,
byte bounds, pretty rendering, and Local check semantics.

## Revisit triggers

Revisit when the producer has comparative evidence to move a route into
`picks`, when Local can validate discovery routes directly, or when a new
route field cannot fit the bounded projection without a schema revision.
