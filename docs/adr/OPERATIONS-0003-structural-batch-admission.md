# ADR-OPERATIONS-0003: Contain destructive structural batches at producer admission

Date: 2026-11-11

## Status

Proposed

## Context

CLI combo rebuilds reset children, while the TypeScript shared consumer
intentionally preserves hidden values. A batch writing Gen's model.mode,
then switching model from max to base and back to max can therefore succeed
locally and replay inconsistently (#1101).

## Decision

Builds on [OPERATIONS-0002](OPERATIONS-0002-version-structured-workflow-operations.md):
the frozen vocabulary and consumer compatibility contract remain binding.
Narrow `apply_specs` admission to refuse a destructive rebuild after a write
to a schema-owned child on the same resolved target. Follow nested combo
ownership; only removed DynamicGroup rows conflict with a shrink. Use resolved
op identity rather than textual prefixes. Snapshot original workflow containers,
execute the producer once in place, and restore those containers on failure.
Shallow snapshots retain original child references, including containers detached
by deletion or rebuild. Refusal returns no ops and preserves caller state even
after an alias-producing add; successful work retains existing in-place behavior.
Report the failing spec and zero applied count.

Repeated `control_after_generate` marker names use the native writer's first
saved slot and its actual combo owner. Another same-named marker elsewhere in
the node does not establish a destructive dependency.

## Alternatives considered

Rejecting every structural edit would unnecessarily forbid no-ops, group growth,
unrelated nodes, and selector-before-child work. Prefix-only checks confuse
ordinary dotted scalar names with ownership. Changing replay would require a
separate cross-consumer design and compatibility agreement.

## What we reject

This repair does not add a replay algorithm, stamp barrier, or consumer changes.
The historical raw-op #1101 exception remains documented in the
[operation vocabulary](../op-vocabulary-v1.md).

## Consequences

Refused work is atomic, with a useful reordering/separate-batch hint. Successful
work retains ordinary validation and existing guards, but admission proves
only absence of this destructive within-batch ordering. It does not establish
arbitrary replay-order convergence, repair earlier batches, or reconcile CLI
reset behavior with the consumer's hidden-value preservation. Snapshot storage
and traversal cost scale with the workflow's containers and references. Identity
and op nonces are minted once per attempted producer operation; failed attempts
still consume nonces, which are not rolled back or reused. No lifetime-cost
improvement is claimed.

## Validation

`test_rebuild_batch_guard.py` covers unsafe success regressions, aliases,
rollback after add_node and deletion, held container identities, generated-ID
collisions, single op-ID allocation, interruption, CLI file bytes, nested owners,
interior target identity, no-ops, scalar dotted names, growth and shrink/regrowth. Existing batch-order
and DynamicGroup tests retain coverage of ordinary successful work.

## Revisit triggers

Revisit when a coordinated shared-consumer replay contract resolves #1101,
when schema ownership introduces new structural families, or when workflow
snapshot costs require measured optimization without weakening atomic refusal.
