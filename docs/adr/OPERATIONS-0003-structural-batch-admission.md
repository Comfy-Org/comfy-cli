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
op identity rather than textual prefixes. Validate on an isolated graph and
spec list so refusal returns no ops and preserves caller state even after an
alias-producing add. After successful validation, execute the existing producer
path on the caller, retaining its in-place workflow and node behavior.
Report the failing spec and zero applied count.

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
reset behavior with the consumer's hidden-value preservation. Copying the
workflow costs memory proportional to its size. Successful batches also execute
the producer twice, including private dry-run nonce allocation; no performance
or lifetime-cost improvement is claimed.

## Validation

`test_rebuild_batch_guard.py` covers unsafe success regressions, aliases,
rollback after add_node, CLI file bytes, nested owners, interior target identity,
no-ops, scalar dotted names, growth and shrink/regrowth. Existing batch-order
and DynamicGroup tests retain coverage of ordinary successful work.

## Revisit triggers

Revisit when a coordinated shared-consumer replay contract resolves #1101,
when schema ownership introduces new structural families, or when workflow
copy costs require measured optimization without weakening atomic refusal.
