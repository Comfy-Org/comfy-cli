# ADR-OPERATIONS-0002: Version Structured Workflow Operations as a Frozen Contract

Date: 2026-08-20

## Status

Accepted

## Context

Workflow edits are produced and replayed by multiple processes, including the
CLI, agents, test harnesses, and merge consumers. Convergence depends on shared
operation shapes, identity, idempotency, conflict rules, and batch semantics.
An informal Python API is not a sufficient compatibility boundary.

The established contract was introduced in commit `3ff9f55a` through PR #511
and is maintained in [`op-vocabulary-v1.md`](../op-vocabulary-v1.md).

## Decision

Structured workflow edits use an explicit, versioned operation vocabulary.
The detailed vocabulary document is normative for operation fields and replay
semantics and is cited by immutable commit SHA when consumers coordinate a
specific version.

The implementation publishes machine-readable operation sets, and contract
tests keep the document, those sets, and dispatch tables aligned. Unsupported
operation kinds are rejected rather than silently ignored. Proposed additions
remain clearly marked until ratified into the frozen vocabulary.

## Alternatives considered

- Treat Python functions as the contract. Rejected because remote producers
  and consumers cannot depend on one process's call graph.
- Accept unknown operations as no-ops. Rejected because silent acceptance
  makes divergent graphs appear successful.
- Document operation behavior without executable checks. Rejected because the
  implementation and prose had no mechanical drift signal.

## Consequences

### Positive

- Producers and replay consumers share explicit convergence semantics.
- Contract drift fails tests instead of surfacing only in distributed use.
- Commit-addressed versions let downstream consumers coordinate upgrades.

### Negative

- Vocabulary changes require coordinated documentation, implementation, and
  test updates.
- Proposed operations need an explicit ratification step before becoming part
  of the frozen contract.

## Validation

`tests/comfy_cli/test_op_vocabulary_contract.py` compares the documented table
with `FROZEN_OPS`, `DEFERRED_OPS`, `BATCHABLE_OPS`, and the operation dispatch
surfaces in `comfy_cli/workflow_ops.py`.
