# ADR-IDENTIFIER: Title

Date: YYYY-MM-DD

Relation: <Builds on | Refines | Amends | Supersedes> ADR-<DOMAIN>-<NNNN>. State
what remains true in that ADR. Omit this line when the ADR stands alone.

<!--
Context, Decision, Alternatives considered, and Consequences are recommended
review guidance. The validator intentionally enforces identity and lifecycle
metadata, not these section names or their prose.
-->

## Status

Proposed

## Context

Describe the problem, the forces that constrain the decision, and why the
decision needs to be made now.

## Decision

Describe the chosen direction and the architectural invariants it establishes.
Name how implementation will prove those invariants. For production behavior,
include health evidence and rollback conditions.

## Alternatives considered

Describe the viable alternatives and why they were not chosen.

## What we reject

List patterns later work must not introduce and why. These are standing rules for
compliance review, distinct from alternatives considered at decision time.

## Consequences

### Positive

- What becomes easier, safer, or more consistent?

### Negative

- What cost, risk, or constraint does the decision introduce?

### Known limits

- What gap remains open, and why is it outside this decision?

## Validation

Describe how implementation and future changes can prove compliance.

## Revisit triggers

State measurable conditions that reopen this decision after acceptance, including
the value and direction that matter.

## Notes

Optional references or follow-up context.
