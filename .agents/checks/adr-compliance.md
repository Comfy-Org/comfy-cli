---
name: adr-compliance
description: Checks changes against the repository's architecture decision records
severity-default: medium
tools: [Read, Grep, glob]
---

Check changed code against the Architecture Decision Records in `docs/adr/`.

## Review method

1. List ADR files, excluding `TEMPLATE.md`.
2. Read each ADR's Status, Decision, and Consequences sections.
3. Compare only the changed behavior and boundaries with those decisions.
4. Flag an actual contradiction with an Accepted ADR as medium or high,
   depending on blast radius.
5. Flag a contradiction with a Proposed ADR as low-severity directional
   guidance.
6. If a change establishes a durable boundary, invariant, data-ownership rule,
   compatibility contract, or dependency choice without an ADR, ask the author
   to record the decision or explain why it is local implementation detail.

Reference the exact ADR identifier and file in every finding. Do not flag
pre-existing behavior or demand an ADR for routine refactors and reversible
implementation choices. A PR description may acknowledge and motivate an ADR
amendment; a source-code comment does not amend an ADR.
