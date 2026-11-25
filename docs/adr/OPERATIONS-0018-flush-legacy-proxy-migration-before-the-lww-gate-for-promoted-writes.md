# ADR-OPERATIONS-0018: Flush Legacy Proxy Migration Before the LWW Gate for Promoted Writes

Date: 2026-11-25

## Status

Proposed

Relation: Amends [ADR-OPERATIONS-0002](OPERATIONS-0002-version-structured-workflow-operations.md).
No operation kind is added, removed, or re-scoped; `FROZEN_OPS` / `DEFERRED_OPS`
/ `BATCHABLE_OPS` are untouched. This records Amendment v1.15 in
[`op-vocabulary-v1.md`](../op-vocabulary-v1.md), which states the full rule and
the wire-shape detail.

## Context

A promoted `set_widget` host write (`op-vocabulary-v1.md` §14.1) used to run
the legacy `proxyWidgets` forward-migration flush (§14.4) only when the write
target was itself a pending legacy entry. A write to an unrelated, already-
linked declared widget on the same instance skipped the flush even while
other legacy entries were still pending, so the instance's `widgets_values`
stayed laid out in legacy PROXY order. `set_host_value` indexes that array by
DECLARATION-order `value_index` — a different order whenever the two diverge
— so the write landed on a different widget's slot and silently clobbered its
value (comfy-cli#1168, discussion r4670514870).

A second gap in the same area: only the first write on a replica that ever
observed the pending plan carried a `repair` payload. If a later, higher-
stamped write on that replica (whose own op carries no `repair`, because the
migration had already run locally) reached a peer first, that peer's LWW
register was claimed without the flush running there, and the earlier,
repair-carrying op could then lose the LWW gate on arrival and never run it
either — leaving that replica permanently unmigrated (discussion
r4623941925).

Both gaps are about WHEN the deterministic, idempotent repair runs relative
to the LWW-gated value write, not about what op kinds exist or what a
register is keyed on. ADR-OPERATIONS-0002 already treats "coordinated
documentation, implementation, and test updates" for a vocabulary change as
this program's standing cost of extending the frozen contract; this decision
is the smallest one that pays it.

## Decision

`_apply_set_widget` (`workflow_ops.py`) and `_apply_one_slot_impl`
(`cql/engine.py`) now recompute `plan_proxy_migration` from the live document
on every apply of a promoted host write, and run the flush whenever that
plan is non-empty at all — before the §11.1 LWW gate for the value, and
independent of its outcome. Forking a shared definition
(`_isolate_shared_subgraph`) stays gated on `any(e.repairable for e in
plan)`, since the flush only mutates the definition for a repairable entry;
an `alreadyLinked`/`quarantine`-only plan still flushes (consuming the
legacy tuples, realigning `widgets_values`) without forking anything.

The minting side's wire payload keeps the same shape §14.4 already
specifies: `promoted.repair = {ids}` (no `entry`) is attached to an already-
linked write only when the instance's plan contains at least one repairable
entry. A plan with no repairable entry carries no `repair` payload — the
unconditional-on-apply flush above does not depend on the op carrying one.

**Convergence invariant.** The repair is derived solely from `(instance
path, source node, widget)` via SHA-256 (§14.4) — deterministic, never
random — so running it from live state on every apply, independent of this
op's own LWW outcome, produces the same document on every replica regardless
of delivery order. Only the register for the VALUE stays LWW-gated; the
migration itself is not a register.

## Alternatives considered

- **Gate the flush on `op["promoted"]["repair"]` being present**, matching
  the original §14.4 text. Rejected: that is exactly the race in discussion
  r4623941925 — the field is replica-local and op-order-dependent, not a
  property of the document, so two replicas can each skip it.
- **Flush only on a write to the legacy entry itself, and separately fix the
  wrong-slot read.** Rejected: the wrong-slot write (r4670514870) and the
  wrong-slot read it also produces are the same root cause — `widgets_values`
  staying in proxy order while something indexes it by declaration order —
  and only running the flush removes that order mismatch for both.
- **Treat this as a local, non-contract bugfix and update only the PR
  description.** Rejected: §8.7/§14.4 are pinned contract text a remote
  replication consumer relies on verbatim: when the flush runs, and what
  `promoted.repair` can contain, are exactly those sections' subject. A
  consumer still coded against the old wording's "only on the legacy write"
  trigger stays WIRE-compatible (the payload shape is unchanged:
  `promoted.repair = {ids}`, no `entry`), but it is not REPLAY-compatible for
  every delivery order: if that consumer meets the marker-free, higher-
  stamped write first and the earlier, repair-carrying op second, it still
  gates the flush on the arriving op's own `repair` field, so the second op
  loses the LWW gate and the flush never runs there — the exact race
  r4623941925 describes, left open on that consumer for that one delivery
  order. Converging on every delivery order needs the revised trigger
  (recompute `plan_proxy_migration` from live state on every apply), not
  just the unchanged wire shape; that is why the trigger condition itself is
  contract text, not an implementation detail.

## What we reject

- Gating the migration flush on anything carried by the op itself
  (`repair`, or any other replica-local field) instead of recomputing from
  live document state on every apply.
- A flush that forks a shared definition for a plan that would not change
  it (no change to the `any(e.repairable for e in plan)` fork gate from
  Amendment v1.5).
- Raising out of an offline (no-catalog) apply that reaches this flush path.
  `flush_proxy_migration` itself refuses to run without a catalog (correct:
  it cannot resolve anything); the caller skips the flush and applies the
  write as-is when `graph is None`, matching every other offline-degradation
  site in `workflow_ops.py` (`_apply_inputcount_bump`, `_autogrow_group_port`).

## Consequences

### Positive

- A write to any declared widget on an instance with pending legacy entries
  now converges to the same document on every replica, regardless of which
  op a peer happens to receive first.
- The fix is expressible entirely as "run the existing, already-deterministic
  repair earlier and more often" — no new op field is required for the
  `alreadyLinked`/`quarantine`-only case.

### Negative

- An already-linked write that previously skipped the flush (and was
  therefore cheaper when nothing was pending) now always re-evaluates
  `plan_proxy_migration`, even for an instance with no legacy entries at all
  (`plan_proxy_migration` returns `[]` immediately for those — see its own
  early-out — so the added cost is one lookup, not a full flush).
- Offline (no-catalog) replay of a write that reaches this path now silently
  defers the migration instead of running it, which the previous, narrower
  trigger rarely hit. This is an accepted, pre-existing limitation of
  offline replay (§8.4 documents the same tradeoff for the inputcount-bump
  write), not a new one.

### Known limits

- The flush still refuses outright when `graph is None` AND it is actually
  invoked with a non-empty plan and no catalog — callers must keep the
  `graph is not None` guard this decision adds; a future caller of
  `flush_proxy_migration` that omits it reintroduces the offline crash.

## Validation

- `tests/comfy_cli/cql/test_proxy_migration.py`:
  `test_apply_slot_flushes_an_alreadylinked_only_plan_without_forking`,
  `test_apply_slot_does_not_fork_a_shared_definition_for_an_unrepairable_plan`,
  `test_apply_op_replays_a_pending_alreadylinked_write_offline_without_crashing`.
- `tests/comfy_cli/test_op_vocabulary_contract.py` keeps passing unmodified:
  this amendment changes neither the frozen-kinds table nor any dispatch
  surface it checks.

## Revisit triggers

- A merge consumer other than this CLI needs to decide, from the op alone
  (without recomputing `plan_proxy_migration` against its own document
  state), whether a flush happened. That needs a stronger signal than
  `promoted.repair`'s current optional, repairable-only presence.
- The cost of recomputing `plan_proxy_migration` on every promoted write
  becomes measurable on a workflow with many legacy instances; the early-out
  for an instance with no `proxyWidgets` at all would need to move earlier
  in the write path.
