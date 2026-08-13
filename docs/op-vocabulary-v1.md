# Op vocabulary — v1 (frozen)

Status: **FROZEN**. This document is the normative contract for the structured-edit
op vocabulary in `comfy_cli/workflow_ops.py`: the op kinds, their argument shapes,
their idempotency and conflict rules, and the batch protocol. Downstream repos
(cloud `services/agent`, `harness`, the merge consumer) cite this document **by
commit SHA**, not by branch.

Machine-readable projection: `workflow_ops.FROZEN_OPS`, `workflow_ops.DEFERRED_OPS`,
`workflow_ops.BATCHABLE_OPS`. `tests/comfy_cli/test_op_vocabulary_contract.py`
enforces that this document, those constants, and the `apply_op` / `apply_specs`
dispatch tables agree. A change to any of the three without the others fails CI.

This freeze describes the code on the **unmerged branch
`fix/validate-lowers-ui-to-api`**. Until that branch merges to `master`, a SHA
citation must point at a commit on that branch.

## 1. Frozen op kinds

Six kinds. No other kind is valid in v1: `apply_op` rejects an unknown kind with
`ValueError("unknown op ...")` — it never ignores one.

| Kind | Batchable | Standalone command | Summary |
|------|-----------|--------------------|---------|
| `add_node` | yes | `comfy workflow add-node` | Mint and insert one node |
| `connect` | yes | `comfy workflow connect` | Wire one output slot to one input slot |
| `set_widget` | yes | `comfy workflow set-widget` | Set one widget value by name |
| `delete_node` | yes | `comfy workflow delete` | Remove one node and its incident links |
| `clear` | no | `comfy workflow clear` | Remove every node, link, and group |
| `reset_doc` | no | `comfy workflow reset-doc --confirm` | Reset the whole document to an empty baseline |

Batchable = the kind is accepted by `apply_specs` (the `workflow apply` /
`workflow foreach` batch surface). `clear` and `reset_doc` rewrite the whole
document, so they are standalone-only: a batch containing either is rejected
atomically with its own registered error code —
`workflow_clear_not_batchable` / `workflow_reset_doc_not_batchable` — and a hint
naming the standalone command. Nothing from such a batch is applied.

Every op carries the common envelope stamped by `_new_op`:

```json
{
  "op": "<kind>",
  "op_id": "<uuid4 hex>",
  "actor": "<origin string, section 7>",
  "base_version": 0,
  "stamp": [<base_version>, "<actor>"]
}
```

### 1.1 `add_node`

Spec form (batch input):

```json
{"op": "add_node", "class_type": "KSampler", "at": [x, y], "as": "sampler"}
```

`at` is optional (layout assigns a collision-free position at mint time; the
position freezes into the op). `as` is optional and declares a batch-local alias
(section 5). Minted op fields beyond the envelope: `node_id` (mint_id int),
`class_type`, `pos`, `node` (the complete node object — replay inserts it verbatim).

* Idempotency: re-applying the same `op_id` is a no-op; independently, replaying
  an `add_node` whose `node_id` already exists in the graph is a no-op.
* Conflict: none — ids are minted leaderlessly (section 6), so two concurrent
  `add_node` ops never target the same identity.
* Invalid: an unknown `class_type` is rejected at mint time (`UnknownNodeType`,
  rendered as `node_not_found` with close matches).

### 1.2 `connect`

Spec form:

```json
{"op": "connect", "from": "$up.MODEL", "to": "$sampler.model"}
```

`from`/`to` are `<node>.<slot>` where `<node>` is an int id, a bare alias, or a
`$`-prefixed alias, and `<slot>` is a name or an index. Minted op fields:
`link_id` (mint_id int), `from_node`, `from_slot` (resolved output index),
`to_node`, `to_slot` (resolved input index; `null` for autogrow), `link_type`,
and optionally `grow` (autogrow slot descriptor: `{name, type, widget?, inputcount?}`).

* Idempotency: `op_id` no-op; a link tuple with an already-present `link_id` is
  not appended twice.
* Conflict: a concrete input holds at most one link — a connect to an occupied
  input replaces it and fully retires the prior link (`_remove_link`). Two
  concurrent connects to the same concrete input are an update-vs-update
  conflict on that input (section 3). Autogrow connects are non-clobbering:
  each grows a fresh slot keyed by `grow_id` (the link id), so both survive.
* Invalid: type-mismatched slots are rejected at mint time; a link cannot cross
  a subgraph boundary (rejected with the boundary explanation).

### 1.3 `set_widget`

Spec form:

```json
{"op": "set_widget", "node": "$sampler", "widget": "steps", "value": 30}
```

`node` is an int id, alias, `$alias`, or a subgraph-scoped id (section 6).
Minted op fields: `node_id`, `widget` (name, never index), `value`, `old`; for a
subgraph interior write also `path` (resolved node path, list of strings) and
`inner_widget`; optionally `warnings` (e.g. `normalized_value`).

* Idempotency: `op_id` no-op.
* Conflict: last-writer-wins per `(node, widget)` target (section 3).
* Invalid: an unknown widget name or shape-mismatched value is rejected at mint
  time; at replay an unknown widget name on a live node also rejects
  (`_widget_index` raises), while a missing node is a no-op (delete wins).

### 1.4 `delete_node`

Spec form:

```json
{"op": "delete_node", "node": "$sampler"}
```

Minted op fields: `node_id`, `removed_links` (ids of every link incident to the
node at mint time). Replay removes the node, drops incident links (both the
recorded ones and any link whose endpoint is the node), and scrubs dangling
input/output references.

* Idempotency: `op_id` no-op; replaying a delete of an already-absent node is a
  no-op.
* Conflict: delete wins over concurrent updates (section 3).
* Invalid: deleting a node that does not exist is rejected at mint time
  (`node not found` with the live node inventory).

### 1.5 `clear` — standalone only

Command: `comfy workflow clear <file>`. Minted op fields: `removed_nodes` (ids
present at mint time). Replay empties `nodes`, `links`, and `groups`.
`last_node_id` / `last_link_id` are preserved so ids minted after a clear stay
monotonic — id reuse would let a merge resurrect a deleted node's identity.

* Batchable: **no**. `apply_specs` rejects it with the registered code
  `workflow_clear_not_batchable`; the batch is discarded atomically and the hint
  names the standalone command.
* Idempotency: `op_id` no-op; clearing an empty document changes nothing.

### 1.6 `reset_doc` — standalone only

Command: `comfy workflow reset-doc <file> --confirm`. Implemented by amendment
v1.1 (§10); `DEFERRED_OPS` is now empty. Minted op fields: `removed_nodes` (ids
present at mint time), same as `clear`.

Replaces the entire document with the empty baseline, **including apply
bookkeeping** — unlike `clear`, which preserves the id high-water marks and the
applied-op history. `last_node_id` / `last_link_id` go to 0 (safe: ids come from
`mint_id`, never from the high-water marks — §8.3), `_applied_ops` and
`_widget_stamps` are dropped, and only the document `id` survives. Because it
erases replay history it is a **history barrier**: ops minted against a
pre-reset `base_version` do not replay across it.

* Guard: the CLI surface requires an explicit `--confirm`; without it the
  command fails closed with `workflow_reset_doc_unconfirmed` and writes nothing.
  The check runs before the file is read, so an unconfirmed call cannot fail
  halfway. It is the only edit command with a guard, because it is the only one
  no later op can undo.
* Idempotency: the reset's own `op_id` is written into the freshly-emptied
  `_applied_ops`, so a re-delivered `reset_doc` is a no-op, not a second wipe.
* Batchable: **no**, for the same reason as `clear` — rejected with
  `workflow_reset_doc_not_batchable`.
* Never emitted implicitly: no `--emit-ops` surface and no bulk writer (§8.8)
  mints one. It exists only where a caller asked for it by name.

## 2. Idempotency and identity

* Every op carries `op_id`: uuid4 hex, minted by the **creator, before
  dispatch** (`_new_op`). Receivers never regenerate or rewrite an `op_id`.
* `apply_op` records applied `op_id`s in the document's `_applied_ops` list and
  drops any op whose `op_id` is already there. **Uniqueness scope is
  PER-WORKFLOW**: the same `op_id` can exist in two different workflow documents
  without interaction; within one document each op applies exactly once.
* `_applied_ops` (and `_widget_stamps`) are apply-time bookkeeping, stripped
  before serialization to disk (`strip_internal`).

## 3. Conflict rules

Scalar conflicts resolve by last-writer-wins on the op stamp. The stamp is
`stamp = [base_version, actor]` (stamped by `_new_op`); the exact comparison is
`_stamp_key` in `workflow_ops.py`:

```python
def _stamp_key(op: dict) -> list:
    stamp = op.get("stamp") or [op.get("base_version", 0), op.get("actor", "")]
    return [stamp[0], stamp[1], op["op_id"]]
```

and the gate is `_lww_gate`: a write applies iff `_stamp_key(op) > list(prior)`
for its target. Higher `base_version` wins; ties break by `actor`, then by the
unique `op_id` — so no two distinct ops ever compare equal, the order is total,
and the surviving value is independent of apply order.

| Scenario | Ruling | Where in code |
|----------|--------|---------------|
| update vs update (same widget) | LWW on `stamp` with `op_id` tiebreak; loser dropped | `_lww_gate` / `_stamp_key` |
| update vs delete | **delete wins**: `set_widget` to a deleted node is a no-op; `connect` with either endpoint deleted is a no-op; replay never raises on a since-removed target | `_apply_set_widget` (missing node → return), `_apply_connect` (missing endpoint → return) |
| concurrent moves | no `move` op exists in v1 — positions are decided once at `add_node` mint time and frozen into the op; live position editing is frontend view state, out of scope until the FE stable-ID reconciliation (section 6) | `add_node` / `layout.cascade_pos` |
| edges referencing deleted nodes | the connect no-ops (delete wins); a delete removes incident links and scrubs every dangling input/output reference, so no dangling edge survives either order | `_apply_connect`, `_apply_delete_node` |
| duplicate entity creation | impossible by construction across writers (random 53-bit `mint_id`, no shared counter); a replayed `add_node` whose `node_id` already exists is a no-op; a re-sent op is dropped by `op_id` | `mint_id`, `_apply_add_node` |
| concurrent autogrow connects to one base | both survive: each grows a fresh slot keyed by `grow_id`; their display order is the one sequence decision a leaderless writer cannot make and is surfaced by `detect_conflict` for the merge consumer | `_apply_connect` (grow path), `detect_conflict` |
| invalid / inapplicable ops | explicit per kind — unknown kind: **reject** (`apply_op` raises); malformed op (missing required field): **reject**; well-formed op whose target node is gone: **no-op** (delete wins); `set_widget` naming a widget the live schema does not have: **reject**; `clear`/`reset_doc` inside a batch: **reject** with `workflow_clear_not_batchable` / `unknown op`. Rejection is never silent | `apply_op`, `apply_specs`, `_widget_index` |

## 4. Partial batches: abort-remainder

Ruling: **abort-remainder**. In a batch of `n` ops, if op `k` fails, ops
`k..n` are **not applied**. The ack reports:

```json
{"applied_count": <k-1>, "failed": {"index": <k>, "op": {...}, "code": "<error code>"}}
```

A retried batch converges by the idempotency rule: every op that did apply is
dropped on re-apply by its `op_id`, so retrying the whole batch is exactly-once
per op. The retrier fixes or removes the failing op; it never re-mints `op_id`s
for ops that may already have landed.

The local CLI batch surface (`comfy workflow apply`) is stricter than the
minimum: it discards the **entire** batch on any failure (`applied_count` is
always 0 on failure — nothing is written) and restates the surviving node
inventory in the error. That is a conforming implementation of abort-remainder;
a merge consumer MUST NOT apply any op after the failing index and MUST report
`applied_count` truthfully.

## 5. Aliases

An `add_node` spec may declare a batch-local alias with `"as": "<name>"`. Later
specs in the same batch reference the minted node by alias.

* Both reference forms are valid: bare (`"up.MODEL"`, `"node": "up"`) and
  `$`-prefixed (`"$up.MODEL"`, `"node": "$up"`). Exactly one leading `$` is
  stripped before lookup (`resolve_ref`); the two forms resolve identically.
* **`$`-prefixed is the canonical form** — use it in documentation and
  generated batches. It makes an alias visually distinct from a node id or a
  class name.
* `${` is **rejected**: `${name}` is reserved for recipe parameters
  (`substitute_params` fills them from `--param`; an undeclared `${name}` is a
  `RecipeError`). A `${...}` reaching `resolve_ref` means an unsubstituted
  recipe parameter and fails with a message saying exactly that — it is never
  treated as an alias or a node id.
* A duplicate alias in one batch is rejected (`alias ... is already defined by
  an earlier spec`); an unknown alias falls through to id resolution and fails
  as `node not found` with the live node inventory. Both behaviors predate this
  freeze and are unchanged.
* Aliases are batch-scoped. They do not persist into the document or across
  batches; the ack maps each alias to its minted `node_id`.

## 6. Stamping and IDs

* `op_id`: uuid4 hex, minted by the creator pre-dispatch. Receivers never
  regenerate one (section 2).
* Node and link ids: `mint_id()` — random ints in `[2^40, 2^53)`. Leaderless
  and collision-free without coordination; always inside JS
  `Number.MAX_SAFE_INTEGER`; always larger than small frontend counter ids.
  `last_node_id` / `last_link_id` are advisory high-water marks, never
  allocators.
* Subgraph-scoped ids: an interior node is addressed as `57:3` (the flattened
  form the UI→API lowering mints and `validate` / server errors print) or
  `57/3` (the edit-path form); both resolve to the same interior target. **Ops
  must carry fully-scoped ids** — a bare interior id is meaningless at the top
  level and is rejected, not guessed.
* OPEN: ID representation is to be reconciled with the FE stable-ID workstream
  before this document's v1.1. Until then, the shapes above are the contract.

## 7. Attribution origins

The `actor` field carries the origin of the op. Frozen origin grammar:

| Origin | Format | Example |
|--------|--------|---------|
| agent turn | `agent:<thread>:<turn>` | `agent:th_8f2c:12` |
| human editor | `human:<user>:<tab>` | `human:u_41ab:tab_2` |
| system-minted | `system:mint` | `system:mint` |

The actor participates in LWW tie-breaking (section 3), so origin strings must
be stable within a writer session. The CLI's `--actor` flag carries the origin;
its default `cli` is a legacy value accepted for interactive local use —
merge-consumer traffic uses the structured forms above.

## 8. Replication and replay semantics

Rules a second implementation (JS/TS merge consumer, multi-player) must follow
to converge with the Python applier. Grounded against `workflow_ops.py` and the
V1-007 CRDT replay spike. The unit of replication is the **op**: every replica
applies every op exactly once (any order) through an applier with these
semantics. Exchanging raw document state between concurrently-editing replicas
is not equivalent and does not inherit these guarantees.

### 8.1 Stamp comparison is code-point lexicographic

`_stamp_key` builds `[base_version, actor, op_id]` and relies on Python
sequence comparison: element-wise, first difference decides. The frozen rule,
so any implementation compares identically:

* `base_version`: numeric comparison.
* `actor`, then `op_id`: **Unicode code point order** (Python `str` `<`),
  compared character by character; a strict prefix sorts before its extension.
* No locale, no case folding, no normalization.

`op_id` is lowercase ASCII hex (8.2), so code-point order equals byte order
for it. `actor` strings MUST be ASCII (the section 7 grammar is) — for ASCII,
JS UTF-16 `<` agrees with code-point order; above the Basic Multilingual Plane
it does not, which is why non-ASCII actors are not valid.

### 8.2 `op_id` format is LWW-load-bearing

`_new_op` emits `uuid.uuid4().hex`: exactly **32 lowercase hex characters
`[0-9a-f]`, no dashes**. This is a frozen format, not an implementation
detail: `op_id` is the final LWW tiebreaker (8.1), so its generation and its
lexicographic comparison decide conflict outcomes, not just deduplication. An
implementation that emits a different shape (uppercase, dashed UUID, shorter)
changes who wins ties. Receivers never regenerate or normalize an `op_id`.

### 8.3 `last_node_id` / `last_link_id` are max-registers

Both are advisory high-water marks, never allocators (ids come from
`mint_id`). Register semantics: **max-register** — a write is
`max(current, new)`, and merging two replicas' values is `max(a, b)`; a plain
overwrite is wrong under concurrency.

`_apply_add_node` implements this for nodes:
`workflow["last_node_id"] = max(workflow.get("last_node_id") or 0, op["node_id"])`.
`_apply_connect` does **not** bump `last_link_id` today — no apply path writes
it. The intended, frozen rule is symmetric: connect SHOULD set
`last_link_id = max(last_link_id, link_id)`; the omission is a known gap, and
because the field is advisory, an implementation that already bumps it does
not diverge semantically from one that does not. `clear` preserves both
(section 1.5).

### 8.4 `inputcount`-family autogrow: one op, two registers

A connect whose `grow.inputcount` is set (the kijai `*Multi` family) performs
two writes under one `op_id`:

1. **Structural growth**: a new input slot with a bare `{elem}_N` name
   (`_next_inputcount_name` — never the dotted `base.elemN` autogrow shape),
   keyed by `grow_id = link_id` for idempotent, non-clobbering replay.
2. **A stamped widget write** to the family's count widget
   (`_apply_inputcount_bump`), passing through the same `_lww_gate` /
   `_lww_commit` as an explicit `set_widget`, **stamped with the connect's own
   `op_id` / `stamp` / `base_version`**. It therefore occupies the same LWW
   register (`("widget", node_id, widget)`) as a concurrent explicit
   `set_widget` on that widget, and the winner is decided by 8.1 regardless of
   apply order.

The written value is the mint-time-planned count — a static property of the
op, never re-derived from a post-collision slot number — so both apply orders
carry the same winning value and the graph converges. Known, accepted
limitation: the register is LWW, not a monotonic counter, so a slot that loses
a bare-key naming race can leave the count low until the next write to that
widget. When the applier has no catalog (`graph is None`), the slot still
grows and the count write is skipped.

### 8.5 `op.node` on `add_node` is authoritative

`_apply_add_node` inserts `op["node"]` verbatim (`copy.deepcopy`, one append).
Receivers MUST use the payload as-is and MUST NOT re-mint the node from the
schema catalog at apply time: widget defaults drift between catalog versions,
so a re-derived node diverges from the creator's. No catalog is needed to
apply an `add_node`.

### 8.6 Bootstrap: one common initial snapshot

All replicas of a workflow document MUST fork from one seeded initial
snapshot. Independently re-seeding the same base workflow on two replicas
creates content with distinct internal identities that **duplicates on first
merge** — silently, because each replica looks correct alone (verified in the
spike). Creating a document and seeding its base state is a single-writer
event; replication starts from that snapshot.

### 8.7 Subgraph scope

Current contract, pinned:

* **Only `set_widget` is subgraph-scoped.** Three address forms are accepted
  and normalize to ONE write target: flat promoted (`57.text`, routed through
  the instance's `proxyWidgets`), nested interior (`57/3.steps`), and the
  flattened UI→API alias (`57:3.cfg`). The minted op carries the **resolved**
  `path` (e.g. `["57", "27"]`) plus `inner_widget`, so replay needs no
  proxyWidgets logic, and the LWW target is
  `("widget", ("57", "27"), "text")` — a flat-form and a nested-form
  concurrent write to the same interior widget converge under 8.1.
* **`connect` refuses subgraph scope** with a structural explanation: an
  interior endpoint is rejected with "a link cannot cross the subgraph
  boundary"; a promoted-widget target is rejected with "promoted widget (a
  value), not a link input".
* **`add_node` and `delete_node` cannot address interior nodes** at all; an
  interior id fails as node-not-found against the top-level inventory.
* An interior write to a **shared** definition forks the definition at apply
  time (`engine._isolate_shared_subgraph`): the definition is deep-copied
  under `"sg-" + sha256(def_id + "\x00" + instance_id)[:32]` — deterministic,
  never random — and the instance's `type` is repointed, so two replicas
  replaying the same op produce byte-identical graphs and sibling instances
  are never aliased.
* OPEN: the shared-definition forking semantics above are apply-time behavior
  that rewrites `instance.type` without an explicit op saying so. A full
  specification (fork visibility, interaction with concurrent interior writes
  to sibling instances, definition garbage collection) is owed before this
  document's v1.1, together with the FE stable-ID reconciliation (section 6).

### 8.8 Bulk writers emit ops, they do not re-seed

A **bulk writer** is any command that replaces the working file wholesale rather
than editing it: `comfy templates fetch -o <file>` today, `workflow get -o`
next. Downstream, such a replacement used to become a new document — the
consumer re-minted a snapshot from the new file. §8.6 forbids exactly that for a
replica, and even for the store owner it throws away the attributed history the
op log exists to keep.

`workflow_ops.replace_ops(old, new)` is the alternative, and `templates fetch
--emit-ops` is its first caller. The rules:

* **Shape**: `delete_node` for every node in `old` (in order), then `add_node`
  for every node in `new`, then `connect` for every link. No `set_widget` ops —
  widget values ride inside the `add_node` payload, which §8.5 makes
  authoritative.
* **Identity is re-minted, never inherited.** Template graphs are numbered from
  small frontend counters; replaying those ids into a live document reuses
  identities a concurrent replica may still hold (§1.5's resurrection hazard).
  Every node and link gets a fresh `mint_id` and every interior reference
  (`inputs[].link`, `outputs[].links`, the `links` tuples) is remapped onto it.
* **Dual shape.** Each emitted entry is a fully minted op (envelope + the kind's
  minted fields) AND carries that kind's spec keys (`class_type`/`at`/`as`,
  `from`/`to`, `node`). The same array therefore replays through `apply_op`
  losslessly and is accepted verbatim by `apply_specs`. The two are not
  equivalent: `apply_specs` re-mints each node from the live catalog, so it
  reproduces the structure (classes + wiring) while the op path reproduces the
  graph exactly, widget values included.
* **All or nothing.** A graph the vocabulary cannot express — a subgraph
  definition, a canvas group, a reroute point, a malformed node or link — emits
  **no ops at all** (`NotExpressibleError`, surfaced as `ops_skipped`), never a
  partial batch. A partial batch applies cleanly and leaves a document that is
  not the graph the caller asked for; the consumer is expected to keep its
  whole-document fallback for these cases.
* **`reset_doc` is never part of a bulk batch** (§1.6). Replacing a canvas is
  expressed as deletes + adds, which merge; a history barrier does not.

## 9. Amendments

* Post-freeze changes require a **versioned amendment section** appended to
  this document (`## Amendment v1.x — <date>`), stating what changed and why.
  Silent edits to frozen sections are not valid; the contract tests pin the
  frozen table against the code.
* Downstream repos cite this document by commit SHA and upgrade by moving the
  SHA, never by tracking a branch.
* Adding, removing, or re-scoping an op kind requires updating `FROZEN_OPS` /
  `DEFERRED_OPS` / `BATCHABLE_OPS`, the dispatch tables, and this document in
  one commit — `tests/comfy_cli/test_op_vocabulary_contract.py` fails otherwise.

## 10. Amendment v1.1 — 2026-08-12 (V1-038 / BE-7171)

**`reset_doc` is un-deferred.** `DEFERRED_OPS` is now empty; `apply_op`
dispatches `reset_doc` and `apply_specs` rejects it as standalone-only with its
own registered code. §1.6 is rewritten from "semantics when implemented" to the
implemented contract, and the frozen table's standalone-command cell names
`comfy workflow reset-doc --confirm` instead of "(deferred)". No frozen kind was
added, removed, or re-scoped: `reset_doc` was already in `FROZEN_OPS` and
already `Batchable = no`.

*Why now*: the bulk-writers ticket needed a real, guarded "start this document
over" primitive so that "replace the canvas" and "erase the document" stopped
being the same operation. They are now distinct: §8.8's bulk batch replaces the
canvas with merging deletes+adds, and `reset_doc` is the explicit, confirmed
barrier a caller asks for by name.

**§8.8 is new** and normative for bulk writers (`replace_ops`,
`templates fetch --emit-ops`). It adds no op kind — it constrains how existing
kinds are minted for a whole-file replacement.

**No change to §§2-7, 8.1-8.7.** Stamping, LWW, abort-remainder, aliases and
replication semantics are untouched.
