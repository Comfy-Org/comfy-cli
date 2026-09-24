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

Eight kinds as of amendment v1.6 (§15, below) — `set_node_field` is the
eighth and is **PROPOSED**, not yet ratified; see §1.8. `apply_op` rejects an
unknown kind with `ValueError("unknown op ...")` — it never ignores one.

| Kind | Batchable | Standalone command | Summary |
|------|-----------|--------------------|---------|
| `add_node` | yes | `comfy workflow add-node` | Mint and insert one node |
| `connect` | yes | `comfy workflow connect` | Wire one output slot to one input slot |
| `set_widget` | yes | `comfy workflow set-widget` | Set one widget value by name |
| `set_node_field` | yes | `comfy workflow set-node-field` | Set or clear one node's durable field — `title`, `mode`, `flags.collapsed`, `flags.pinned` (§1.8, PROPOSED) |
| `delete_node` | yes | `comfy workflow delete` | Remove one node and its incident links |
| `clear` | no | `comfy workflow clear` | Remove every node, link, and group |
| `reset_doc` | no | `comfy workflow reset-doc --confirm` | Reset the whole document to an empty baseline |
| `insert_workflow` | no | `comfy workflow insert-workflow` | Merge a complete workflow template in one transaction |

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
{"op": "add_node", "class_type": "KSampler", "at": [x, y], "as": "sampler", "mode": 4}
```

`at` is optional (layout assigns a collision-free position at mint time; the
position freezes into the op). A string `"x,y"` or `"[x, y]"` is parsed to the
same two numbers; the minted `pos` is always numeric. `as` is optional and declares a batch-local alias
(section 5). `mode` is optional (amendment v1.4): the litegraph execution mode
the node is minted with — `0` always (default, omitted), `1` on-event, `2` mute,
`3` on-trigger, `4` bypass. Mute/bypass change what executes, so a recipe that
dropped them rebuilt a different API prompt. Minted op fields beyond the
envelope: `node_id` (mint_id int), `class_type`, `pos`, `node` (the complete
node object — replay inserts it verbatim; a nonzero mode is stamped into it and
echoed as an op-level `mode` field). `allow_deprecated` is optional and spec-only
(never minted into the op): a `class_type` the catalog marks `deprecated` is
refused with `node_deprecated` unless it is `true`.

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
* Conflict: a concrete input holds at most one link, so its occupant is a
  **scalar LWW register** on the target `("input", to_node, to_slot)`, resolved
  by `_stamp_key`/`_lww_gate` exactly like a `set_widget` (section 3, and
  amendment v1.2 for the full rule). The winning connect retires the prior
  occupant with `_remove_link`; the losing connect is dropped whole — no link
  tuple, no out-link entry. Autogrow connects are non-clobbering and therefore
  **not** gated: each grows a fresh slot keyed by `grow_id` (the link id), so
  both survive.
* Invalid: type-mismatched slots are rejected at mint time; a link cannot cross
  a subgraph boundary (rejected with the boundary explanation).

### 1.3 `set_widget`

Spec form:

```json
{"op": "set_widget", "node": "$sampler", "widget": "steps", "value": 30}
```

`node` is an int id, alias, `$alias`, or a subgraph-scoped id (section 6).
The doc host's `insert_workflow` remaps template ids to
`insert:<op>:root:node:<id>`. If a bare `<id>` names no node and exactly one
top-level node has that remapped id, the bare id addresses that node, and
the minted `node_id` is the full `insert:` id. If two inserts remapped the
same id, the not-found error lists both.
Minted op fields: `node_id`, `widget` (name, never index), `value`, `old`; for a
subgraph interior write also `path` (resolved node path, list of strings) and
`inner_widget`; optionally `warnings` (e.g. `normalized_value`). A value that
clearly means one option is rewritten at mint time and recorded with a
`normalized_value` warning. Examples: a directory-prefixed model name, or a
JSON boolean written to a combo whose only options are `'true'`/`'false'`
(case-insensitive). The minted `value` is the real option.

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

### 1.7 `insert_workflow` — standalone only

Command: `comfy workflow insert-workflow <file> <template>`, where `<template>`
may be `-` to read the template payload from stdin. The CLI preserves the
template payload verbatim and emits exactly one stamped op:

```json
{
  "op": "insert_workflow",
  "op_id": "<uuid4 hex>",
  "actor": "cli",
  "base_version": 0,
  "stamp": [0, "cli"],
  "workflow": {"nodes": [], "links": [], "groups": [], "definitions": {"subgraphs": []}}
}
```

The `workflow` payload is authoritative and requires a top-level `nodes` array;
`links`, `groups`, and `definitions` are optional. When present, `links` and
`groups` must be arrays and `definitions` must be an object. The CLI checks only
this outer shape. It does not validate graph semantics, remap IDs, apply the op,
or write a mutated workflow document. Per the contract decision recorded in the TDD
(vetoable), cmp owns deterministic ID remapping from the op envelope ID. Only `define_subgraph` and
`insert_workflow` ops may carry definitions; edit ops reject a `definitions`
field as `malformed_op`. The kind is not batchable and a spec batch rejects it as
`workflow_insert_workflow_not_batchable`.

### 1.8 `set_node_field` — PROPOSED (amendment v1.6, not yet ratified)

Command: `comfy workflow set-node-field <file> <node_id> <field> <value>` or
`... --clear`. Spec form (batch input):

```json
{"op": "set_node_field", "node": "$sampler", "field": "title", "value": "My Sampler"}
```

`node` resolves exactly like `set_widget`'s (int id, alias, or `$alias`).
`field` is one of the closed set `WRITABLE_NODE_FIELDS`: `title`, `mode`,
`flags.collapsed`, `flags.pinned`. `value` matches that field's type (`title`
is a string, `mode` an int, the two `flags.*` fields booleans), or is `null`
to clear the field back to absent — so a flag returns to absent the way
workflow JSON round-trips it. Minted op fields beyond the envelope:
`node_id`, `field`, `value`.

* Idempotency: `op_id` no-op, exactly like `set_widget`.
* Conflict: last-writer-wins per `(node_id, field)`, on its own register —
  `("node_field", node_id, field)` — never the `("widget", node_id, widget)`
  one, since none of the four writable fields is a catalogued widget name or
  carries a `widget_order` position. One register per field means a `title`
  write and a `flags.collapsed` write on the same node never contend, and
  neither contends with a `set_widget` write on that node. Two writers of the
  IDENTICAL value to the same `(node, field)` target are not escalated by
  `detect_conflict` — the equal-value carve-out `set_widget` already had is
  extended to `set_node_field`, since both are plain LWW registers. See §3
  and §1.8.1 below.
* Invalid: `field` outside the closed set is rejected at mint time; a
  non-`null` `value` that doesn't match that field's type (`title` a string,
  `mode` a non-boolean int in `{0, 1, 2, 3, 4}`, `flags.collapsed` /
  `flags.pinned` a boolean) is rejected as `malformed_op` at BOTH mint time
  and replay (`_validate_node_field_value`) — a peer-authored or replayed op
  that skipped the CLI's own check gets no less scrutiny than a freshly
  minted one. A missing node is rejected at mint time (mirrors `set_widget`'s
  not-found handling), and at replay a since-deleted target is a no-op
  (delete wins), same as every other write. A `null`-delete against a node
  whose `flags` container is missing or not an object is also a true no-op —
  it never materializes an empty `flags: {}`, which would otherwise make
  replicas that did/didn't see the op diverge.
* No catalog is required to mint or apply this op.
* Why the field list is closed: everything outside it either has an op of
  its own (`widgets_values` is `set_widget`'s; `inputs`/`outputs` belong to
  `connect`) or is node identity (`id`, `type`) that only
  `add_node`/`delete_node` may move. `flags` as a whole is excluded too — a
  whole-object write would reintroduce exactly the clobber this op exists to
  avoid.
* No interior or subgraph-promoted variant exists in this amendment. A
  subgraph-interior node's fields are out of scope; if that need is
  confirmed, it should follow `set_widget`'s interior shape (`path` + a
  field-equivalent of `inner_widget`) rather than growing a distinct
  mechanism.

#### 1.8.1 Relationship to comfy-multi-player's `set_node_field` (#235)

comfy-multi-player PR #235 (merged to that repo's `main`, commit
`28d2e6166867f16bc291f2ae60d7f4a6a0c0c9f7`) landed the generalized
`set_node_field` op in its CRDT applier — superseding that repo's earlier,
title-only `set_title` prototype (PR #232 / ADR-032) — because the
title-stomp bug it originally closed generalized cleanly to the same four
fields once `flags.collapsed`/`flags.pinned` sync came up against the same
whole-node-upsert clobber, and a client-side sync fix should not wait on this
document's own release cycle. This amendment brings comfy-cli's local op
vocabulary into alignment with that merged upstream shape rather than the
withdrawn single-field one.

comfy-multi-player's op additionally carries an optional
`node_incarnation` field — its Y.Doc can, in principle, see a node id reused
after a tombstoned delete, so its LWW register is scoped by `(node_id,
node_incarnation, field)`. **comfy-cli has no equivalent field and does not
need one**: node ids here are minted by `mint_id()` as leaderless random
53-bit integers and are never reused (§6, §1.5's resurrection-hazard note),
so a bare `("node_field", node_id, field)` register is already
collision-free. A future reconciliation pass (once this amendment is
ratified) should confirm comfy-multi-player's applier accepts an op that
omits `node_incarnation` rather than requiring one only because its own
local model can need it.

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

Gated targets: the `set_widget` rows, `set_node_field`'s
`("node_field", node_id, field)` register (§1.8, amendment v1.6), the
connect-embedded `inputcount` bump (8.4), and — since amendment v1.2 — a
concrete `connect`'s `("input", to_node, to_slot)`. **Node ids in a target are
compared as strings** (v1.2): ids are legitimately either JSON type, and
comparing them raw gave `7` and `"7"` two registers for one node.

| Scenario | Ruling | Where in code |
|----------|--------|---------------|
| update vs update (same widget, or same node field) | LWW on `stamp` with `op_id` tiebreak; loser dropped. Two writes of the identical value are not escalated by `detect_conflict` (§1.3, §1.8) | `_lww_gate` / `_stamp_key`, `detect_conflict` |
| **concurrent `connect` to the same concrete input** | LWW on `stamp` with `op_id` tiebreak, target `("input", to_node, to_slot)`; the loser is dropped whole (no link tuple, no out-link entry) and the winner retires the prior occupant. Amendment v1.2 — previously **undefined** and decided by arrival order | `_apply_connect` (concrete branch) / `_lww_gate` |
| update vs delete | **delete wins**: `set_widget` to a deleted node is a no-op; a `connect` whose destination is gone is a no-op; a `connect` whose SOURCE is gone still claims its input register and leaves that input empty (v1.2 — otherwise the incumbent's survival depends on when the delete arrives); replay never raises on a since-removed target | `_apply_set_widget` (missing node → return), `_apply_connect` (missing endpoint → return) |
| concurrent moves | no `move` op exists in v1 — positions are decided once at `add_node` mint time and frozen into the op; live position editing is frontend view state, out of scope until the FE stable-ID reconciliation (section 6) | `add_node` / `layout.cascade_pos` |
| edges referencing deleted nodes | the connect no-ops (delete wins); a delete removes incident links and scrubs every dangling input/output reference, so no dangling edge survives either order | `_apply_connect`, `_apply_delete_node` |
| duplicate entity creation | impossible by construction across writers (random 53-bit `mint_id`, no shared counter); a replayed `add_node` whose `node_id` already exists is a no-op; a re-sent op is dropped by `op_id` | `mint_id`, `_apply_add_node` |
| concurrent autogrow connects to one base | both survive: each grows a fresh slot keyed by `grow_id`; their display order is the one sequence decision a leaderless writer cannot make and is surfaced by `detect_conflict` for the merge consumer | `_apply_connect` (grow path), `detect_conflict` |
| invalid / inapplicable ops | explicit per kind — unknown kind: **reject** (`apply_op` raises); malformed op (missing required field): **reject**; well-formed op whose target node is gone: **no-op** (delete wins); `set_widget` naming a widget the live schema does not have: **reject**; `set_node_field` naming a field outside `WRITABLE_NODE_FIELDS`, or a value of the wrong shape for its field: **reject** with `malformed_op` (§1.8, at both mint and replay); `clear`/`reset_doc` inside a batch: **reject** with `workflow_clear_not_batchable` / `unknown op`. Rejection is never silent | `apply_op`, `apply_specs`, `_widget_index`, `_validate_node_field_value` |

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
* **Promoted widgets are HOST writes** (`cql.promoted`). A
  `set_widget` on a promoted input carries `promoted: {instance_path,
  value_index}` instead of `path`, and its LWW target is the host register
  `("widget", "57", "text")` — the flat form, the nested interior form of the
  widget the input links, and the flattened alias all converge there. A
  legacy `properties.proxyWidgets` entry the definition does not back with a
  linked input is first repaired the way the frontend's forward migration
  repairs it on load (`promoted.flush_proxy_migration`): the op additionally
  carries `promoted.repair = {entry, ids}` where `ids` maps every repairable
  entry of that instance to the subgraph-input uuid and boundary-link ids the
  repair mints, derived with SHA-256 from `(instance path, source node,
  widget)` — deterministic, never random — so replay on any replica produces
  a byte-identical document. The repair mutates the definition, so a shared
  one is forked first exactly like an interior write.
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

## 10. Amendment v1.1 — 2026-08-12 (V1-038)

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

## 11. Amendment v1.2 — 2026-08-12 (concrete-input contention; id-type identity)

Two convergence rules that v1 left undefined, both found by **adversarial
testing against the TypeScript port of this applier** — not by review. The
Python `apply_op` and the port agreed with each other in every case below,
which is what made these contract gaps rather than port bugs.

### 11.1 A concrete input is an LWW register

**The rule.** The occupant of a CONCRETE input slot is a scalar target
`("input", to_node, to_slot)` under exactly the comparison of §3/§8.1 —
`[base_version, actor, op_id]`, numeric then code-point, `op_id` breaking
exact ties. `_apply_connect`'s concrete branch now runs `_lww_gate` /
`_lww_commit` around its write, the same pair `_apply_set_widget` uses.

**The repro** (writer A and writer B, each keeping its own causal order):

```
A: [add_node 400, connect 400 -> 200.positive]
B: [connect 300 -> 200.positive, delete_node 300]
```

Before v1.2, order A-then-B left `200.positive` EMPTY (B's connect displaced
A's link by arrival, then B's delete retired B's link) while order B-then-A
left link 9003 in place. Same op set, two legal interleavings, two different
graphs: one user sees a wired sampler, the other an unwired one.

**The displaced link.** A concrete input holds at most one link, so exactly one
link record survives per register:

* the **winning** connect fully retires the prior occupant (`_remove_link`:
  the link tuple plus the old source's out-link entry). The displaced link is
  deleted, never orphaned and never re-parented to another slot.
* the **losing** connect is dropped WHOLE — no link tuple, no out-link entry,
  no slot write. It still consumes its `op_id` (§2): a dropped write is a
  protocol-level apply, exactly like a losing `set_widget`.

**Composition with delete-wins.** Claiming the register is unconditional once
the gate passes, and it happens BEFORE the source endpoint is resolved:

* destination node gone → the slot does not exist and never will (ids are
  never reused), so there is no register and the op is a plain no-op;
* source node gone → the winning connect still claims the register and clears
  the input. Deferring the retirement until the link is known to be
  installable would reintroduce order dependence: whether the incumbent
  survives would depend on whether the concurrent delete had arrived yet.
  "Delete wins" therefore means *the new link does not appear*, not *the
  previous link is preserved*.
* a stamp outlives the node it names, which is what makes the composed case
  converge: a later, lower-stamped connect onto that input is still dropped.

**Autogrow is explicitly NOT gated.** An autogrow connect grows a fresh slot
keyed by `grow_id`, so two concurrent autogrows onto one base never contend:
both survive (§1.2). Gating `("input", to_node, "grow", base)` would silently
discard one writer's connection. That target keeps its §3 role as conflict
*identity* for `detect_conflict`, not as a gate.

**Alternatives considered and rejected.**

1. *Allow multiple links on one concrete input and let the projection pick.*
   Rejected: it breaks the graph invariant that a concrete input has at most
   one link, and every downstream consumer (`convert_ui_to_api`, the executor,
   the frontend) assumes it. It also just relocates the decision into a
   projection rule that would itself have to be stamp-ordered.
2. *Reject the later connect (return an error to the second writer).*
   Rejected: it breaks "nobody's work is rejected" — a merge consumer replays
   ops that were already accepted from the writer's point of view, and §4's
   abort-remainder would then discard the remainder of an innocent batch. LWW
   drops a write silently and locally; rejection propagates.
3. *Order by receipt (status quo).* Rejected: that is the finding.

**What v1.2 does NOT close** (filed, tested, unchanged):

* `outputs[].links` is appended in arrival order, so two connects out of ONE
  source into two DIFFERENT inputs record the same set in two different
  sequences. No link is lost or invented; closing it means canonicalizing a
  set-valued field in both implementations' projections.
* An autogrow connect racing a delete of its source leaves the grown slot
  present in one order and absent in the other — the structural sibling of the
  gap above, on a target that is deliberately not a register.
* Two `add_node` ops with the SAME `node_id` and different payloads resolve
  first-writer-wins by arrival (`("node", node_id)` is reserved but ungated).
  §1.1 rules this out by construction — `mint_id` draws 53-bit random ids — so
  it is a property of hand-authored or replayed streams, not of minted ones.

**Batch caveat, now stated.** `apply_specs` stamps every op in one batch with
the same `base_version`, so two writes to the SAME target inside one batch are
decided by the `op_id` tiebreak, not by spec order — "last spec wins" does not
hold. This has been true of `set_widget` since the freeze; v1.2 extends the
same property to `connect` and names it rather than leaving it implicit.

### 11.2 Write targets compare node ids as strings

`_write_target` built its key from the raw `node_id` / `to_node` while every
apply-path lookup resolves ids as strings. An op carrying `7` and one carrying
`"7"` therefore addressed the same node through two different registers: the
gate never compared them and the pair converged by arrival order. Node ids are
legitimately either JSON type — historical workflows carry string ids, and
subgraph-scoped addresses are strings like `"57:3"` (§6) — so this is legal
traffic, not malformed input. Interior `set_widget` targets already normalized
their path (`tuple(str(s) for s in path)`) and were unaffected.

**The rule:** every node id in a write target is normalized with `str()`.
Equivalently: **node identity is compared as a string throughout the apply
path.** `_apply_add_node`, `_apply_set_widget`, `_apply_connect` and
`_apply_delete_node` now resolve nodes by string id (`_find_by_str`) so the
register key and the node it names can never disagree. `last_node_id` stays a
max-register over INT ids only — a string id is not comparable and never bumps
it (§8.3).

This changes the BYTES of a stamp key (`["widget", 7, "steps"]` becomes
`["widget", "7", "steps"]`). Stamp maps are apply-time bookkeeping stripped
before serialization (`strip_internal`, §2), and the doc-side `__stamps` map
lives only inside a live document, so the change is not a data migration; a
document mid-flight across the upgrade loses prior stamp claims for
numerically-keyed targets and falls back to first-writer-wins for those
targets until the next write. Downstream repos that pin this document by SHA
must move the SHA and their applier pin together.

**No change to §§2, 4-7, 8.1-8.8** beyond the §3 table row and the §1.2
conflict bullet cited above. No op kind was added, removed, or re-scoped;
`FROZEN_OPS` / `DEFERRED_OPS` / `BATCHABLE_OPS` are untouched.

## 12. Amendment v1.3 — 2026-08-17 (slot-drift totality; oracle id normalization; dict widgets; implicit seed markers)

Adversarial review of the apply path against shape-drifted documents (a merge
consumer replaying ops minted from a different catalog generation) found four
guard gaps, each a sibling of a rule that already existed elsewhere. All are
apply/replay semantics the conformance consumers must mirror.

### 12.1 Concrete `connect`: slot drift is as total as node drift

Totality (§1.2) covered a vanished *node*; it did not cover a vanished *slot*.
A concrete `connect` whose `to_slot` does not exist on the replayed document
(out of range, or a malformed entry) raised `IndexError` — **after** claiming
the LWW register and **before** recording the `op_id`. That pairing is a
poison state: a retry of the identical op loses to the failed attempt's own
stamp and the connect is silently dropped forever, even after the document is
repaired.

**The rule:** a concrete `connect` whose destination slot is absent or
malformed is a total no-op that claims **no** register — there is no slot, so
there is nothing to occupy. A SOURCE slot that is absent or malformed gets the
deleted-source treatment (§1.2): the register claim stands, the input stays
empty, no link is recorded. Additionally, the apply dispatcher now guarantees
that an exception escaping any handler restores the stamp map to its
pre-dispatch state — no code path may leave a stamp committed without its
`op_id` recorded.

### 12.2 `canonical()` accepts the id mix v1.2 declares legal

v1.2 normalized *write targets* with `str()` but left the convergence oracle
sorting nodes and links by raw id — `canonical()` raised `TypeError` on a
document holding both int and string node ids, i.e. it could not compare the
exact traffic v1.2 legitimized. Every key and sort key in `canonical()` now
normalizes with `str()` (nodes, links, `grow_id`, and the link→slot-identity
lookup, which previously missed when the link stored `"7"` and the node `7`).
Ordering inside `canonical()` output changes for pure-int documents
(lexicographic, not numeric) — immaterial, since the oracle is an equality
check both replicas compute with the same function.

### 12.3 Dict-shaped `widgets_values` is projected, never destroyed

The VHS_* family serializes `widgets_values` as a named dict. Write paths that
read it through the list-only view treated it as "no values": one `set_widget`
replaced the whole dict with a sparse list (siblings destroyed, silently), the
`inputcount` bump wrote an integer key INTO the dict (leaving the count stale
next to a garbage key), and `capture` crashed. Wherever a catalog is in scope,
the dict form is now **projected onto the class's default widget order** —
named values land at their schema positions and survive the write; without a
catalog the old "values unknown" degradation stands. `comfy workflow slots`
now reports the real values for such nodes.

### 12.4 The implicit seed companion reaches every order surface

The frontend companions a seed-like INT with `control_after_generate`
regardless of the schema flag, and partner nodes ship such inputs unflagged
under many names (`image_seed`, `model_seed`, `Seed`, `rand_seed`,
`variation_seed`). The engine's order surfaces disagreed about this:
`widget_order_for_node` applied an exact-name implicit rule while
`widget_order`, `widget_order_default` and `widget_defaults` honored only the
explicit flag — so the exported **widget catalog** (the name↔index contract
the doc host consumes, pinned by `catalog_version`) was off by one for every
implicitly-companioned node. All four surfaces now share one predicate whose
implicit rule matches the converter's: an INT whose leaf name contains
``seed`` (case-insensitive). The UI→API converter's companion guard also now
peeks at the next *widget-owning* input rather than the next declared input,
so a connection-only input between a seed and a COMBO no longer defeats the
guard and eats the combo's real value. **`catalog_version` hashes change** for
affected classes; consumers pinning the catalog re-pin with this SHA.

### 12.5 `applied_count` compliance

§4 has always said "``applied_count`` is always 0 on failure — nothing is
written." The implementation reported the number of specs applied before the
abort (all discarded). The code now complies with the doc; no contract change.

**No change to §§2, 4-7** beyond the compliance fix above. No op kind was
added, removed, or re-scoped; `FROZEN_OPS` / `DEFERRED_OPS` / `BATCHABLE_OPS`
are untouched. Downstream repos pinning this document by SHA move the SHA and
their applier/catalog pins together.

## 13. Amendment v1.4 — 2026-08-19 (node mode; capture/apply agreement on UI-only nodes)

### 13.1 `add_node` carries an optional `mode`

A spec (and the minted op) may set `mode` to a litegraph execution mode
(`0` always — the default, omitted; `1` on-event; `2` mute; `3` on-trigger;
`4` bypass). Mute and bypass are graph-semantic — a bypassed node passes its
input through instead of executing — so capture→apply previously revived
bypassed nodes and produced a *different API prompt* from the source workflow.
The mode is stamped into `op.node` (which stays authoritative for replay, §8.5)
and echoed as an op-level field when nonzero. An op without `mode` is exactly
the pre-amendment shape, so existing ops replay unchanged.

### 13.2 `capture` no longer emits ops `apply` refuses

`add_node` has always rejected UI-only node types (`Note`, `MarkdownNote`,
`PrimitiveNode`, `GetNode`, `SetNode`, `Reroute`) — they exist only in the
editor graph and never reach the API. `capture` nevertheless emitted `add_node`
specs for them, so any workflow containing so much as a documentation note
captured into a recipe the (correctly atomic) `apply` discarded whole. capture
now skips UI-only nodes and preserves the data flow they carried: links through
`Reroute` chains and `GetNode`→`SetNode` pairs are spliced to the real upstream
source (the same resolution the UI→API converter applies), and a
`PrimitiveNode`'s value is captured as the fed widget's value. Skipped nodes
are reported as structured warnings on the capture envelope. The recipe
rebuilds the executable graph — the API prompt — not the canvas decoration.

**No change to §§2-8.** No op kind was added, removed, or re-scoped;
`FROZEN_OPS` / `DEFERRED_OPS` / `BATCHABLE_OPS` are untouched.

## 14. Amendment v1.5 — 2026-08-28 (promoted subgraph inputs: host writes, one register per declared input, opaque positional writes)

The frontend (ComfyUI_frontend ADR 0009) keeps a promoted subgraph widget's
value on the HOST instance — `widgets_values[i]` on the instance, positional
over the definition's inputs that resolve to an interior widget — and runs
that value over the interior default. Interior `path` writes (§8.7) therefore
never reached the canvas for a promoted widget. Both sides (this repo,
comfy-multi-player Amendment A15) now speak the shapes below; a subgraph
instance's `type` is a definition UUID with no catalog entry, so every replica
stores its widgets opaquely (positional) and these ops carry what an opaque
store needs.

### 14.1 `set_widget` host write: the `promoted` payload

A write to a promoted input is a top-level `set_widget` on the INSTANCE
(`node_id`, `widget` = the declared input name; no `path`/`inner_widget`)
carrying

```json
"promoted": {"value_index": <int>, "instance_path": [<id>, …],
             "host_widgets_values": [<full materialized array>]}
```

`value_index` is the input's position among the definition's widget-backed
inputs (socket-only inputs own no slot); `instance_path` is one segment for a
top-level instance and the interior node path for a nested host, resolved and
forked exactly like an interior `path`; `host_widgets_values` is the array
after the write, seeded from each input's current effective value so the
positional array stays aligned with the definition. Apply writes ONE slot;
a shorter stored array is extended from the payload as **best-effort
repair** — only the written index is under the register, so two concurrent
writes carrying different tails settle the tail by apply order (an accepted
limitation: a truncated replica is no longer left truncated). The register is
the ordinary `("widget", node_id, widget)` (§11.2 string identity), so the flat
address and the interior address that backs it (`57/13.width`, which the
minting side redirects to the host — `redirected_from` records the given
address, informational) share one register. The host-write register and an
interior `path` register for the interior widget behind the same promotion
are deliberately NOT unified (that would need the promotion table at apply
time); a merge consumer treats them as distinct targets.

### 14.2 `connect` onto a promoted input: one register per declared name

A `connect` whose target is a declared subgraph input the instance does not
yet carry an `inputs[]` entry for carries

```json
"grow": {"name": <declared input name>, "type": <declared type>,
         "promoted": true, "widget": <name, only when the input backs a widget>}
```

Unlike autogrow (§1.2 carve-out), a promoted input is ONE register named by the
definition — `("input", to_node, "grow", <full name>)`, never split on a dot
(declared names such as `images.image0` contain one) — and is gated by
`_lww_gate`/`_lww_commit` exactly like a concrete input (§11.1): the higher
stamp owns the entry in either apply order, `grow_id` follows the winner, the
loser's link is retired, and the claim is unconditional once the gate passes.
Apply reuses an existing entry by name (a concurrent materialization shares
it) and otherwise appends `{name, type, link, grow_id[, widget:{name}]}`
verbatim — no numbering. Every connect to a declared promoted input is this
promoted grow, even once the entry exists and even when addressed by INDEX
(`57.1` landing on the materialized `width` entry maps onto the name at mint
time): the register is the declared name for the life of the input, so a
replica that receives a later connect before the materializing one still
lands it (a concrete `to_slot` op would find no slot, be dropped with its
`op_id` consumed, and never replay). The claim is unconditional once the
gate passes: a promoted grow whose source was concurrently deleted still
claims the register and leaves the entry empty (delete wins over the link,
not over the claim — §11.1), so every interleaving converges. Scope: an op
minted by a pre-v1.5 replica as a concrete `to_slot` onto such an input does
not gate against the promoted register; both replicas must run v1.5 for the
register to be shared.

### 14.3 Opaque positional writes: frontend-only `PrimitiveNode`

A write that resolves to a frontend-only `PrimitiveNode` (no catalog entry;
`widgets_values[0]` holds the value) carries `legacy_primitive: true` AND the
§14.1 `promoted` payload with `value_index: 0` and `instance_path` = the node
id, so an opaque store applies it as a plain positional write. Apply treats a
`promoted` payload on a non-instance node as that positional write.

### 14.4 Legacy `proxyWidgets` entries are repaired before the host write

A `properties.proxyWidgets` entry the definition does not back with a linked
input is repaired first — the frontend's own forward migration
(`proxyWidgetMigration.ts`), ported as `promoted.flush_proxy_migration` — and
the op additionally carries `promoted.repair = {entry, ids}` with the
subgraph-input and boundary-link ids the repair mints, derived by SHA-256 from
`(instance path, source node, widget)` so replay anywhere is byte-identical.
The pinned contract text in §8.7 states the full rule.

## 15. Amendment v1.6 — 2026-09-22 (`set_node_field`: a generalized node-field op — PROPOSED, pending ratification)

> **Status of this amendment: PROPOSED.** Every prior amendment in this
> document (§§10–14) recorded a change already decided and shipped. This one
> is different in kind: it adds a new frozen op kind, which §9 says requires
> updating `FROZEN_OPS`/`BATCHABLE_OPS`, the dispatch tables, and this
> document together — and this commit does exactly that so the contract test
> (`tests/comfy_cli/test_op_vocabulary_contract.py`) stays green — but adding
> a kind to a document three other repos pin BY SHA (see the top of this
> document) is a decision for
> this repo's maintainers, not something a single PR can ratify by merging.
> Treat `set_node_field` as implemented-and-tested-but-not-yet-blessed until a
> maintainer says otherwise; a downstream repo should not move its pin to
> this SHA on the strength of this amendment alone.
>
> **This amendment supersedes the earlier, withdrawn `set_title` proposal**
> that briefly occupied this same §1.8/§15 slot. That proposal covered only
> `title`; it is replaced outright, not layered on top of, by the
> generalized op below — see "Supersedes `set_title`" below for why.

### What changed and why

A node's `title`, `mode` and `flags` enter a workflow document only as
passthrough fields on `add_node`'s initial `node` snapshot (§1.1). No op
wrote any of them afterward: a rename, a mute/bypass, a collapse or a pin
made after the node already existed — by hand, by a script, or by the in-app
agent — produced no replayable op, so a merge consumer had nothing to
receive and a concurrent edit from two sources resolved by accident (arrival
order in whatever process last touched the file) rather than by any of this
document's convergence guarantees. §1.8 adds `set_node_field` to close that
gap for all four fields at once, following `set_widget`'s existing top-level
shape (§8.1's stamp comparison, §2's idempotency, the delete-wins rule in
§3) rather than introducing new machinery — one LWW register per
`(node_id, field)` so none of the four, nor a `set_widget` write on the same
node, ever contend with each other.

### Supersedes `set_title`

An earlier revision of this amendment proposed a title-only `set_title` op.
It is withdrawn in favor of `set_node_field` because the identical
whole-node-upsert clobber that motivated a title op turned out to apply
equally to `mode` (mute/bypass) and the two `flags.*` fields once
ComfyUI_frontend needed to sync `flags.collapsed`/`flags.pinned` out of the
canvas — the same gap, one field wider each time. Rather than land three more
single-field ops in sequence, this revision generalizes to the closed field
list up front, matching the shape comfy-multi-player's own upstream PR
converged on for the same reason (see below). No comfy-cli release ever
shipped `set_title`; this is a same-PR replacement, not a deprecation.

### Prior art: comfy-multi-player PR #235 (merged)

comfy-multi-player PR #235 (merged to that repo's `main`, commit
`28d2e6166867f16bc291f2ae60d7f4a6a0c0c9f7`) generalized its own earlier,
title-only prototype (PR #232 / ADR-032) into the same `set_node_field` shape
this amendment defines, ahead of this document, because the fix was scoped
and needed there and a client-side sync fix should not wait on this
document's own release cycle. This document's §1.8 defines the shape
comfy-cli now implements; §1.8.1 records the one payload difference
(`node_incarnation`) and why comfy-cli's model does not need it. Ratifying
this amendment is what lets comfy-multi-player cite a real upstream
definition for its already-merged op instead of the two living in permanent
disagreement.

### Scope of this change

* `FROZEN_OPS` and `BATCHABLE_OPS` gain `set_node_field`
  (`comfy_cli/workflow_ops.py`); `WRITABLE_NODE_FIELDS` is the closed field
  list (`title`, `mode`, `flags.collapsed`, `flags.pinned`).
* `apply_op` dispatches it via `_apply_set_node_field`; `apply_specs`
  dispatches the `{"op": "set_node_field", "node": ..., "field": ...,
  "value": ...}` spec form via the new `set_node_field()` primitive.
* `_write_target` gains the `("node_field", node_id, field)` register
  (§1.8) — one register per field, not per node.
* `comfy workflow set-node-field <file> <node_id> <field> <value>` (or
  `--clear`) is the new standalone/batch-authoring command, mirroring
  `set-widget`'s CLI shape.
* No change to any existing kind's semantics, to `DEFERRED_OPS`, or to
  §§2–8.8 beyond the new §1.8 addition and this section.

### Hardening — 2026-09-23, adversarial review

Independent adversarial review (rather than testing against a live consumer)
found the kind landed without three things a real freeze needs, plus two
apply-time robustness bugs, both since fixed and folded into §1.8 above:

* **Value validation was missing entirely.** The op validated the field NAME
  against `WRITABLE_NODE_FIELDS` but never the VALUE, at mint time or replay —
  unlike `add_node`, which has always enforced `_VALID_NODE_MODES` for `mode`.
  A bogus `mode` (a string, `99`, a bool, a container) could reach the document
  and break `workflow_to_api`'s exact `mode in (_MODE_MUTED, _MODE_BYPASS)`
  check, or make `ls-nodes`/`print` raise `TypeError: unhashable type`.
  `_validate_node_field_value` now runs at both mint (`set_node_field`) and
  replay (`_apply_set_node_field`).
* **The equal-value carve-out excluded it.** `detect_conflict` skips
  escalating two independent writes of the IDENTICAL value for `set_widget`
  (so two actors setting the same value don't get an unnecessary ask-to-merge)
  but gated that carve-out on both ops being `set_widget`. `set_node_field` is
  the same kind of plain LWW register — the carve-out now covers both.
* **§3's gated-target list didn't mention it.** `_write_target` already
  returned `("node_field", node_id, field)` for the kind, but §3's prose only
  named the `set_widget` rows and the connect-embedded registers; it now lists
  `set_node_field` explicitly.
* `node.setdefault(head, {})` in `_apply_set_node_field` assumed an existing
  `flags` was dict-shaped, but the schema does not forbid `flags: null` or any
  other shape — that node raised a raw `TypeError`/`AttributeError` that
  escaped the handler's `ValueError`/`KeyError` envelope wrapping and corrupted
  replay on any replica holding such a document. It now coerces a non-dict
  container to `{}` only when there is a real (non-null) write to make.
* The container was created, and hence the node mutated, BEFORE `op["value"]`
  was read — a missing `value` raised `KeyError` after the mutation had
  already happened, and rollback only undoes LWW stamps, not that partial
  mutation, leaving a stray empty `flags: {}` behind. `_apply_set_node_field`
  now reads and validates every required op field before any mutation, and a
  null-delete against an absent or malformed container is a true no-op: it
  never creates `flags` at all.
* `set-node-field` is now registered in `discovery.COMMAND_SCHEMAS` like every
  sibling structured-edit command, so `comfy discover` advertises its
  `output_schema`.
* The batch-spec dispatch now reads `spec["node"]`, not
  `spec.get("node", spec.get("node_id"))`: `node` is the documented, canonical
  key (matching `set_widget`'s `node`); `node_id` was never documented and is
  no longer accepted. A missing `node` now raises the same `spec #i
  (set_node_field) is missing required field 'node'` every sibling branch
  raises, instead of silently resolving to `None` — which, since node matching
  is by `str()`, could spuriously match a node whose id happens to be the
  string `"None"`.

### What is explicitly NOT done here

* **Not ratified.** See the status callout above.
* **No interior/subgraph-promoted `set_node_field` variant** — §1.8 states
  this is out of scope, mirroring comfy-multi-player#235's own scope.
* **No reconciliation commit against comfy-multi-player** — that is
  follow-up work for once (and if) this amendment is accepted.
