# Op vocabulary — v1 (frozen)

The accepted architectural role of this frozen, versioned contract is recorded
in [ADR-OPERATIONS-0002](adr/OPERATIONS-0002-version-structured-workflow-operations.md).

Status: **FROZEN**. This document is the normative contract for the structured-edit
op vocabulary in `comfy_cli/workflow_ops.py`: the op kinds, their argument shapes,
their idempotency and conflict rules, and the batch protocol. Downstream
consumers (the cloud agent, its test harness, the merge consumer) cite this
document **by commit SHA**, not by branch.

Machine-readable projection: `workflow_ops.FROZEN_OPS`, `workflow_ops.DEFERRED_OPS`,
`workflow_ops.BATCHABLE_OPS`. `tests/comfy_cli/test_op_vocabulary_contract.py`
enforces that this document, those constants, and the `apply_op` / `apply_specs`
dispatch tables agree. A change to any of the three without the others fails CI.

This freeze describes code on `main`. A SHA citation points at the commit that
contains the contract version a consumer implements.

## 1. Frozen op kinds

Nine kinds as of amendment v1.8 — `disconnect` is the ninth and arrives with
that amendment; see §1.2.1. `set_node_field` is the eighth and is **PROPOSED**,
not yet ratified; see §1.8 and §15. No other kind is valid in v1: `apply_op`
rejects an unknown kind with `ValueError("unknown op ...")` — it never ignores
one.

| Kind | Batchable | Standalone command | Summary |
|------|-----------|--------------------|---------|
| `add_node` | yes | `comfy workflow add-node` | Mint and insert one node |
| `connect` | yes | `comfy workflow connect` | Wire one output slot to one input slot |
| `disconnect` | yes | `comfy workflow disconnect` | Remove the link from one input slot |
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

`set_dynamic_group_rows` is a batch spec, not an op kind. It replaces a
DynamicGroup's rows (`{"op": "set_dynamic_group_rows", "node": ..., "group":
"loras", "rows": [{...}]}`) and emits only `set_widget` ops: the row count, then
every widget of each written row, including a field's `<field>.0` control mode.
Rows with a connected or promoted field keep their index, are not rewritten, and
are left out of `rows`. comfy-cli's `apply_op` replays these in order because its
workflow model materializes rows from the count; the shared document accepts them
in any order. An earlier write to the group in the same batch is refused.

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
dropped them rebuilt a different API prompt. `title` is optional (amendment
v1.7, §16): a custom display title for the node, in place of the class's
default title. Minted op fields beyond the envelope: `node_id` (mint_id int),
`class_type`, `pos`, `node` (the complete node object — replay inserts it
verbatim; a nonzero mode or a given title is stamped into it and echoed as an
op-level `mode`/`title` field). `allow_deprecated` is optional and spec-only
(never minted into the op): a `class_type` the catalog marks `deprecated` is
refused with `node_deprecated` unless it is `true`.

* Idempotency: re-applying the same `op_id` is a no-op; independently, replaying
  an `add_node` whose `node_id` already exists in the graph is a no-op.
* Conflict: ids are minted leaderlessly (section 6), so two concurrent
  `add_node` ops never target the same identity. Amendment v1.10 adds one
  explicit ordering conflict between a reused-subgraph add and a concurrent
  mutation inside the definition it turns from single-instance to shared.
* Invalid: an unknown `class_type` is rejected at mint time (`UnknownNodeType`,
  rendered as `node_not_found` with close matches), except for the exact id of
  a definition already present with a reusable top-level host instance as
  specified by amendment v1.10 (§19).

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

### 1.2.1 `disconnect` (amendment v1.8)

Spec form: `{"op": "disconnect", "to": "$sampler.model"}`. The standalone
command accepts the same `<node>.<input>` target. Minted fields are `link_id`
(the occupant observed at mint time), `to_node`, and the resolved numeric
`to_slot`. A dynamic destination additionally carries exactly one identity
field:

* `grow` is present for a promoted subgraph input and names the declaration;
  it claims `("input", to_node, "grow", grow.name)`.
* `grow_id` is present for an ordinary autogrow or widget-converted input and
  preserves the creating connect's collision-free slot identity; it claims
  `("input", to_node, "grow_id", grow_id)`. A reconnect to that existing input
  carries the same `grow_id` and therefore shares the register. For an interior
  input, the target is path-scoped as
  `("input", tuple(path), to_node, "grow_id", grow_id)`.

For either dynamic form, replay uses the identity field above rather than
treating the observed numeric `to_slot` as a concrete-slot register.

`disconnect` and concrete `connect` claim the same scalar LWW register,
`("input", to_node, to_slot)`. A winning disconnect removes whichever link
occupies the slot when it is replayed, including both endpoint references; a
losing disconnect changes nothing. A missing node or drifted slot is a replay
no-op. Minting against an existing empty input is rejected so callers cannot
mistake an already-disconnected or incorrectly addressed slot for a mutation.

When the destination is an interior node, the op also carries `path` (the
subgraph instance path) and scopes the same numeric-slot register to that path.
It removes only a link whose source and destination are both inside the same
single-instance definition. A boundary-crossing link and a definition shared
by multiple live instances are rejected at mint time.

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
node at mint time). An interior address such as `70/22` additionally carries
the owner-instance `path` (`["70"]`); it is refused when the definition is
shared by multiple live instances, or when the node is the source of a
promoted widget (its host value would shift onto the wrong input). Replay
removes the node, drops incident links (both the recorded ones and any link
whose endpoint is the node), and scrubs dangling input/output references in
that same graph scope — for an interior delete, including the definition's
boundary `inputs[].linkIds` and `outputs[].linkIds`.

An interior delete reuses the `delete_node` kind, so a consumer that predates
`path` cannot reject it. comfy-multi-player 0.3.11 and earlier ignore `path`,
remove the top-level node with the same id, and report the op `applied`. Send
interior deletes only to a doc host on comfy-multi-player 0.3.12 or later.

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
  neither contends with a `set_widget` write on that node. See §3 and §1.8.1
  below.
* Invalid: `field` outside the closed set, or a `value` that doesn't match
  that field's type (and isn't `null`), is rejected at mint time; a missing
  node is rejected at mint time (mirrors `set_widget`'s not-found handling),
  and at replay a since-deleted target is a no-op (delete wins), same as
  every other write.
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
  their canonical digests in `_applied_op_digests`. An identical retry is a
  no-op; the same `op_id` carrying a different canonical payload rejects with
  `op_id_reuse`. A legacy document with only the list keeps its historical
  duplicate behavior. See **Amendment v1.11** (§20) for the canonical form,
  the bounds and the ledger contract. **Uniqueness scope is
  PER-WORKFLOW**: the same `op_id` can exist in two different workflow documents
  without interaction; within one document each op applies exactly once.
* `_applied_ops`, `_applied_op_digests` (and `_widget_stamps`) are apply-time
  bookkeeping, stripped before serialization to disk (`strip_internal`).

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

Gated targets: the `set_widget` rows, the connect-embedded `inputcount` bump
(8.4), and — since amendment v1.2 — a concrete `connect`'s
`("input", to_node, to_slot)`. **Node ids in a target are compared as strings**
(v1.2): ids are legitimately either JSON type, and comparing them raw gave `7`
and `"7"` two registers for one node.

| Scenario | Ruling | Where in code |
|----------|--------|---------------|
| update vs update (same widget) | LWW on `stamp` with `op_id` tiebreak; loser dropped | `_lww_gate` / `_stamp_key` |
| **concurrent `connect` to the same concrete input** | LWW on `stamp` with `op_id` tiebreak, target `("input", to_node, to_slot)`; the loser is dropped whole (no link tuple, no out-link entry) and the winner retires the prior occupant. Amendment v1.2 — previously **undefined** and decided by arrival order | `_apply_connect` (concrete branch) / `_lww_gate` |
| update vs delete | **delete wins**: `set_widget` to a deleted node is a no-op; a `connect` whose destination is gone is a no-op; a `connect` whose SOURCE is gone still claims its input register and leaves that input empty (v1.2 — otherwise the incumbent's survival depends on when the delete arrives); replay never raises on a since-removed target | `_apply_set_widget` (missing node → return), `_apply_connect` (missing endpoint → return) |
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

* **`set_widget`, `connect`, and `disconnect` have subgraph-scoped forms.** For
  `set_widget`, three address forms are accepted and normalize to ONE write
  target: flat promoted (`57.text`, routed through the instance's
  `proxyWidgets`), nested interior (`57/3.steps`), and the flattened UI→API
  alias (`57:3.cfg`). The minted op carries the **resolved** `path` (e.g.
  `["57", "27"]`) plus `inner_widget`, so replay needs no proxyWidgets logic,
  and the LWW target is `("widget", ("57", "27"), "text")` — a flat-form and
  a nested-form concurrent write to the same interior widget converge under
  8.1.
* **`connect` and `disconnect` permit links wholly inside one supported,
  single-instance definition.** Their ops carry the instance `path`, and their
  LWW input identity is scoped by that path. A link that crosses the subgraph
  boundary is still rejected with "a link cannot cross the subgraph boundary";
  a promoted-widget target is still rejected with "promoted widget (a value),
  not a link input". Both operations refuse a shared definition rather than
  silently rewiring every instance that uses it.
* **`delete_node` permits an interior node in one supported, single-instance
  definition.** Its op carries the owner-instance `path`; its node-presence
  identity is scoped by that path, and replay removes incident definition
  links plus their endpoint references. It refuses a shared definition rather
  than deleting from every instance, and refuses a node that backs a promoted
  widget. A later interior `set_widget` on the deleted node is a no-op (delete
  wins). `add_node` still cannot address an interior location from the CLI
  authoring surface.
* An interior **widget** write to a shared definition forks the definition at
  apply time (`engine._isolate_shared_subgraph`): the definition is deep-copied
  under `"sg-" + sha256(def_id + "\x00" + instance_id)[:32]` — deterministic,
  never random — and the instance's `type` is repointed, so two replicas
  replaying the same op produce byte-identical graphs and sibling instances
  are never aliased. Interior connect/disconnect use the refusal above instead.
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

The frontend (ComfyUI_frontend ADR-SUBGRAPH-PROMOTION-0009) keeps a promoted subgraph widget's
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

### What is explicitly NOT done here

* **Not ratified.** See the status callout above.
* **No interior/subgraph-promoted `set_node_field` variant** — §1.8 states
  this is out of scope, mirroring comfy-multi-player#235's own scope.
* **No reconciliation commit against comfy-multi-player** — that is
  follow-up work for once (and if) this amendment is accepted.

## 16. Amendment v1.7 — 2026-09-20 (node title at creation)

### 16.1 `add_node` carries an optional `title`

A spec (and the minted op) may set `title` to a string: a custom display title
for the node, in place of the class's default class-derived title. Before this
amendment `add_node()` took no `title` kwarg, `_build_node()` never wrote a
`title` key into a fresh node, and neither `add-node`'s CLI surface nor the
`apply_specs` vocabulary exposed one — there was no way to name a node at
creation, nor any separate rename/set-title op to do it afterward.

`title` is omitted from the node object entirely when not given (the
pre-amendment shape), so a fresh node keeps falling back to the frontend's
class-derived default exactly as before. When given, it is stamped into
`op.node["title"]` (which stays authoritative for replay, §8.5) and echoed as
an op-level `title` field, mirroring how `mode` (amendment v1.4) is carried.
An op without `title` is exactly the pre-amendment shape, so existing ops
replay unchanged.

This is independent of amendment v1.6's `set_node_field` (§15): that op
writes `title` (among other fields) on a node that already exists, while this
amendment is the only way to set it at `add_node` mint time. The two do not
contend — `add_node` mints the node before any `set_node_field` write could
target it — and neither depends on the other landing.

**No change to §§2-8.** No op kind was added, removed, or re-scoped;
`FROZEN_OPS` / `DEFERRED_OPS` / `BATCHABLE_OPS` are untouched.

## 17. Amendment v1.8 — 2026-09-22 (`disconnect`)

This amendment adds `disconnect` as the ninth frozen op kind so a caller can
remove an input link without replacing the whole workflow or deleting either
endpoint. It is batchable, has the standalone `comfy workflow disconnect`
command, and shares each destination input's last-write-wins register with
`connect`; §1.2.1 defines the wire shape and replay semantics.

Dynamic inputs keep their logical identity instead of treating a shifted
numeric slot as the register. Promoted inputs use `grow.name`; ordinary
autogrow and widget-converted inputs use the creating connect's `grow_id`.
Interior links wholly inside one eligible, single-instance definition carry
the instance path in that identity. Links that cross a subgraph boundary, and
links inside a shared definition, remain refused.

The `grow_id` rule deliberately re-scopes a reconnect-shaped `connect` from
the older concrete `("input", to_node, to_slot)` target to
`("input", to_node, "grow_id", grow_id)`. It is therefore a coordinated
applier upgrade: an older replica that does not recognize `connect.grow_id`
must not participate in mixed-version replay with v1.8. Downstream appliers
must add the same target before moving their pinned vocabulary SHA to this
amendment.

The replay path also retains a winning reconnect that arrives before the
connect which materializes its grown input, then applies that pending write
when the stable `grow_id` slot appears. Tests cover both replay orders for
ordinary autogrow and widget-converted inputs, concrete and promoted inputs,
and path-scoped interior links.

`FROZEN_OPS` and `BATCHABLE_OPS`, the spec and replay dispatch tables, the
standalone CLI, and this contract change together as required by §9.
`DEFERRED_OPS` remains empty; no existing op kind is removed.

## 18. Amendment v1.9 — 2026-09-22 (authorable annotation nodes: `Note` / `MarkdownNote` carry `text`)

### 18.1 `add_node` mints `Note` and `MarkdownNote`

§1.1 / §13.2 stated that `add_node` rejects every UI-only type. Production
agent traces show the agent repeatedly asking for `Note` / `MarkdownNote` and
being refused; every downstream replica (comfy-multi-player applier, frontend
follower) already round-trips those nodes — the applier stores an uncatalogued
positional `widgets_values` opaquely and the follower materializes a note with
its text box — so the catalog refusal on this surface was the only thing
between "add a note" and a note on the canvas.

`add_node` now mints the two **annotation** UI-only types. The authorable set
is `AUTHORABLE_VIRTUAL_NODE_TYPES = {"Note", "MarkdownNote"}` (a strict subset
of `UI_ONLY_NODE_TYPES`). The spec gains one optional, spec-and-node key:

```json
{"op": "add_node", "class_type": "Note", "text": "trigger word: zxc", "as": "n"}
```

* `text` (string, default `""`) becomes the note's single positional widget:
  `op.node.widgets_values == [text]`. The minted node has no sockets
  (`inputs == outputs == []`), `mode`/`flags`/`properties` at their defaults,
  and a layout-assigned `pos`/`size` unless `at` is given. `op.node` stays
  authoritative for replay (§8.5), so a replica with no notion of `Note`
  inserts it verbatim.
* A non-string `text` is `ValueError` at mint time (batch: aborts atomically,
  `applied_count == 0`, §4). An explicit `"text": null` in a spec counts as
  non-string — omit the key for an empty note. `text` on a **catalog** class is likewise
  `ValueError` — catalog widgets are addressed by name through `set_widget`;
  a stray `text` is an error, not a dropped field.
* CLI: `comfy workflow add-node <file> Note --text "..."`. Envelope unchanged.
* The class is decided by **name, before the catalog**: `Note` / `MarkdownNote`
  mint the annotation shape even if a backend happens to register a class with
  that name (the frontend treats those names as editor-only regardless of the
  backend, so the CLI must not turn an agent's note into an executable node).
* Recipe `${param}` substitution (`apply --param`, `foreach`) does **not**
  interpolate `text`: a note is prose for a human reader, so
  `"text": "seed is ${seed}"` applies verbatim and does not require a declared
  `seed` param. Every other add_node key still interpolates.
* Execution: unchanged. `workflow_to_api` owns the exclusion of UI-only types
  from the API prompt and keeps dropping these nodes; a test pins the exact
  shape `add_node` mints against the converter.
* Telemetry: `text` joins the redacted argument names (it is user prose, like
  a prompt), so a `--text` value never reaches the tracking payload.

`Reroute`, `PrimitiveNode`, `GetNode`, `SetNode` remain refused
(`node_not_found`, `details.ui_only: true`): they carry data flow the converter
resolves specially, and authoring them here would need link semantics this
surface does not model.

### 18.2 `capture` deliberately unchanged

§13.2's capture behaviour stands: `capture` still skips UI-only nodes,
including `Note` / `MarkdownNote`, and reports them as `ui_only_node_skipped`
warnings — worded as "skipped by design", not "apply cannot mint them", since
the latter is no longer true for the two annotation types. The recipe
rebuilds the executable graph, not the canvas decoration. Capturing notes into
recipes (now that `apply` would accept them) is a possible follow-up; it is not
part of this amendment so a recipe captured before v1.9 and one captured after
stay byte-identical.

### 18.3 `replace_ops` carries `text` on the spec half

`replace_ops` (behind `templates fetch --emit-ops` / `replace`) emits
dual-shape `add_node` entries: the op half carries the full `node`, the spec
half is what `apply_specs` replays. For a `Note` / `MarkdownNote` the spec
half now includes `text` (= the source node's `widgets_values[0]` when it is a
string, else `""`), so replaying the emitted batch through `apply` reproduces
the note body instead of reminting every note as `""`.

### 18.4 `set_widget` on a note

A minted note has no named widgets, so `set_widget` on it is refused with a
targeted message ("its body is `text`, set at add time; delete and re-add"),
instead of the generic "(none — all inputs are links)". Editing an existing
note's body in place is a follow-up (it needs a positional write path every
replica agrees on, like the `legacy_primitive` one); it is out of scope for
v1.9.

**No change to §§2-8.** No op kind was added, removed, or re-scoped;
`FROZEN_OPS` / `DEFERRED_OPS` / `BATCHABLE_OPS` are untouched. Downstream repos
pinning this document by SHA move the SHA together with their comfy-cli pin.

## 19. Amendment v1.10 — 2026-10-25 (`add_node` reuses an existing subgraph definition)

An `add_node` spec may use the **exact definition id** of a subgraph already in
the workflow when that definition also has at least one top-level host instance.
This is an extension of the existing `add_node` kind, not a new operation kind.
Unknown ids and definitions with no reusable top-level host still fail closed as
`node_not_found`; definition names are not an authoring alias.

The first matching top-level host in document order is the authoritative schema
template. Legacy hosts whose `type` is the definition name are resolved through
the same definition index as the rest of the engine. The fresh host copies the
template's `inputs`, `outputs`, `properties`, `widgets_values`, optional
`widgets_values_form`, and valid saved `size`. It clears every input/output link,
resets `flags`, `order`, and `mode`, mints a fresh collision-free node id, and
uses the requested position/title. Malformed host containers fail closed rather
than emitting a partially shaped node. The copied host state is independent;
the definition is deliberately shared.

The minted op carries the complete authoritative `node` payload as before and
adds:

```json
{
  "subgraph_reuse": {
    "definition_id": "2454ad83-157c-40dd-9f19-5daaf4041ce0",
    "existing_instance_paths": [["88"]]
  }
}
```

`existing_instance_paths` is captured before the new host is applied and
includes nested live paths that resolve to the reused definition. It is
conflict metadata: `detect_conflict` flags this add against a concurrent
interior `set_widget`, `connect`, `disconnect`, `delete_node`, or promoted-input
repair whose mutated definition path is that instance path or a descendant.
Without that flag, replaying the interior mutation before versus after the new
shared instance can fork/refuse in different orders (§8.7) and silently diverge.
The merge consumer must therefore ask for an order for such a pair; all other
`add_node` identity/idempotency rules remain unchanged.

Batch layout uses the same normalized saved host size as the emitted node,
including LiteGraph's object form (`{"0": width, "1": height}`), so an
unpinned sibling in the same batch is not placed through the reused host.

No operation kind is added or removed. `FROZEN_OPS`, `DEFERRED_OPS`, and
`BATCHABLE_OPS` are unchanged. Downstream consumers that move their pinned
vocabulary SHA to v1.10 must preserve `subgraph_reuse` conflict metadata even
though replay itself continues to insert `op.node` opaquely.

## 20. Amendment v1.11 — 2026-10-24 (canonical op identity: `op_id` binds one payload)

**`op_id` identifies one immutable semantic-op envelope, not just one delivery.**
Before this amendment `apply_op` dropped any op whose `op_id` was already in
`_applied_ops` without looking at the payload, so a sender that reused an id
with a changed payload had its second edit silently discarded — and two
replicas that received the pair in different orders ended up with different
documents and no error on either side. This mirrors comfy-multi-player schema
**Amendment A8**; the two implementations must agree byte-for-byte on the
canonical form, because the digest is the only thing that distinguishes an
identical retry from reuse.

### 20.1 The canonical form

Compact JSON over the **whole envelope**:

* object keys sorted by **Unicode code point** at every depth; array order
  preserved; no insignificant whitespace (`,`/`:` separators, no spaces). The
  same caveat as 8.1 applies: for ASCII, JS UTF-16 `<` agrees with code-point
  order, but above the Basic Multilingual Plane it does not, so an
  implementation must sort by code point and not rely on
  `Object.keys(...).sort()`'s default comparator. Every key an op envelope
  carries today is ASCII, so this only bites a dict widget whose own keys are
  not;
* **numbers rendered as ECMAScript renders them** (`Number::toString`), not as
  Python's float repr does. An integral float is `20`, not `20.0`; negative zero
  is `0`; an exponent is `1e-7`, not `1e-07`. This is load-bearing rather than
  cosmetic: the counterpart builds the canonical form with `JSON.stringify`, so
  without it a FLOAT widget op relayed through JS digests differently on the two
  sides and an identical redelivery is rejected as `op_id_reuse`;
* **every non-finite number renders as `null`**, which is what `JSON.stringify`
  does. The alternative — rejecting the op — made a widget whose *current* value
  is `NaN`/`Infinity` impossible to overwrite, because the op captures that
  value in `old` (and `json.load` accepts both spellings on the way in);
* a **lone surrogate** renders as a `\udXXX` escape, matching ES2019
  well-formed `JSON.stringify`. Emitted raw it cannot be UTF-8 encoded at all,
  so the digest died with a bare `UnicodeEncodeError` outside the
  `malformed_op:` prefix callers classify errors by.

The ledger records the lowercase SHA-256 hex digest of that string.

### 20.2 The ledger contract

* `_applied_op_digests` maps `op_id` → digest, beside `_applied_ops`. Both are
  stripped by `canonical()` and `strip_internal()`, and `reset_doc` clears both
  as one history barrier.
* **Rejects before mutation.** `op_id`, `_applied_ops` and `_applied_op_digests`
  all arrive from untrusted JSON and are all validated before any handler runs:
  a non-string `op_id`, or a ledger that is not an object/array, is
  `malformed_op`. So is a payload canonicalization refuses. A document that
  arrived without a ledger is never left carrying an empty one by a rejected op.
* **A deferred op is already bound.** A `set_widget` waiting in
  `_deferred_widget_ops` for a selector to reveal its target has not reached
  `_applied_ops`, so the gate checks the deferred queue too; a second payload on
  that `op_id` rejects instead of overwriting the pending entry.
* **The enriched envelope is what is bound.** A promoted or `legacy_primitive`
  `set_widget` only learns the host's materialized `host_widgets_values` by
  reading back the document the write produced, so the ledger entry is corrected
  to the envelope the caller actually broadcasts. If that enriched envelope is
  not canonical the earlier digest stands: the document is already written, and
  reporting failure for an applied op is worse than holding a digest a
  redelivery will refuse in the pre-mutation branch.
* **Legacy documents are untouched.** One carrying `_applied_ops` and no
  `_applied_op_digests` keeps the historical drop-the-duplicate behavior, and
  deliberately without digesting the replayed payload first — a non-canonical
  replay must still be the documented no-op rather than a new error.

### 20.3 Bounds

Canonicalization is bounded to **64 levels of nesting**, **4,096 entries per
collection** and **262,144 approximate cost units**. These are not chosen here:
they are mirrored exactly from `comfy-multi-player`'s `src/limits.ts` at
`221b2b2e470c79afaface24b1edcb53cdafc8922` (`MAX_PAYLOAD_DEPTH`,
`MAX_COLLECTION_ENTRIES`, `MAX_OP_COST`), cited by SHA rather than by branch.

They exist only so a hostile or runaway payload cannot pin a replica inside
canonicalization, and every ordinary op has to pass. "The digest is computed on
every apply, not only on a reuse comparison" is true of **both** sides — CMP's
A8 canonicalizer charges the whole envelope in `applyOps` before its dedupe gate
— so a looser bound on one side rescues no edit. It only moves where an
oversized op fails, from a local `malformed_op` to a `malformed_op` from the
shared package after this side already accepted the payload. A comfy-cli-only
budget was briefly carried here (1,048,576 / 33,554,432) and is withdrawn as
exactly the cross-implementation validation drift one shared op package exists
to prevent. If a real edit needs more room, widen it in `comfy-multi-player` and
move this pin.

Measured headroom at the shared values: the widest op this vocabulary mints from
a real workflow fixture is ~1.6K cost units over 13 entries, under 1% of the
budget. The three edits previously said to need a looser bound bind only well
past anything observed — `clear`/`reset_doc` at more than 4,096 nodes in one
document, a hub `delete_node` at more than 4,096 incident links on one node, and
a widget string at 131,064 characters when the op also captures an equal-sized
`old` (262,133 without one). The largest workflow in the comfy-cli fixtures has
11 nodes.

The asymmetric case is the one to keep in mind: `{value: 25}` followed by the
same `op_id` carrying `{value: 30}` rejects, while the same payload with its keys
in a different insertion order is an ordinary no-op.

**No change to §§1, 3-8.** No op kind was added, removed, or re-scoped;
`FROZEN_OPS` / `DEFERRED_OPS` / `BATCHABLE_OPS` are untouched. §2's identity
bullet now cites this section. Downstream repos pinning this document by SHA
move the SHA together with their comfy-cli pin, and must move their CMP A8 pin
with it.

## 21. Amendment v1.12 — 2026-10-31 (a batch is bounded at 1,024 ops)

**A batch carries at most 1,024 ops.** `comfy-multi-player`'s `src/limits.ts` at
`221b2b2e470c79afaface24b1edcb53cdafc8922` fixes `MAX_OPS_PER_BATCH` at 1024 and
refuses a longer batch **whole** — `applyOps` checks it before any op is
processed and returns a `malformed_op` rejection for every op in the batch, and
`inspectOps` raises `malformed_op` from the storage preflight. comfy-cli had no
counterpart, so `apply_specs` would accept and apply a batch of any length and
`replace_ops` would hand a caller one, and the refusal arrived from the shared
package only after this side had committed to the write.

That is the same mint-then-refuse divergence v1.11 §20.3 closed for the three
payload bounds, one level up: §4's abort-remainder contract is about which ops in
a batch apply, and an over-long batch has no answer under it at all because none
of them do. The bound is therefore mirrored, not chosen here, and cited by SHA
rather than by branch for the same reason §20.3 gives. If a real edit needs a
longer batch, widen it in `comfy-multi-player` and move this pin; never widen it
here alone.

Where it is enforced, and the two different refusals:

* **`apply_specs`** checks it TWICE, because one spec is not one op. On entry
  it raises `malformed_op: batch of <n> specs exceeds the 1024-op limit;
  rejected before any op was processed` — before the `at` coercion and before
  the position planner, both of which are per-spec work, so the message's
  promise is literally true. That check cannot be the whole bound:
  `set_dynamic_group_rows` expands into a row-count write plus one `set_widget`
  per row field, so a spec list within the bound can still mint a batch
  past it. After the loop, and still inside the batch's discard handler, it
  raises `malformed_op: batch mints <n> ops, more than the 1024-op limit`;
  `applied_count` is 0 there like every other mid-batch refusal.
* **`replace_ops`** raises **`NotExpressibleError`**, the existing "there is no
  legal batch for this graph" signal, rather than a bare `ValueError`. Its
  reasons are now *structure no frozen kind can create*, *a translation longer
  than the bound*, and *one op whose own payload is out of §20.3's bounds* —
  that last one because these ops are handed to a caller without ever passing
  through `apply_op`, so nothing else canonicalizes them. All three want the
  same caller behaviour, which both production callers already implement —
  `templates fetch --emit-ops` and `generate`'s emit path each report
  `ops_skipped` beside the file they still wrote. The count binds on the SUM of
  `old` and `new`, because §8.8's replacement shape emits a `delete_node` per
  node already in `old` before the adds.

Measured headroom, and what the measurement does NOT cover. Across all 735
workflows in `Comfy-Org/workflow_templates` — the gallery
`templates fetch --emit-ops` reads — the longest batch `replace_ops` actually
mints is **50 ops**, 4.9% of the bound. That measured the `new` side only. The
`old` side is whatever document the caller is replacing, which is user data and
is bounded by nothing here: a fetch or an emit into a path already holding a
workflow of ~1,000 nodes can exceed the bound on the `delete_node` half alone
while the incoming template is tiny. Both callers degrade rather than fail on
that case (`ops_skipped` beside a file they still write). 447 of those templates carry a
subgraph definition, a canvas group or a reroute point and are refused as
inexpressible before size is considered; across all 735, nodes+links peaks at
212, so even if every one of them became expressible the worst case stays a fifth
of the bound. The largest expressible workflow in comfy-cli's own fixtures mints
3 ops.

The one document-sized payload this bound does **not** cover is a standalone
`clear` or `reset_doc`, which enumerates every node id inside ONE op rather than
as one op each. Those are bounded by §20.3's 4,096-entry collection cap instead,
and §1.5/§1.6 keep them out of `BATCHABLE_OPS`, so no large document turns into a
long batch behind a caller's back.

**No change to §§1-3, 5-8.** No op kind was added, removed, or re-scoped;
`FROZEN_OPS` / `DEFERRED_OPS` / `BATCHABLE_OPS` are untouched. §4 gains an outer
precondition — a batch must be within the bound before abort-remainder has
anything to say about it — and is otherwise unaltered. Downstream repos pinning
this document by SHA move the SHA together with their comfy-cli pin.

## 22. Amendment v1.13 — 2026-10-31 (v1.11 corrections: canonical form and ledger edges)

Six corrections to Amendment v1.11, all found by review of the implementation
rather than by a failure in the field. None of them re-scopes an op kind and none
needs a change in `comfy-multi-player`; each is comfy-cli coming into line with
what v1.11 and CMP A8 already say.

### 22.1 An integer past 2^53 is spelled as the double it becomes

**§20.1's "numbers rendered as ECMAScript renders them" covers integers too, and
comfy-cli did not implement it that way.** It rendered every `int` with Python's
`str`, which is exact at any magnitude. The counterpart has no such type: JSON
carries no integer, `JSON.parse` reads every numeric literal as an IEEE-754
double, and CMP's canonicalizer refuses a `BigInt` payload outright
(`applier.ts`: `rejectBigIntPayload`). So past 2^53 the other side holds the
ROUNDED value and `Number::toString` spells that.

The consequence was the failure §20.1's float rule exists to prevent, one type
over. A seed at an INT widget's `0xffffffffffffffff` maximum spelled
`18446744073709551615` here and `18446744073709552000` after a relay through JS,
and `10**21` spelled `1e+21` there — so an **identical** op echoed back through
the shared package digested differently and was refused as `op_id_reuse`.

The rule, which changes nothing inside the exact range:

* `-2^53 <= n <= 2^53` keeps its exact digits — the two spellings already agree
  there, and this is every integer any widget, node id, link id or stamp
  carries;
* outside it, the value is rendered through the same double path as a float, so
  it reads as `JSON.stringify(JSON.parse(literal))` would;
* past the double range entirely (`10**400`), `JSON.parse` yields `Infinity` and
  `JSON.stringify` renders `null`, matching §20.1's non-finite bullet;
* an integer **object key** is the one place the two diverge, because a property
  name is coerced with `String()` rather than rendered by `JSON.stringify`:
  there the non-finite end is `"Infinity"` / `"-Infinity"`. Only a
  programmatically built op can reach a non-`str` key at all — parsed JSON
  always has string keys.

The **cost charge is unchanged** at a flat 8 units per number, deliberately:
`src/limits.ts` documents the model as "8 per numeric/leaf", so charging a huge
integer by its digit count would refuse a payload CMP accepts — the same
validation drift §20.3 withdrew a comfy-cli-only budget to avoid.

CMP already produces these spellings, because it canonicalizes with
`JSON.stringify`; this records that comfy-cli now agrees with it. The only
behavioural effect is on an op carrying an integer past 2^53, which now digests
as its JS spelling.

### 22.2 The cost bound refuses at the charge, not after the render

§20.3's budget exists to cap the WORK a hostile payload costs, and it was
checked only at the END of each `_canonical_value` call. A string is charged by
length and then handed to the renderer, which runs `json.dumps` plus a surrogate
scan over the whole thing and allocates up to ~6x its size for control-heavy
input — so an oversized string, or an oversized object key at the `+= len(name)`
charge, was fully rendered before being refused. The bound therefore did not
bound the work.

Charging and refusing at the same point fixes that, and it does not change
WHETHER a payload is refused, because the budget only ever grows: every payload
that exceeded at the end exceeds at the charge that crossed the line, and every
charge that crosses the line is followed by an end that is also over. The shared
package already has this property for the same reason — `opBoundsRefusal` walks
and charges the payload before `normalize` renders anything.

What it can change is WHICH refusal a payload that breaks two bounds at once
reports. An object with an oversized key AND a value nested past the depth limit
used to charge every key without checking, reach the values, and report
`payload_too_deep:`; it now reports the cost budget at the key. Both are
refusals of the same payload, and the cost-first order is the one CMP's
depth-first walk already produces, so this moves comfy-cli toward the
counterpart rather than away from it. No single-violation payload changes its
code.

### 22.3 Object keys that collide after coercion are `malformed_op`

§20.1 renders an object key as "its string form, matching ECMAScript property
coercion", and that coercion is not injective: `1` and `"1"`, `True` and
`"true"`, `None` and `"null"`, `1.5` and `"1.5"` are distinct Python keys naming
the same JS property. Emitting both members produced an object literal no JS
object can hold, and because the member sort is stable the digest of such a dict
then depended on INSERTION order — so the same logical payload could digest two
ways and §20's whole "one `op_id`, one payload" guarantee had a hole in it.

Such a pair is refused as `malformed_op` rather than merged: there is no intent
in it a merge would preserve. Only a programmatically built op can reach the
case at all, since parsed JSON always has `str` keys, so no wire payload
changes behaviour.

### 22.4 A `null` digest entry is malformed, not a legacy ledger

§20.2's "legacy documents are untouched" rule is keyed on a document having
`_applied_ops` and **no** `_applied_op_digests` entry for the op. comfy-cli
decided that with `recorded is None`, which cannot tell a MISSING entry apart
from an explicit `null` one, and the up-front validation checks the ledger's
type rather than its values — both arrive from the same untrusted JSON.

So a document carrying `{<op_id>: null}` read as a legacy list-only ledger and
returned a changed-payload redelivery as a silent no-op, which is precisely the
divergence v1.11 exists to stop. Every other non-string value failed closed on
the digest comparison, so `null` was the one shape that failed open.

The legacy case is now decided by MEMBERSHIP, and a present-but-non-string
digest entry is `malformed_op` — not `op_id_reuse`, because the document's
bookkeeping is broken rather than the sender's id being reused. It is refused on
read rather than by walking every ledger entry on every apply.

### 22.5 The deferred-queue digest is taken defensively

§20.2's deferred-queue gate digested the QUEUED entry with the strict
canonicalizer, under a comment claiming it could not raise because a queued op
was canonicalized on the way in. `_deferred_widget_ops` comes off the document
and is never validated on load, so that does not hold: an oversized or deeply
nested queued entry raised its own `malformed_op:` / `payload_too_deep:` against
the INCOMING op, blocking a well-formed op with an error naming a payload that
is not the one being applied.

The queued entry is now digested with the same tolerant helper the retry loop
uses, and a `None` digest is a mismatch — correctly, because the incoming op has
already been digested at that point and is therefore canonical, so an
uncanonical queued entry cannot be the same payload. The refusal is
`op_id_reuse`, which is what it is.

### 22.6 Kind and `definitions` are validated before the ledger is seeded

§20.2 promises that "a document that arrived without a ledger is never left
carrying an empty one by a rejected op". Two validations sat on the wrong side of
that line: the read of the op's `op` field, and the refusal of `definitions` on
a non-`insert_workflow` op, both ran after `_applied_ops` had been seeded and
both raise outside the handler rollback. So a first-seen op with no `op` key, or
a non-insert op carrying `definitions`, was correctly rejected and left
`{"_applied_ops": []}` behind on a document that had no ledger at all.

Both now run before the seed. No message or code changes.
