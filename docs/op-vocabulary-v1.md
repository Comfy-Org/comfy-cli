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
| `reset_doc` | no | (deferred) | Reset the whole document to an empty baseline |

Batchable = the kind is accepted by `apply_specs` (the `workflow apply` /
`workflow foreach` batch surface). `clear` and `reset_doc` rewrite the whole
document, so they are standalone-only: a batch containing `clear` is rejected
atomically with error code `workflow_clear_not_batchable` and a hint naming the
standalone `comfy workflow clear` command. Nothing from such a batch is applied.

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

### 1.6 `reset_doc` — standalone only, deferred

Defined here; **implementation is deferred to the bulk-writers ticket**.
`apply_op` currently rejects it (`unknown op 'reset_doc'`), and the contract
tests pin that it stays rejected until it is un-deferred by amendment.

Semantics when implemented: replace the entire document with the empty baseline,
including apply bookkeeping — unlike `clear`, which preserves the id high-water
marks and the applied-op history. Because it erases replay history, it is a
history barrier: ops minted against a pre-reset `base_version` do not replay
across it. Guard semantics: the CLI surface requires an explicit `--confirm`
flag; without it the command fails closed and applies nothing. Not batchable,
for the same reason as `clear`.

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

## 8. Amendments

* Post-freeze changes require a **versioned amendment section** appended to
  this document (`## Amendment v1.x — <date>`), stating what changed and why.
  Silent edits to frozen sections are not valid; the contract tests pin the
  frozen table against the code.
* Downstream repos cite this document by commit SHA and upgrade by moving the
  SHA, never by tracking a branch.
* Adding, removing, or re-scoping an op kind requires updating `FROZEN_OPS` /
  `DEFERRED_OPS` / `BATCHABLE_OPS`, the dispatch tables, and this document in
  one commit — `tests/comfy_cli/test_op_vocabulary_contract.py` fails otherwise.
