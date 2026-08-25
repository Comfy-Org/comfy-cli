# DESIGN: Knowledge Delivery

**Status:** proposed. Phase 0 shipped in PR #784; phases 1–3 are not built.
**Date:** 2026-08-25

## Architecture Decision: knowledge is a place the agent reads

Curated knowledge today only reaches an agent by riding along inside the
`data` of a discovery command the agent ran for some other reason. That makes
delivery conditional on things unrelated to whether the agent needs to know
something. The decision recorded here is to stop treating enrichment as the
primary channel, and to make model choice a question the agent asks directly
against a corpus that survives context loss.

**Rationale:** the enrichment channel has four independent failure modes, and
an agent hits them without doing anything wrong. See "The four
disappearances" below.

**Alternative (rejected):** write more rules telling the agent to read the
block. Already tried. `SKILL.md` rule 1 has said "read `knowledge.picks`
before choosing" since the enrichment landed, and the eval below still chose
a rank-2 model. Advisory text loses to whatever is actually in front of the
model at the moment it decides.

## The failure this comes from

A staging eval asked: *"Generate a video from first and last frames — I have
both of them. What is the best way to do this?"* The bundle ranks
`minimax-h3` first for `first-last-frame`. The agent answered Seedance 2.5.

The Langfuse trace shows the knowledge was correct and present the whole
time, and never in front of the model when it chose:

| step | call | what happened |
|---|---|---|
| 0 | `templates ls --tag "Image to Video" --select rows.#.name` | `--select` returns before enrichment runs, so no picks. 1,610 of 21,221 bytes emitted. |
| 1 | `get_template api_bytedance_seedance1_5_flf2v` | guessed name, not in `picks`. Failed on a doc-host `LoadImage` error. |
| 2 | `get_template video_ltx2_3_ia2v` | guessed name, not in `picks`. Same doc-host error. |
| 3 | `generate list --select models.#.id` | `--select` again, so no picks. |
| 4 | `nodes search "first last frame video"` | **first and only knowledge block.** `picks[0]` = `minimax-h3`. |
| 5 | `show_node ByteDance2FirstLastFrameNode` | already committed to rank 2. |
| 8 | — | step 4's tool result elided from history. `picks` gone from context. |
| 9 | final answer | recommends Seedance 2.5. |

Two contributing facts worth separating from the knowledge system itself.
The doc-host errors at steps 1 and 2 are an unrelated bug
(`createNodeMap(LoadImage): widgets_values has 2 entries but widget_order
names only 1`), but they matter here because `picks` names *templates*, so a
broken `get_template` makes the ranked answers unreachable and pushes the
agent onto the node path where rank is invisible. And an agent that meets an
error abandons the path: after two failures the agent never returned to
templates.

## The four disappearances

1. **Projection.** `--select` replaces `data` with whatever the path
   evaluates to, and the command returns before `knowledge.attach` is ever
   called (`command/templates.py:646` returns; `attach` sits at `:669`).
   This is deliberate and tested (`test_ls_select_is_never_enriched`), not a
   bug. A projection returns exactly what was asked for.
2. **Qualification.** `attach()` returns early on `qualified=False`, so an
   unfiltered listing carries nothing. Also deliberate and tested: a curated
   row picked out of 3,655 listed nodes reads as the answer to a question
   nobody asked.
3. **Eviction.** A knowledge block is a tool result, and tool results get
   elided. Step 4's 8,995 bytes were replaced by a head snippet plus a
   `recall_ref`, and the head happened to contain node inputs rather than
   `picks`.
4. **Burial.** When the block did arrive it led with the full Wan model card
   (8 pitfalls, 8 routing rows, 2 warnings, 2 corrections) because the node
   search matched `WanFirstLastFrameToVideo`. `picks` sat after all of it.
   `_fit` only trims when the block exceeds `MAX_BLOCK_BYTES`, and this one
   fit, so nothing was dropped.

Each is individually defensible. Together they mean knowledge arrives when
the shape of an unrelated call happens to allow it.

## Phase 0 — shipped (PR #784)

Make the ranked answer something the agent can ask for by name.

- `comfy knowledge pick` takes an **optional** argument. The capability
  vocabulary was previously reachable only by failing on purpose, since the
  argument was required and the full list came back on the error path as
  `details.known`. The argument help said exactly that.
- A **miss is an answer, not an error**. It now returns an `ok` envelope with
  `zero_hit`, the echoed `query` clipped to `MAX_QUERY_CHARS`, a `nudge`, and
  the ids to retry with. This matches the shape an enrichment block already
  uses, so one concept has one shape. A hit carries `zero_hit: false` so the
  field is always readable.
- `knowledge_unknown_capability` retired from the registry, since it became
  unreachable and the never-raised check would fail.
- `SKILL.md` rules 1 and 4 rewritten to route model choice through
  `knowledge pick` (user's own phrasing first) and `knowledge resolve` (when
  the user already named a model).

**Why a miss stopped being an error.** Rule 1 now tells the agent to throw
the user's own words at `pick`, and prose misses often. If that returns
exit 1, the agent's *correct* first move looks like a failure, and the trace
above shows what an agent does with failures.

## Phase 1 — `validate` as a correctness backstop

Attach knowledge to `validate` (`command/workflow.py:1674`), which already
walks the graph, detects partner nodes and sets `spends_credits`. It knows
exactly which models were chosen, and the agent must call it before running.
It is the last gate before credits burn.

**Correctness grade only.** Emit an advisory when a chosen model is
`deprecated`, has a `superseded_by`, or carries a pitfall that applies to
this exact graph. **Never rank.**

**Rationale for excluding rank:** a rank advisory arrives after the graph is
built, so acting on it means tearing the graph down and rebuilding. Rank is
a preference, and both models work. A deprecation advisory is a one-node
class swap, which is a small diff; a rank advisory can imply a different
route and a different graph topology, which is not. Churn is only worth it
when the alternative is a wrong result the user pays for.

## Phase 2 — dual emission: a machine index plus a readable corpus

Emit the bundle in two forms from one source.

**Keep the JSON bundle as the machine index.** It carries three things
markdown cannot: `picks[]` with `rank`/`route`/`template`/`caveat`, which is
a decision structure rather than a document; JSON Schema validation, which
catches drift and is exercised in tests; and reverse indexes from node class
and template name back to model id, which is the only reason `find_nodes` on
`WanFirstLastFrameToVideo` surfaced the Wan row at all. Deprecation as
`status` + `superseded_by` is a chain you can walk rather than a sentence you
hope gets read.

**Additionally emit an Open Knowledge Format bundle.** OKF is Google's v0.1
spec (June 2026): a directory of markdown files, one concept per file, path
as identity, YAML frontmatter with exactly one required field (`type`). Map
directly onto the existing planes:

```
knowledge/
├── index.md
├── models/
│   ├── wan.md              # type: model
│   └── minimax-h3.md
└── capabilities/
    └── first-last-frame.md # type: capability
```

Existing fields already map: `as_of` → `timestamp`, model id → `title`,
capability description → `description`.

**Rationale:** this addresses disappearances 1, 3 and 4 at once. Files are a
place, not an attachment, so nothing can project them away, no filter gates
them, and eviction of one tool result does not take the corpus with it.
One file per concept also gives the self-containment the LLM Wiki pattern
argues for: the failure above elided the Wan card and every pick as a single
unit, where per-concept files would let the agent re-read only what it needs.
Markdown also carries caveats properly. The reason the agent took rank 2 is
H3's caveat ("there is no OSS `video_minimax_h3_flf2v` file — that filename
is API-only"), and an integer rank cannot express that. Ranking without room
for that nuance actively misleads.

**Alternative (rejected): replace the JSON bundle with OKF.** OKF explicitly
defines no ranking or recommendation mechanism and no formal schema
validation. Those are the two things this system most needs.

**Alternative (rejected): adopt the LLM Wiki's ingest/query/lint
self-maintenance loop.** The bundle encodes accountable human judgments —
"do not fetch 5B, it is hard to use", credit rates verified on a date, an
explicit note that the source document contradicts itself about Wan 2.7. An
agent auto-ingesting into that would sand off exactly the opinions that make
it worth having. That loop suits a corpus you accumulate, not a curated
recommendation set someone signs their name to.

**Alternative (rejected): bake a capability→rank-1 table into `SKILL.md`.**
Cost is not the objection: measured at 74 tokens (ids only) to 638 tokens
(with caveats) against a file already at 68,938 bytes / ~17,234 tokens, and
it lands in the cached prefix (the trace shows 339,109 cache-read tokens
against 44,805 cache-creation). Staleness is the objection. `SKILL.md` is
written at `comfy skills install` time (`skills/__init__.py:547`), while the
bundle has a URL and a TTL and moves underneath it. A stale table names a
rank-1 model the bundle no longer ranks first, which is worse than no table
because it is confidently wrong. Phase 2 gets the same benefit without
freezing anything.

## Phase 3 — measure coverage from traces, not from `zero_hit`

**`zero_hit` will under-report coverage gaps and must not be the curation
metric.** Once an agent has seen the capability ids it maps requests to the
nearest one, because models are built to use the options in front of them.
Requests you do not cover get absorbed into the closest thing you do cover,
`zero_hit` trends toward zero, and the data claims complete coverage. Even
with no list shown, the query string is the agent's *translation* of user
intent into guessed vocabulary, so it measures the agent's mapping rather
than user demand.

**Use the trace instead.** `agent.turn.input` carries the user's verbatim
words, captured before any tool ran, with no agent mediation. Join three
things per turn:

1. `agent.turn.input` — what was actually wanted.
2. the `knowledge pick` span output — whether knowledge had an answer.
3. the built graph or `agent.turn.output` — what the agent actually chose.

Gaps fall out of turns where intent is clear and no `pick` span ever hit, or
where the built model diverges from rank 1. The eval above is exactly that
shape and is detectable with no cooperation from the agent.

**What `zero_hit` is for:** an agent control signal. It tells the agent
nothing is curated so go check the live lists, which is what `SKILL.md`
rule 3 already says. Keep it for that.

## Explicitly not changing

**List enrichment stays as it is.** Unfiltered listings already attach
nothing via the `qualified` gate, which is correct. A *filtered* list named a
subject and is therefore a query, so it should keep attaching. The real axis
in this codebase is qualified vs unqualified, not list vs search, and that is
the better split.

**`--select` stays unenriched.** A projection returning exactly what was
asked for is a contract worth keeping. Phase 2 removes the need to break it,
because the corpus is readable independently of any payload.

## Open questions

1. Who generates the OKF bundle — the knowledge repo at build time, or
   `comfy knowledge` at fetch time? Fetch time keeps it as fresh as the JSON
   and costs a write to the cache directory.
2. Does the agent get told where the OKF bundle lives via `SKILL.md`, or does
   `comfy knowledge status` report the path? The latter is one less static
   string to go stale.
3. Should `knowledge resolve` also return `zero_hit` instead of
   `knowledge_unknown_model`, for symmetry with `pick`? Currently the two
   verbs disagree about what a miss is.
4. Does Phase 2 make `data.knowledge` enrichment redundant enough to shrink?
   Probably not; it is genuinely useful when it fires. It just stops being
   load-bearing.
