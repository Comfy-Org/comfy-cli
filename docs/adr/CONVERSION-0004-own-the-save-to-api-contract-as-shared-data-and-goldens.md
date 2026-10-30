# ADR-CONVERSION-0004: Own the save-to-API contract as shared data and goldens, not shared code

Date: 2026-10-28

## Status

Accepted 2026-10-30.

The Decision section below binds two repositories other than this one:
`ComfyUI_frontend` generates the widget data, and `comfy-cloud-mcp-server`'s
converter consumes the bundle instead of its own tables. Neither affected path
carries a per-path `CODEOWNERS` entry, so there was no single named reviewer to
request, and acceptance was taken by the author in their capacity as a code
owner of both. The two obligations are tracked as issues in those repositories
(see Notes) and remain open to pushback there on implementation and cost; what
is settled here is the design.

## Context

Converting a saved workflow to the API prompt the backend executes is
implemented four times: the frontend's `graphToPrompt()`
(`ComfyUI_frontend/src/utils/executionUtil.ts`), `@comfyorg/api-to-workflow`
(sourced from `comfy-cloud-mcp-server/src/converter/`), comfy-cli's
`convert_ui_to_api` (`comfy_cli/workflow_to_api.py`), and — for the adjacent
save-format slot-write contract — a Go implementation in a separate internal
repository. The frontend is ground truth because it is the only one with the
live LiteGraph runtime.

The whole of `graphToPrompt()`'s widget contract is eleven lines: iterate
`node.widgets`, skip a widget with no name or `options.serialize === false`,
apply `serializeValue`, wrap arrays. Everything else in that function is link
resolution and DTO flattening. The only fact a headless converter cannot
derive is therefore the content and order of `node.widgets` for a given class.
That fact is a per-class property of the installed frontend extensions, and it
is currently approximated by hand three times — as TypeScript tables in the
package, as `object_info`-flag predicates in comfy-cli, and (unread here) in
Go. Each approximation has drifted, in opposite directions, undetected: the
package's `PreviewAny` entry emits three keys that no golden carries and cites
a source file that no longer exists, while comfy-cli covers six `audioUI`
classes to the package's two and emits no injected widget at all.

Drift was undetectable for three structural reasons. The only cross-
implementation job compares two implementations to each other rather than to
the oracle, so it is silent whenever both are wrong — and the two share a
lineage, making correlated error the likely kind. It pins comfy-cli to a SHA
that is 139 commits off main and not an ancestor of it, and asserts only that
the pin matches itself. And the 500 goldens, captured 2026-09-11, are
evaluated against an `object_info` that a daily cron rewrites, so 28% of the
corpus fails for catalog movement alone, teaching reviewers to read defects as
staleness.

## Decision

The save-to-API contract is owned as **versioned data plus a golden corpus**,
generated from the frontend and consumed by every other implementation. It is
not owned as shared converter code.

- **Generated, frontend-owned data.** `widget-layout.json` (per class, the
  ordered `node.widgets` with each widget's slot index, `promptEmit` derived
  verbatim from the oracle loop's own emit condition — `options.serialize !==
  false` **and** a non-empty `name`, because `graphToPrompt()` skips an unnamed
  widget regardless of its serialize flag — and an optional `serializeValue`
  in one of three closed forms) and `widget-type-census.json` (the registered
  widget-type set, which is literally `useWidgetStore().widgets`) are emitted
  by jobs in `ComfyUI_frontend` CI. A hand edit to either is rejected.
- **Jointly owned, evidence-bearing data.** `classifier-vectors.json` carries
  the ordered classification rule procedure as executable cases. A new vector
  is admitted on evidence alone, never gated on who currently passes it: each
  implementation records its own expected-fail set **in its own repo**, next to
  its own ratchet, and that per-repo set may only shrink. The shared file
  therefore holds no cross-implementation gap ledger — adding a vector that
  every implementation fails today is the normal way a new defect class enters
  the contract, and it reddens each repo only until that repo records the
  expected fail. (An earlier draft put a `failsToday` ledger in the shared
  file. That both contradicted the local-ratchet rule below and made the
  vectors unable to gain a case nobody passes yet, which is the only kind of
  case worth adding.) `tolerances.json`
  carries **corpus capture artifacts only**; its schema requires the
  implementation-gap field to be null.
- **One shared corpus.** The 500 goldens plus a capture-epoch `object_info`
  pinned to them, published as a versioned artifact so the public repo can
  consume it without a private checkout or a secret.
- **One immutable compatibility bundle.** All five inputs — `widget-layout.json`,
  `widget-type-census.json`, `classifier-vectors.json`, the goldens, and the
  capture-epoch `object_info` — are published **atomically under a single
  compatibility id**, and a consumer rejects a set whose ids do not all match.
  Per-artifact content hashes and the ancestry pin bind each file to itself;
  neither binds the *set*. Without one id over all five, a consumer can pair a
  newer layout or classification rule with older goldens and produce output that
  disagrees with the very corpus meant to adjudicate it — a disagreement that
  reads as a converter defect and is not one, which is the same confusion between
  drift and defect that the capture-epoch pin exists to remove. The id changes
  whenever any member changes, so a partial upgrade is a loud mismatch rather
  than a silent reinterpretation.
- **A shared comparison kernel as a specification with self-test vectors**,
  re-implemented per language (~150 LOC each) rather than shared as code or a
  subprocess. The signature grammar, the normalization rules, the verdict
  ladder, and a dozen `(ours, golden, expected-signatures)` triples every
  kernel must reproduce byte-exactly are normative.
- **No shared converter implementation.** Not the positional widget walk, not
  the promoted-subgraph resolver, not `graphToPrompt()` itself.
- **Each implementation keeps its own known-gap ratchet**, in its own repo,
  next to its own tests. Tolerance lists are never shared.
- **Every pin is ancestry-verified and automatically advanced.** A scheduled
  job proves the other implementation's main branch green and opens the bump
  PR; while it is red the pin cannot advance, so ignoring the signal visibly
  freezes the dependency instead of costing nothing.

## Alternatives considered

- **Extract `graphToPrompt()` into a shared headless package.** Rejected: its
  semantics live in `graph.serialize()`, `ExecutableNodeDTO.getInnerNodes()`,
  `node.resolveInput()`, `widget.serializeValue()` and every extension's
  `onNodeCreated`. A thin extraction would drag litegraph, every widget
  constructor and every extension — the whole frontend. The genuinely reusable
  part is the eleven-line widget loop, which all three headless
  implementations already have correctly.
- **Share the positional widget walk.** Rejected: three incompatible models
  exist (comfy-cli's strict/lenient companion heuristic plus its
  `widgets_values_form` representation; the package's `nullPeekIsCompanion`
  double-walk; the optional-padding semantics), each pinned by its own tests.
  A shared walk lands as a rewrite on two of three sides.
- **Share the promoted-subgraph resolver.** Rejected: comfy-cli resolves from
  the definition's declared link ids with budget, memo and cycle machinery
  keyed to object identity; the package walks `properties.proxyWidgets` with a
  depth guard because it must also serve the reverse direction. Converging
  means one side adopting the other's graph model, against a measured ~6-line
  fix that closes the actual gap.
- **Share the tolerance lists.** Rejected, and actively harmful. A tolerance
  encodes one implementation's blind spot. Had the package's list been shared,
  comfy-cli would have inherited a blindness to a defect it does not have.
- **Keep pairwise parity as the gate.** Rejected: comparing two
  implementations is N² comparisons with no truth side and is silent when both
  are wrong. Oracle-vs-N is N comparisons with one.
- **Pin to the other implementation's main branch and block on it.** Rejected:
  an unrelated commit in one repo would redden an unrelated PR in the other,
  and the same commit would pass or fail by wall clock. The ancestry-verified
  pin plus a scheduled advancing job keeps determinism *and* bounds staleness;
  neither half alone does.
- **Hand-author the widget tables and review them carefully.** Rejected on
  evidence: both existing hand-authored tables are measurably wrong today, in
  opposite directions, and neither was checked against the corpus by anything.
- **A mini-expression language for computed widget values.** Rejected for
  n≈3: three closed forms plus a named escape-hatch list.
- **Re-capture goldens in CI.** Rejected: the documented capture path needs a
  real authenticated browser against a production surface with a tab limit
  that fails silently partway through a long run. The capture-epoch catalog
  pin is the substitute, and it is better — it makes the corpus measure
  converter logic rather than N days of upstream churn.

## What we reject

Standing rules for compliance review, distinct from the alternatives above:

- **A fourth hand-maintained widget table.** Any new per-class or per-type widget
  knowledge goes in the shared data files this ADR establishes. A literal added to
  one implementation's allowlist is the defect class this ADR exists to stop: the
  two existing allowlists overlap in two entries out of twenty-four, and neither
  carried `COLORCODE`, which is a confirmed defect on 16 inputs across 15 classes.
  The single carve-out is the generic upload-flag rule used for a class with no
  layout entry (see Consequences). It is permitted because it is not per-type
  knowledge — it is the one rule the frontend itself applies, keyed on input
  flags rather than on any class or widget type — and because failing closed
  instead would make every unmanifested custom pack unconvertible. It is bounded
  by Validation 18 and is not a precedent for a hardcoded class or type list.
- **A tolerance entry as a substitute for a verdict.** A divergence may be
  tolerated only with the side named as correct and the reason recorded. An entry
  that says "UI-only" without naming which implementation matches the reference is
  how `PreviewAny` over-emitted three inputs on 148 of 148 golden nodes across 52
  fixtures without anyone noticing.
- **A reference pinned to a commit that is not an ancestor of its project's
  default branch.** The parity job's comfy-cli pin was 139 commits and 114 days
  off `main` and could not detect its own staleness.
- **A conformance tier whose assertion set is computed from the implementation
  under test.** `computePassingFixtures()` decides which fixtures to assert on by
  running the converter, so a regression removes a fixture from the set rather
  than failing it. Such a tier cannot fail and must not be counted as coverage.
- **A converter number without its measurement setting.** Four incompatible
  "clean fixture" counts for the same two implementations were in circulation
  (266, 394, 136, 205 of 500) because each was taken under a different
  combination of catalog epoch, tolerances, and `_meta` handling.

## Consequences

### Positive

- The class of bug that is currently invisible — a hand-maintained table
  disagreeing with the frontend — becomes unrepresentable, because the table
  is generated by the frontend.
- Converter defects and reference staleness are separated mechanically rather
  than by a hand-written predicate, so a reviewer can no longer relabel one as
  the other.
- A pin cannot silently rot: ancestry is asserted, and the advance is
  automated with a visible cost for ignoring it.
- Each implementation's known-gap ratchet is local, so no repo inherits
  another's blind spot.
- The frontend gains a `graphToPrompt()` regression guard it does not have.

### Negative

- Generated data is only as wide as the frontend's node manifest, so a class
  outside it has no layout entry. **A missing entry must not be read as "no
  injected widgets."** That default is already wrong for twelve classes, and
  because upload injection is keyed on the input flags `image_upload`,
  `video_upload` and `audio_upload` rather than on class identity, treating
  absence as emptiness would silently map a custom pack's `widgets_values` onto
  the wrong inputs — the exact failure mode this ADR exists to stop, reintroduced
  as a default. The layout file therefore distinguishes three states per class:
  an entry (authoritative), an explicit `"injects": []` (known empty, generated
  from a class the manifest covers), and **absent** (unknown).

  For an absent class, the consumer falls back to deriving injection from the
  `object_info` input flags, which is what comfy-cli does today, and emits a
  diagnostic naming the class. This is a deliberate carve-out from "no per-type
  widget knowledge outside the shared files" below: the flag rule is not
  per-type knowledge, it is the one generic rule the frontend itself applies,
  and the alternative — failing closed on any unmanifested class — would make
  every custom pack unconvertible. Validation 18 bounds the carve-out by failing
  if the absent set grows.
- More moving parts than a hardcoded pin: a dated bundle under one compatibility
  id, auto-bump PRs carrying its digest, and two publish channels on every
  release. The single id is also a coupling cost — any one artifact changing
  invalidates the whole set, so a census-only correction forces every consumer to
  re-pin. That is the intended trade: a loud re-pin is cheaper than a silent
  mismatch between a layout and the goldens meant to judge it.
- A `slot`-indexed layout assumes a class's widget set is static at creation.
  Two cases break that, and they are different in kind.

  **Dynamic input *types* — `COMFY_DYNAMICCOMBO_V3`, `COMFY_AUTOGROW_V3`.**
  Here the variation comes from the input type, not from the class, so these are
  explicitly **not** per-class escape hatches: enumerating them as such would put
  every class using them into the escape list and blow past the revisit threshold
  below on adoption day. The layout instead records, for the declaring input, its
  type and the sub-port expansion rule for that type — the selector's options and
  each option's ordered sub-ports for `DYNAMICCOMBO`, the per-instance growth unit
  for `AUTOGROW` — and the consumer expands slots from that rule plus the saved
  file. The rule is shared data keyed by input type, of which there are a handful;
  the classes using it are unbounded and unenumerated.

  **A class whose extension branches on a widget *value* at creation time.**
  This genuinely cannot be expressed as data and stays a named per-language
  escape hatch, which is what the "roughly five entries" threshold counts.
- The corpus cannot adjudicate two real shapes (`widgets_values_named`,
  populated legacy `proxyWidgets` host arrays). Those need hand-authored
  fixtures whose verdicts must never be quoted as oracle-backed.

## Validation

1. Every input type string in the `object_info` snapshot resolves to exactly
   one bucket; an unresolved type fails with its usage count and three example
   `Class.input` sites. The buckets are **not** `useWidgetStore().widgets`
   alone — that set covers only registered widget types, so every link type
   (`MODEL`, `LATENT`, any custom-pack type) would fail as unknown on every
   real catalog. `widget-type-census.json` therefore carries three generated
   buckets: `registeredWidget` (literally `useWidgetStore().widgets`),
   `dynamicInput` (the V3 types whose expansion rule the layout records), and
   `link` (the residual — any type the frontend constructs no widget for,
   which is what makes it a socket). A type in no bucket is the failure.
2. Each implementation's classifier agrees with the census verdict on a
   synthetic spec of every census type, and with every `classifier-vectors`
   case not listed in its own repo's expected-fail set (the per-repo ledger the
   Decision section defines; the shared file carries no such list).
3. Every member of a widget-type allowlist either has a declared input in the
   snapshot or is tagged injected with its injecting class list.
4. No converter output over the corpus contains an input named `upload`.
5. `PreviewAny` emits exactly `source` on all corpus instances.
6. Each fixture's divergence-signature set equals its manifest rows exactly —
   an extra signature and a resolved-but-still-listed signature both fail.
7. `catalogSensitive` is computed from the two-catalog run and must equal the
   manifest's stored value.
8. The tolerance file's implementation-gap field is null, and every entry
   carries the two evidence fields the tolerance rule above demands: the
   implementation named as **correct** for that divergence, and the recorded
   reason. The schema rejects a missing or empty either one. Requiring the
   verdict without checking for it is how `PreviewAny` was tolerated as "UI-only"
   on 148 of 148 golden nodes without anyone naming which side matched the
   reference.
9. Every tolerance entry carries a cited fixture and a sunset date, and the
   schema rejects one that does not. An entry producing no signature is a hard
   failure when its class is present in the corpus; when its class is **absent**
   from the corpus it is also a failure, because a tolerance for a class the
   corpus cannot exercise is unfalsifiable by construction and is exactly the
   dead tolerance this ADR rejects. A past sunset date fails regardless.
10. The pinned cross-repo SHA is an ancestor of that repo's main branch **and
    within a bounded distance of it** — 30 days and 50 commits, whichever binds
    first. Ancestry alone does not bound staleness: the 114-day pin this ADR
    cites as motivating evidence was an ancestor the whole time, which is why it
    went undetected. Exceeding either bound fails with the pin's age and commit
    distance. The bound is what makes the staleness claim above true, so a red
    upstream main does not license an indefinite freeze: it makes the bump a
    blocking item rather than an automatic one.
11. The suite executes at least one assertion per corpus fixture, with zero
    pending tests — a wholesale skip reddens.
12. A poison fixture whose golden is deliberately wrong in three ways produces
    exactly those three signatures; a kernel neutered to return empty sets
    fails because the expected signatures go missing.
13. A poison type claimed as a widget by the census and as a link by an
    implementation fails the agreement check.
14. Conversion never constructs an eager full catalog graph.
15. Each language's kernel reproduces every shared self-test vector byte-
    exactly.
16. A vendored data table is byte-identical to the published bundle, checked
    against a **content hash recorded in the consuming repo** rather than
    against a version label. A label is not an integrity check: a replaceable
    release asset, a re-published tag or a moved dist-tag all let a swapped
    bundle pass while the label still matches. The hash is what the auto-bump PR
    changes, so the diff shows the new digest and a bump that does not move the
    digest is a no-op.

17. A hand-authored fixture is excluded from every oracle-derived clean count
    and floor.
18. The absent-class set — classes with no layout entry, which fall back to
    flag-derived injection per the carve-out above — does not grow between
    releases. Growth means the frontend's node manifest is shedding coverage,
    which is the degradation the revisit trigger below watches for, and it is
    only visible if it is counted.
19. The absent-class fallback is asserted on a fixture, not merely bounded by
    its count: a class absent from the layout but declaring `image_upload`
    derives its injected widget from that flag, maps the remaining
    `widgets_values` to the correct inputs, and emits the diagnostic naming the
    class. A fixture whose class gains a layout entry later must keep asserting
    the fallback against a synthetic absent class, since the carve-out is the
    path every unmanifested custom pack takes and an untested fallback is how
    the twelve-class default went unnoticed.
20. Every consumed artifact set shares one compatibility id; a mixed set fails
    before any conversion runs, naming each artifact and its id.
21. Each implementation's expected-fail set for the shared classifier vectors is
    a **subset of its own previous set** — an entry may be removed, never added.
    This is the enforcement the local-ratchet rule depends on and is why the
    ledger is per repo: without a subset check, a newly added entry silences a
    regression at exactly the moment it appears, which is the failure mode a
    ratchet exists to prevent. Admitting a *new vector* that an implementation
    fails is the separate, allowed case — the vector is new, so there is no
    previous set to grow relative to, and the entry is recorded in the same
    change that adds the vector.

## Revisit triggers

- The frontend's widget contract stops being eleven lines — in particular, a
  widget whose `options.serialize` or `serializeValue` depends on runtime
  interaction rather than class identity, which would break the data-only
  premise and is the risk this ADR is least able to detect.
- The generated layout needs a fourth `serializeValue` form, or the escape-
  hatch list exceeds roughly five entries; at that point the rejected
  expression-language alternative should be reconsidered.
- Class coverage of the generated tables shrinks between releases, or the node
  manifest backing the export degrades.
- A fifth implementation of this contract appears, or the Go slot-write
  implementation adopts ComfyUI_frontend's ADR-SUBGRAPH-PROMOTION-0009 host-wins
  value model, at which point that contract may be foldable into this one.
- Re-capture becomes feasible on demand, which would make the capture-epoch
  catalog pin optional rather than mandatory.
- Any implementation's own ratchet is lowered rather than raised, or a
  tolerance is added without a cited fixture and a sunset date.

---

## Notes

Roadmap, per-wave acceptance checks and open questions: comfy-cli#1077.

Cross-repo obligations accepted with this ADR, tracked where the work lands:
ComfyUI_frontend#22420 (generate `widget-layout.json` and
`widget-type-census.json` from frontend CI) and comfy-cloud-mcp-server#1908
(converter consumes the shared bundle instead of its own tables).

Evidence this ADR rests on, all measured over the 500 template fixtures that ship
a frontend `graphToPrompt()` golden in `comfy-cloud-mcp-server`:
comfy-cloud-mcp-server#1869 (`PreviewAny`), comfy-cloud-mcp-server#1867 (the
stale parity pin), comfy-cli#1070 (promoted subgraph values, whose original
diagnosis this work retracted), comfy-cli#893 (widget-type classification),
comfy-cli#1079 (`set-widget` writing the wrong slot). The Comfy Hub conversion
audit that prompted this work is triaged in the issue tracker.
