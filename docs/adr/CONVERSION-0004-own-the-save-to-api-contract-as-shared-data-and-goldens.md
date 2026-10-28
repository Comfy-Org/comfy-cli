# ADR-CONVERSION-0004: Own the save-to-API contract as shared data and goldens, not shared code

Date: 2026-10-28

## Status

Proposed

## Context

Converting a saved workflow to the API prompt the backend executes is
implemented four times: the frontend's `graphToPrompt()`
(`ComfyUI_frontend/src/utils/executionUtil.ts`), `@comfyorg/api-to-workflow`
(sourced from `comfy-cloud-mcp-server/src/converter/`), comfy-cli's
`convert_ui_to_api` (`comfy_cli/workflow_to_api.py`), and — for the adjacent
save-format slot-write contract — `Comfy-Org/cql`. The frontend is ground
truth because it is the only one with the live LiteGraph runtime.

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
  verbatim from `options.serialize !== false`, and an optional `serializeValue`
  in one of three closed forms) and `widget-type-census.json` (the registered
  widget-type set, which is literally `useWidgetStore().widgets`) are emitted
  by jobs in `ComfyUI_frontend` CI. A hand edit to either is rejected.
- **Jointly owned, evidence-bearing data.** `classifier-vectors.json` carries
  the ordered classification rule procedure as executable cases with a
  per-implementation `failsToday` ledger that may only shrink. `tolerances.json`
  carries **corpus capture artifacts only**; its schema requires the
  implementation-gap field to be null.
- **One shared corpus.** The 500 goldens plus a capture-epoch `object_info`
  pinned to them, published as a versioned artifact so the public repo can
  consume it without a private checkout or a secret.
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

- Generated data is only as wide as the frontend's node manifest; classes
  outside it get no entry, and a missing entry must mean "no injected
  widgets" — today's wrong-for-twelve-classes default. Honest coverage
  downgrade for unmanifested packs in exchange for correctness on manifested
  ones.
- More moving parts than a hardcoded pin: a dated bundle, auto-bump PRs, and
  two publish channels on every release.
- A `slot`-indexed layout assumes a class's widget set is static at creation.
  A class whose extension branches on a widget *value* at creation time cannot
  be expressed and stays a named per-language escape hatch.
- The corpus cannot adjudicate two real shapes (`widgets_values_named`,
  populated legacy `proxyWidgets` host arrays). Those need hand-authored
  fixtures whose verdicts must never be quoted as oracle-backed.

## Validation

1. Every input type string in the `object_info` snapshot appears in the census
   in exactly one bucket; an unknown type fails with its usage count and three
   example `Class.input` sites.
2. Each implementation's classifier agrees with the census verdict on a
   synthetic spec of every census type, and with every `classifier-vectors`
   case not listed in its own `failsToday`.
3. Every member of a widget-type allowlist either has a declared input in the
   snapshot or is tagged injected with its injecting class list.
4. No converter output over the corpus contains an input named `upload`.
5. `PreviewAny` emits exactly `source` on all corpus instances.
6. Each fixture's divergence-signature set equals its manifest rows exactly —
   an extra signature and a resolved-but-still-listed signature both fail.
7. `catalogSensitive` is computed from the two-catalog run and must equal the
   manifest's stored value.
8. The tolerance file's implementation-gap field is null.
9. Any tolerance entry producing no signature, on a class present in the
   corpus, is a hard failure.
10. The pinned cross-repo SHA is an ancestor of that repo's main branch.
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
16. A vendored data table is byte-identical to the published bundle at the
    pinned version.
17. A hand-authored fixture is excluded from every oracle-derived clean count
    and floor.

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
- A fifth implementation of this contract appears, or `cql` adopts
  ADR-SUBGRAPH-PROMOTION-0009's host-wins value model, at which point the
  slot-write contract may be foldable into this one.
- Re-capture becomes feasible on demand, which would make the capture-epoch
  catalog pin optional rather than mandatory.
- Any implementation's own ratchet is lowered rather than raised, or a
  tolerance is added without a cited fixture and a sunset date.
```

---

## Notes

Roadmap, per-wave acceptance checks and open questions: comfy-cli#1077.

Evidence this ADR rests on, all measured over the 500 template fixtures that ship
a frontend `graphToPrompt()` golden in `comfy-cloud-mcp-server`:
comfy-cloud-mcp-server#1869 (`PreviewAny`), comfy-cloud-mcp-server#1867 (the
stale parity pin), comfy-cli#1070 (promoted subgraph values, whose original
diagnosis this work retracted), comfy-cli#893 (widget-type classification),
comfy-cli#1079 (`set-widget` writing the wrong slot). Triage of the Comfy Hub
conversion audit that prompted it: BE-24127.
