---
name: partial-change-completeness
description: Judges whether a deliberately partial change is complete enough to land
severity-default: medium
tools: [Read, Grep, glob]
---

Judge a change that fixes **part** of a known problem. Most changes in this
repository are not whole fixes: the converter, the catalog and the widget walk
are each worked in slices, and a slice is the normal unit of landing. The
question is therefore not "is the problem solved" but "is this slice honest
about what it leaves".

Written because the completeness judgement was the only review axis with no
recorded rule, and it is the axis that most often declines a change here. An
author with no standard to meet cannot tell a partial fix that should land from
one that should not, and ends up either defending scope in prose or widening the
change until it is unreviewable.

## Review method

1. Identify what the change claims to fix, from its title, body and tests — not
   from the size of the diff. Title and body come with the review context, not
   from a fetch this check performs; a boundary stated only in a linked tracker
   item is verified by reading that item's own text if it is in the diff or
   context, and left unflagged rather than guessed at otherwise.
2. Find the boundary. A partial change must say where it stops. Accept any of:
   a named follow-up issue or PR, a `## Revisit triggers` section (the ADR
   convention, `docs/adr/`) or exception entry (`program/exceptions.md`), a
   test that pins the unfixed case as currently-expected behaviour, a comment
   at the code site naming what is still wrong, or a documented plan that
   enumerates the later waves (see step 7). Each of these must be
   a resolvable artifact — a real issue/PR number, an existing file section, or
   a test that runs — not prose alone; a sentence like "deliberately not doing
   X" without one of these does not satisfy this step.
3. Flag as medium when a change fixes part of a problem and names no boundary at
   all, **or names one in prose without a resolvable artifact per step 2** —
   a sentence that points at a remainder with none of step 2's artifacts behind
   it (issue, PR, section, test, **or a comment at the code site**) does not
   satisfy step 2, so treat it the same as silence.
   The defect is the silence, not the partiality: the next reader cannot tell
   whether the remainder is known, forgotten, or deliberately refused.
4. Flag as medium when the change cites evidence *for its fix* that the diff
   contradicts — a stated measurement the diff cannot produce, a cited fixture
   that does not exercise the path, or a test claimed to demonstrate the new
   behaviour that in fact passes identically before the change. This does not
   apply to a boundary-pinning test from step 2: that test is supposed to pass
   unchanged, and passing before and after is how it does its job, not a
   contradiction. Read the evidence before accepting it; this check's tools
   cannot execute anything, so treat a claim this check cannot read as unverified
   rather than run it yourself.
5. Where the remainder is named and tracked per step 2, do not flag the
   boundary itself as a step-3 finding. A deliberately partial fix with a
   stated floor is a complete unit of work; do not hold it for the rest of
   its epic. This exempts only the remainder — step 4 still applies to
   whatever evidence the change cites for the fix it does make.
6. Do not flag scope the author explicitly refused with a reason **and** a
   resolvable artifact per step 2. Refused scope with a reason but no artifact
   is the step-3 case (medium): ask for the artifact, not for the scope to grow.
   Refused scope with an artifact but no stated reason is also the step-3 case:
   a tracked follow-up says the remainder is known, not that it was declined on
   purpose, so ask for the reason rather than inferring intent from the artifact
   alone.
7. Where a change is one wave of a documented plan, check it against that plan's
   acceptance for *this* wave only. A wave that meets its own acceptance is
   complete even while later waves are empty.

   **The plan is itself a step 2 artifact** when it is a resolvable document
   that names the later waves — a file in the repository, or a tracker item with
   a number. That resolves the tension between this step and step 3: a wave need
   not also open an issue for work its own plan already enumerates, because the
   plan is where the remainder is tracked. A change that merely *claims* to be
   one wave of a plan, with no such document to read, falls back to step 3.

## What this axis is not

It is not a test-coverage check — a change can be fully covered and still hide
what it leaves undone, and can be thinly covered while being perfectly honest.
It is not a scope-creep check either; that judgement runs in the opposite
direction and the two must not be resolved by widening the diff.
