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
   from the size of the diff.
2. Find the boundary. A partial change must say where it stops. Accept any of:
   a named follow-up issue or PR, a `## Revisit` or exception entry, a test that
   pins the unfixed case as currently-expected behaviour, or a comment at the
   code site naming what is still wrong.
3. Flag as medium when a change fixes part of a problem and names no boundary at
   all. The defect is the silence, not the partiality: the next reader cannot
   tell whether the remainder is known, forgotten, or deliberately refused.
4. Flag as medium when the change's own evidence contradicts its claim — a
   stated measurement the diff cannot produce, a cited fixture that does not
   exercise the path, or a test that passes identically before the change. Run
   or read the evidence before accepting it.
5. Flag as **low** when the remainder is named and tracked. A deliberately
   partial fix with a stated floor is a complete unit of work. Do not hold it
   for the rest of its epic.
6. Do not flag scope the author explicitly refused with a reason. "Deliberately
   not doing X, because Y" is a decision; ask for the reason to be recorded if
   it is missing, not for the scope to grow.
7. Where a change is one wave of a documented plan, check it against that plan's
   acceptance for *this* wave only. A wave that meets its own acceptance is
   complete even while later waves are empty.

## What this axis is not

It is not a test-coverage check — a change can be fully covered and still hide
what it leaves undone, and can be thinly covered while being perfectly honest.
It is not a scope-creep check either; that judgement runs in the opposite
direction and the two must not be resolved by widening the diff.
