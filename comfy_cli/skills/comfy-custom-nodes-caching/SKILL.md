---
name: comfy-custom-nodes-caching
description: Packaged fallback for a custom node's cache signature and why it silently recomputes.
---

# Execution caching fallback

A node declares its cache signature with `fingerprint_inputs` (V3) or `IS_CHANGED` (V1). Declaring neither lets the node's declared inputs decide, which is the right default; add one only when the output depends on something those inputs do not capture.

The signature is taken **before the graph runs**, so a linked input arrives as `None` — only widget values, literals and hidden inputs are real. The exception is an input declaring `rawLink`, handed its `[node_id, output_index]` reference instead. Reaching into a tensor or model through either form raises, and the executor catches that, logs a warning, and makes the signature a not-a-number value that never compares equal to itself. A broken signature therefore shows up as a node that recomputes every run — never as a crash. Keep values hashable: strings, numbers, booleans, containers of those.

The signature is **not local**: a node's cache key embeds every ancestor's signature, so one value that never repeats forces its whole downstream cone to recompute. A wall-clock reading is the usual cause — correct for a live fetch that must always re-read, a cache defeat for everything after it. Derive the value from what the output depends on: resolved widget values, plus external state the node reads, such as a file's size and modification time, environment variables, and a version constant bumped when behaviour changes.

Prove both halves: call the signature twice with identical arguments and require equality, then change one field and require a difference. Failing the first is a cache defeat; failing the second is a stale-output bug. Marking a node non-idempotent or as producing intermediate output changes cache behaviour separately and does not substitute for a correct signature.
