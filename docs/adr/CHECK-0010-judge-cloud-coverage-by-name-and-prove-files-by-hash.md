# ADR-CHECK-0010: Judge Cloud Coverage by Name and Prove Files by Hash

Date: 2026-11-05

## Status

Proposed

Relation: Builds on ADR-AUTOMATION-0003. `comfy workflow check` emits one
`envelope/1` envelope with its own `workflow_check` data schema.

## Context

A person or an agent holds a workflow file and wants to know whether Comfy
Cloud runs it as it is, or whether it needs a custom build first. Comfy Cloud
answers part of this already: its node list (`/api/object_info`) names every
node type it has and, in each loader's dropdown, every model file it ships.
Models the person uploaded are not in those dropdowns; they are in the
person's model list (`/api/assets`).

Three things make the question harder than a lookup:

- A model saved as `ink.safetensors` may sit on Cloud as
  `styles/ink.safetensors`. The workflow runs once the value is changed.
- Two files with the same name are not proven the same bytes. A LoRA trained
  twice keeps its name.
- Several Cloud files may share one name in different folders. Picking one
  would be a guess.

## Decision

`comfy workflow check <file> --where cloud` reads the workflow, Cloud's node
list and, for models the dropdowns do not list, the signed-in person's model
list. It answers with one of four verdicts, checked in this order:

1. `needs_build`: a node type or model is missing, several Cloud files share a
   model's name, or a local copy is proven a different file from Cloud's.
2. `covered_with_remaps`: everything is on Cloud, but a saved value must change
   to point at Cloud's copy in another folder.
3. `covered_needs_check`: nothing is missing, but a model is not proven the
   same file as Cloud's copy.
4. `covered`: every node type and model is on Cloud and each model is proven
   the same file.

A remap outranks an unproven model: the unchanged file does not run while a
remap is due.

Presence is judged by name. Identity is judged by bytes: a model is `same`
only when a local copy under `--models-dir` has the blake3 hash Cloud reports.
A size mismatch alone proves `different`. Without a local copy, or when
several local copies disagree, the model is `unverified`.

A remap happens only when exactly one Cloud file has the name. The check never
writes its input: `--apply-remaps --out <file>` writes a remapped copy, and an
`--out` that names the input is refused.

A failed read of the person's model list ends the command with an error. Without
it, a model the person uploaded would be reported missing.

## Alternatives considered

- Treat the same name as the same file. Simple, and right most of the time, but
  a retrained LoRA would run on Cloud with the old weights and nobody would
  know. The `covered_needs_check` verdict keeps that case visible.
- Hash every local model on each check. That reads every byte of every model
  on the machine; the check hashes only files of the workflow's model names
  under the folders the person passes with `--models-dir`.
- Pick the first match when several Cloud files share a name. The pick would be
  a guess; the check lists `candidates` instead and blocks.
- Rewrite the workflow in place. A person keeps the file they saved; a copy is
  cheap.

## What we reject

- A verdict of `covered` for a model whose bytes were never compared.
- Any write to the checked workflow file.
- Treating a failed Cloud read as "missing".

## Consequences

### Positive

- An agent can route on one field: run the workflow as it is, apply remaps
  first, or start a build.
- Each blocking model carries its file name, its ComfyUI models folder (from
  the workflow's own download notes or the stock loader) and any local copy,
  which is what an upload needs.

### Negative

- Without `--models-dir`, a workflow that uses any model is at best
  `covered_needs_check`.
- Folder names for custom loaders are unknown unless the workflow's download
  notes give them.

### Known limits

- A model value is recognized by its file extension; a loader that takes a name
  without one is not checked.
- The check compares node types, not node versions.

## Validation

`tests/comfy_cli/test_workflow_coverage.py` covers the offline rules: muted and
bypassed nodes, subgraphs, both dropdown forms, remaps, shared names, uploads,
identity and the verdict order. `tests/comfy_cli/command/test_workflow_check.py`
covers each verdict end to end against a mocked Cloud, that `--apply-remaps`
leaves the input unchanged, and each refusal. Each output validates against
`comfy_cli/schemas/workflow_check.json`.

## Revisit triggers

- Cloud's node list starts listing the person's uploads in loader dropdowns:
  the model-list read can go.
- Cloud publishes a per-file hash in its node list: identity could be checked
  without the extra model-list read per model.
