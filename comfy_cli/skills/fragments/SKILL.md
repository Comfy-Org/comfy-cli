---
name: comfy-fragments
description: Typed reusable workflow fragments + YAML blueprint composition — build large pipelines from small tested pieces. Solves the "I keep hand-merging JSONs" problem.
---

This skill is the composition layer. Pair it with `comfy-pipeline` (which
handles cross-workflow orchestration) and the domain skills (`comfy-image`,
`comfy-video`, `comfy-audio`, `comfy-edit`). Fragments are how you avoid
rebuilding the same 8-node IPAdapter block five times.

---

## When to use fragments

Use fragments when:

- A sub-region of a workflow (a ControlNet stack, an IPAdapter block, a
  refiner pass, a save+thumbnail group) is reused across two or more
  workflows — extract it once, instantiate twice.
- A workflow has crossed ~30 nodes and is getting hard to reason about. Group
  related nodes into named, typed fragments.
- An agent is going to repeatedly build similar workflows with the same
  building blocks (the common case for image/video pipelines).

Do **not** use fragments when:

- A workflow is a one-shot. Hand-built JSON is fine.
- You only need to tweak values inside an existing workflow — use
  `comfy workflow slots / set-slot / vary` instead.

---

## 1. The fragment file format

A fragment is one `.json` file in a `fragments/` directory. It has a
`_fragment` metadata header that declares the fragment's typed interface,
followed by the interior ComfyUI nodes (API-format, just like a workflow).

```json
{
  "_fragment": {
    "name":        "image_blend",
    "version":     "1",
    "description": "Blend two images with a configurable mode and factor.",
    "terminal":    false,
    "inputs": {
      "image1": {"type": "IMAGE", "binds": "10.image1"},
      "image2": {"type": "IMAGE", "binds": "10.image2"}
    },
    "outputs": {
      "image":  {"type": "IMAGE", "from": "10", "port": 0}
    },
    "params": {
      "blend_factor": {"type": "FLOAT", "binds": "10.blend_factor", "default": 0.5},
      "blend_mode":   {"type": "COMBO", "binds": "10.blend_mode",   "default": "normal"}
    }
  },

  "10": {
    "class_type": "ImageBlend",
    "_meta": {"title": "blend two passes"},
    "inputs": {
      "image1": "PLACEHOLDER",
      "image2": "PLACEHOLDER",
      "blend_factor": 0.5,
      "blend_mode": "normal"
    }
  }
}
```

### Metadata fields

| field | required | meaning |
|---|---|---|
| `name` | yes | Stable identifier. Blueprints reference fragments by this name. |
| `version` | yes | String version, semver-ish. Bump when the interface changes. |
| `description` | recommended | One-line human description shown in `fragment ls`. |
| `terminal` | optional (default `false`) | `true` if the fragment contains its own `SaveImage`/`SaveVideo`. Stops the composer from appending another save. |
| `inputs` | yes | Each input has a `type` (any ComfyUI socket type — `IMAGE`/`MASK`/`AUDIO`/`VIDEO`/`STRING`, or a graph type like `MODEL`/`CONDITIONING`/`LATENT`/`VAE`/custom) and a `binds: "<interior_node_id>.<input_name>"` pointing at the actual node-field this input feeds. |
| `outputs` | yes | Each output has a `type` and `from: "<interior_node_id>"` plus optional `port` (default `0`). |
| `params` | optional | Settable values (text, seed, strength, model name, etc.). Each has `type` ∈ {`STRING`, `INT`, `FLOAT`, `BOOL`, `COMBO`}, a `binds`, and optionally a `default`. |

### Conventions for interior nodes

- Use simple integer IDs (`"10"`, `"11"`, …). The composer remaps them
  globally so collisions across fragments don't matter.
- Use the literal string `"PLACEHOLDER"` for any input that will be filled
  by the composer at instantiation. (Defaults from `params` overwrite it.)
- Internal edges (`["10", 0]`) are preserved and renumbered automatically.

---

## 2. The blueprint DSL

A blueprint is a YAML file describing one composed workflow. The composer reads
the blueprint, instantiates each listed fragment, wires inputs/params, and
writes one API-format workflow JSON.

```yaml
output_prefix: outputs/my_pipeline

pipeline:
  - fragment: text_card           # name (looked up in ./fragments/)
    alias:    headline            # unique handle for downstream refs
    inputs:
      destination_image: inputs/base.png        # raw path → LoadImage injected
      source_mask:       inputs/mask_top.png    # type MASK → LoadImage + ImageToMask
    params:
      text_prompt: "BREAKING NEWS"
      comp_x: 140
      comp_y: 30

  - fragment: text_card
    alias:    subhead
    inputs:
      destination_image: $headline.image        # ← previous step's output
      source_mask:       inputs/mask_sub.png
    params:
      text_prompt: "...details..."
```

### Input binding values

The composer accepts three things on the right-hand side of an `inputs:` entry:

- **`$alias.output_name`** — reference a prior step's named output. Resolved
  to `[node_id, port]` automatically.
- **A path string** — for `IMAGE`, `MASK`, `AUDIO`, `VIDEO` inputs the composer
  injects the appropriate loader (`LoadImage` / `LoadAudio` / `LoadVideo`,
  plus `ImageToMask` for `MASK`). For `STRING` inputs the value passes through
  as a literal.
- **A literal** — for `STRING` inputs only. Non-string literals for non-STRING
  types are rejected.

Graph-only socket types (`MODEL`, `CONDITIONING`, `LATENT`, `VAE`, and custom
node sockets) have no loader to inject from a path, so they **must** be fed by
a cross-step ref — passing a path to one is an error.

### Cross-step refs work across any output type

`$alias.image`, `$alias.conditioning`, `$alias.mask`, `$alias.audio`,
`$alias.video` — whatever the fragment declared as outputs. The composer
errors clearly if the alias or output name doesn't exist.

### Final save behavior

If the **last** step's fragment has `terminal: true`, the composer leaves the
workflow alone (your fragment handles saving). Otherwise it appends a
`SaveImage` or `SaveVideo` (auto-detected from the final step's first
`IMAGE`/`VIDEO` output) using `output_prefix` as the filename prefix.

---

## 3. The command surface

```bash
comfy workflow compose <blueprint.yaml> [-o out.json] [--lib <fragments_dir>]
comfy workflow fragment ls            [--lib <dir>]
comfy workflow fragment show <name|path> [--lib <dir>]
comfy workflow fragment validate <name|path> [--lib <dir>]
```

`--lib` defaults to `./fragments` relative to cwd. `compose`'s default output
is `<blueprint>.compiled.json`.

All commands emit JSON envelopes under `comfy --json`. `compose` and the
`fragment` commands all exit non-zero on validation errors with structured
error codes (`fragment_invalid`, `blueprint_invalid`, `blueprint_not_found`,
`fragment_lib_not_found`). On success, `compose` emits the blueprint path under
the `blueprint` key alongside `out`, `steps`, `nodes`, and `fragments_used`.

---

## 4. End-to-end example

Project layout:

```
my-project/
  fragments/
    text_encode.json
    sampler.json
    save_still.json
  blueprints/
    portrait.yaml
  inputs/
    seed_photo.png
```

Compose + submit:

```bash
cd my-project
comfy workflow compose blueprints/portrait.yaml -o built/portrait.json
comfy run --workflow built/portrait.json --where cloud --wait
```

That's the full agent loop. The fragment library is reusable across blueprints;
blueprints are small and obvious; the composed workflow is a normal API JSON
that submits like any other.

---

## 5. When to extract a fragment (the "graduate after it works" rule)

Don't author fragments speculatively. Build the full workflow as one JSON
first, get it running end-to-end, then carve reusable pieces out:

1. Build a single 30-node workflow that does what you want
2. Submit + verify it works end-to-end
3. Identify the sub-region you'll reuse (the 5-8 nodes that form a logical
   unit — IPAdapter block, ControlNet stack, refiner pass)
4. Copy those nodes into `fragments/<name>.json`, add a `_fragment` header
   declaring its inputs/outputs/params
5. Run `comfy workflow fragment validate <name>` to confirm it parses
6. Use it from a blueprint; verify the composed workflow matches the original

Always test the fragment by composing a blueprint and submitting the result
before relying on it. Fragments are reusable, which means a bug in one
poisons every blueprint that uses it.

---

## 6. Picking input types

| Input type | Use for | The composer does |
|---|---|---|
| `IMAGE` | Photos, generated images, reference frames | Injects `LoadImage` when the blueprint value is a path; passes through when the value is `$alias.image` |
| `MASK` | Binary/alpha masks | Injects `LoadImage` + `ImageToMask` (channel: red) for paths |
| `AUDIO` | WAV/MP3/FLAC | Injects `LoadAudio` for paths |
| `VIDEO` | MP4/WebM | Injects `LoadVideo` for paths |
| `STRING` | Prompts, model names, captions, any literal | Pass-through. No loader injection. |
| `MODEL` / `CONDITIONING` / `LATENT` / `VAE` / custom | Graph-internal sockets passed between fragments | Nothing — these are ref-only. Wire them with `$alias.output`; a path is rejected. |

Use the type that matches what the interior node actually consumes, and declare
graph-internal sockets (`MODEL`, `CONDITIONING`, `LATENT`, `VAE`, or any custom
node type) by their real type — this is how you build a complex pipeline:
e.g. a loader fragment exposes `outputs: {model: {type: MODEL, ...}}` and a
sampler fragment declares `inputs: {model: {type: MODEL, ...}}`, wired in the
blueprint as `model: $base.model`. Those types have no path loader, so they can
only be fed by a cross-step ref — the composer wires whatever `[node, port]`
the upstream fragment exposes.

---

## 7. What NOT to do

- **Don't put model loading inside every fragment.** Load `CheckpointLoaderSimple`
  once in the blueprint's first step and pass `model`/`clip`/`vae` outputs by
  cross-step ref. Fragments are about reusable sub-regions; the shared model
  state belongs at the top.
- **Don't author huge fragments.** If a fragment has more than ~15 interior
  nodes, it's probably two fragments.
- **Don't skip `comfy workflow fragment validate`** before submitting a
  composed workflow that uses a new fragment. Validation catches missing
  `binds` targets, malformed metadata, and orphan interior nodes locally —
  none of which the cloud server will tell you about clearly.
- **Don't reuse aliases across steps.** Aliases must be unique within a
  blueprint; the composer rejects duplicates.

---

## 8. Failure modes and what they mean

| code | what's wrong | what to fix |
|---|---|---|
| `fragment_invalid` | The fragment file itself is malformed (bad `_fragment` header, missing fields, dangling `binds`) | Read the `error` + `hint` fields; fix the fragment JSON |
| `fragment_lib_not_found` | `--lib` (or default `./fragments`) doesn't exist | Create the directory or pass `--lib <real_path>` |
| `blueprint_not_found` | The blueprint YAML path doesn't exist | Check the path |
| `blueprint_invalid_yaml` | The blueprint file isn't valid YAML | Run it through `yamllint` |
| `blueprint_invalid` | The blueprint semantically fails (missing fragment, missing input, unknown input key, duplicate alias) | Read the `error` field — it names the offending step alias |

All errors have a `details` field with structured context (offending path,
step alias, what was expected).
