---
name: comfy-fragments
description: Compose large Comfy workflows from small reusable fragment pieces — each a self-contained workflow JSON with declared inputs, outputs, and parameters. Use when iterating on complex workflows, when patterns repeat, or when a workflow JSON grows past ~200 lines.
---

This skill is the **composition layer** on top of `comfy`. Where
the core skill teaches you to build large single-graph workflows that
ComfyUI can parallelize, this skill teaches you how to **assemble those
large graphs from smaller reusable pieces** — like functions in code.

It assumes `comfy` (core CLI) is loaded. Pair it with the domain skills
(image, video, audio, editing domains in the core `comfy` skill). Fragments
are how you avoid rebuilding the same 8-node IPAdapter block five times.

---

## When to use fragments

Default to fragments + blueprints for workflows that may be extended. A small
workflow often becomes tomorrow's multi-shot, multi-seed, or multi-provider
pipeline; starting with a named fragment and a blueprint keeps that next step
cheap.

Use fragments when **any of**:

- A simple template or node chain might later need another shot, seed sweep,
  provider swap, refiner, ControlNet, LoRA, or final save variant.
- A sub-region of a workflow (a ControlNet stack, an IPAdapter block, a
  refiner pass, a save+thumbnail group) is reused across two or more
  workflows — extract it once, instantiate twice.
- A workflow JSON is creeping past ~200 lines and you're losing the
  mental model of which node ID does what.
- You're about to copy-paste a pattern from another workflow (text cards,
  inpaint passes, upscale finishers, LLM-driven prompt direction).
- You're iterating on a piece and small edits keep cascading through the
  whole JSON.
- Multiple model providers are chained (Ideogram + Reve + Flux + Magnific)
  and the chain wiring is the hard part.

Avoid fragments only when:

- A workflow is a truly throwaway one-shot and the user explicitly values speed
  over future extension.
- You only need to tweak values inside an existing workflow — use
  `comfy workflow slots / set-slot / vary` instead.
- The whole graph is under ~10-15 nodes, will not be varied or reused, and has
  no natural named sub-region.

---

## The mental model

A fragment is a **function with named inputs, outputs, and parameters**:

```
inpaint_region(
    image: IMAGE,                  # input
    mask:  MASK,                   # input
    prompt: str,                   # param
    guidance: float = 30.0,        # param with default
    seed: int = 2000,              # param with default
) -> IMAGE
```

Inside the fragment, the implementation can be 1 node or 30 — the caller
doesn't care. They pass arguments and get a typed result they can pipe
into the next step.

A **blueprint** chains fragments end-to-end (typically YAML), and a small
**composer** tool emits the final monolithic workflow JSON ready for
`comfy run`.

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
| `version` | no (default `"1"`) | String version, semver-ish. Bump when the interface changes. |
| `description` | recommended | One-line human description. |
| `terminal` | optional (default `false`) | `true` if the fragment contains its own `SaveImage`/`SaveVideo`. Stops the composer from appending another save. |
| `inputs` | no (default `{}`) | Each input has a `type` — any UPPER_SNAKE_CASE socket type (`IMAGE`, `MASK`, `AUDIO`, `VIDEO`, `STRING`, `MODEL`, `CONDITIONING`, `LATENT`, `VAE`, `CLIP`, custom types…) — and a `binds: "<interior_node_id>.<input_name>"` pointing at the actual node-field this input feeds. Path-loadable types (`IMAGE`/`MASK`/`AUDIO`/`VIDEO`) accept file paths — the composer injects a loader node. All other socket types must be fed by a cross-step ref (`$alias.output`), never a path. |
| `outputs` | no (default `{}`) | Each output has a `type` and `from: "<interior_node_id>"` plus optional `port` (default `0`). |
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

Fragment composition is built into the `comfy` CLI:

```bash
# Compose a blueprint into a single workflow JSON
comfy workflow compose blueprints/my_pipeline.yaml -o built/pipeline.json

# Specify a custom fragments directory (default: ./fragments)
comfy workflow compose blueprints/my_pipeline.yaml --lib ./my_fragments -o built/pipeline.json

# List fragments in a library
comfy --json workflow fragment ls [--lib DIR]

# Show a fragment's metadata, ports, and interior node count
comfy --json workflow fragment show <name_or_path>

# Validate a fragment file is well-formed
comfy --json workflow fragment validate <name_or_path>

# Then submit the composed workflow
comfy run --workflow built/pipeline.json --where cloud --wait
```

`--lib` defaults to `./fragments` relative to cwd. Default output is
`<blueprint>.compiled.json`.

With `chunk: N` in a `foreach` blueprint, compose splits items into
N-item batches and writes one numbered file per batch (`<stem>.000.json`,
`<stem>.001.json`, …). The envelope then reports `out: null` (there is no
single runnable file) plus `graphs` (count) and `written[]` (all paths) —
script against `data.written`, not `data.out`, and note any stale
unnumbered `<stem>.compiled.json` from a previous non-chunked compose is
deleted automatically.

All commands emit JSON envelopes under `comfy --json`. The composer
exits non-zero on validation errors with structured error codes
(`fragment_invalid`, `blueprint_invalid`, `blueprint_not_found`,
`fragment_lib_not_found`) — caught at compose time, not after cloud spend.
`fragment_lib_not_found` is raised by `workflow fragment ls` when an
explicit `--lib` path doesn't exist; a missing fragment during `compose`
surfaces as `fragment_invalid` instead.

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
comfy workflow compose blueprints/portrait.yaml -o workflows/portrait.json
comfy run --workflow workflows/portrait.json --where cloud --wait
```

That's the full agent loop. The fragment library is reusable across blueprints;
blueprints are small and obvious; the composed workflow is a normal API JSON
that submits like any other.

---

## 5. Real-world blueprint shape

A typical production pipeline for a single piece:

```yaml
pipeline:
  - fragment: subject_generator         # base photoreal scene
    alias: subject
    ...

  - fragment: text_card                  # branded text card 1
    alias: card_a
    inputs: {destination_image: $subject.image, source_mask: ...}
    ...

  - fragment: text_card                  # branded text card 2
    alias: card_b
    inputs: {destination_image: $card_a.image, source_mask: ...}
    ...

  - fragment: inpaint_region             # surgical fix to a problem area
    alias: fix_face
    inputs: {image: $card_b.image, mask: ...}
    ...

  - fragment: vision_verify              # in-graph QA gate (optional)
    alias: qa
    inputs: {image: $fix_face.image}

  - fragment: magnific_finish            # 4x upscale to print
    alias: final
    inputs: {image: $fix_face.image}
```

A 30-40 line blueprint expands to a 200-500 node workflow. Compose-time
validation catches the typical mistakes (missing inputs, bad alias
references, type mismatches) before you spend cloud compute on a
broken job.

---

## 6. How to create a fragment

**The typical flow** — discover the node, wrap it in a fragment, use it
from a blueprint:

1. Discover the node: `comfy --json nodes show <ClassName>` — check its
   inputs, outputs, and valid parameter values
2. Write `fragments/<name>.json` with:
   - `_fragment` header (name, inputs, outputs, params with binds)
   - Interior nodes (1-15) in standard API format
   - `"PLACEHOLDER"` for inputs that the blueprint will supply
   - Reasonable defaults for optional params
3. Validate: `comfy --json workflow fragment validate <name>`
4. Use from a blueprint and compose to verify it works end-to-end

**Refactoring path** — if you already have a working raw JSON workflow
and want to extract reusable pieces:

1. Identify the sub-region you'll reuse (5-15 nodes that form a logical unit)
2. Copy those nodes into `fragments/<name>.json`, add a `_fragment` header
3. Replace concrete values with `"PLACEHOLDER"`
4. Validate + compose + test

Always test a new fragment by composing a blueprint and submitting the
result before relying on it.

---

## 7. Picking input types

| Input type | Use for | The composer does |
|---|---|---|
| `IMAGE` | Photos, generated images, reference frames | Injects `LoadImage` when the blueprint value is a path; passes through when the value is `$alias.image` |
| `MASK` | Binary/alpha masks | Injects `LoadImage` + `ImageToMask` (channel: red) for paths |
| `AUDIO` | WAV/MP3/FLAC | Injects `LoadAudio` for paths |
| `VIDEO` | MP4/WebM | Injects `LoadVideo` for paths |
| `STRING` | Prompts, model names, captions, any literal | Pass-through. No loader injection. |

Use the type that matches what the interior node actually consumes.
`CONDITIONING` (and `MODEL`, `CLIP`, `VAE`, `LATENT`) are first-class input
types — declare `type: CONDITIONING` and wire it with a cross-step ref like
`conditioning: $encode.conditioning`. Only path-loadable types (`IMAGE`,
`MASK`, `AUDIO`, `VIDEO`) accept file paths; all other socket types must
come from a prior step via `$alias.output_name`.

---

## 8. Starter pattern library

Build these once and reuse forever.

### `subject_generator` — LLM-directed base generation

`ClaudeNode` (positive) + `ClaudeNode` (negative) + `Flux Dev` + LoRA
stack → IMAGE. Sweep on the Claude seed for genuine interpretation
variance, not just noise variance.

### `text_card` — typography card via Ideogram + composite

`IdeogramV3` → `ImageScale` → `ImageCompositeMasked`. Use this whenever
brand text or specific phrases must be legible — Ideogram is reliable
at text where Flux is not.

### `inpaint_region` — context-aware content replacement

`FluxProFillNode` with detailed prompt. Use for replacing a masked
region with new content that integrates with scene lighting. **Note:
this is REPLACE-ONLY — see gotchas section.**

### `magnific_finish` — production upscale

`MagnificImageUpscalerCreativeNode` (Sparkle engine). Defaults to
preserve mode (`creativity=0`, `resemblance=10`) for final delivery.
Bump `creativity` to 2-3 for mild detail enhancement.

### `vision_verify` — in-graph QA gate

`ClaudeNode` with `images` input + a checklist system_prompt. Returns
a structured PASS/FAIL critique. (Note: capturing the TEXT output of
ClaudeNode for retrieval requires a `SaveString` node — its in-graph
output is consumable by downstream nodes but not always exposed by the
job-status API.)

---

## 9. Gotchas baked into fragment defaults

Each of these tripped someone up during real production. Encoding them
into the fragment defaults means they can't be forgotten:

### `ImageCompositeMasked` — mask MUST match SOURCE size

The mask input gets bilinear-upscaled to the source image's dimensions.
If you pass a destination-sized mask (e.g., 1536×1024) with a small
source (e.g., 330×180), the small white-rectangle inside the big mask
shrinks to almost-black and the composite renders almost nothing.

The `text_card` fragment requires you to supply a mask **already sized
to your `scale_width × scale_height` params**. The composer documents
this expectation in error messages.

### `COMFY_DYNAMICCOMBO_V3` inputs use dotted keys

Nodes like `ClaudeNode`, `ReveImageCreateNode` declare a `model` input
with type `COMFY_DYNAMICCOMBO_V3` — when you select a model, that model
brings its own required sub-params (`max_tokens`, `temperature`, etc.).
In API workflow JSON these are flat dotted keys, NOT nested:

```json
// ✅ correct
"inputs": {
  "model": "Opus 4.6",
  "model.max_tokens": 800,
  "model.temperature": 0.95
}

// ❌ wrong — fails validation
"inputs": {
  "model": ["Opus 4.6", {"max_tokens": 800, "temperature": 0.95}]
}
```

### `SAM3Grounding` outputs MASK directly

Looks like a "find boxes" node but actually returns a `MASK`. Wire it
straight into mask-consuming nodes. `SAM3Segmentation` is only needed
when you've built boxes yourself via `SAM3CreateBox` +
`SAM3CombineBoxes`.

### `FluxProFillNode` is REPLACE-ONLY

It has no denoise / strength parameter. Whatever pixels are inside the
mask get fully regenerated from the prompt. **Never use it to "refine"
existing composited content** — it will overwrite that content.

For true refinement (preserve most of the masked region, only smooth
edges and lighting), use **KSampler + VAEEncode + SetLatentNoiseMask**
with `denoise=0.15–0.25`, or run `MagnificImageUpscalerCreativeNode`
with `creativity=2-3` at upscale time.

### `MagnificImageRelightNode` `style="smooth"` drains color

Counter-intuitively, the "smooth" relight style produces a sepia /
monochromatic image. For warming light without color loss, use
`style="brighter"` or `style="clean"`. Always test on a small image
before committing it to a pipeline.

### `MagnificImageUpscalerCreativeNode` parameter ranges

| Param | Range | Note |
|---|---|---|
| `creativity` | 0–10 | 0 = preserve, 4+ = noticeable reinterpretation |
| `resemblance` | -10–10 | NOT 0–100. 10 = max preservation. |
| `hdr` | 0–10 | small values are fine for most scenes |

### Text rendering: use Ideogram, not Flux

Flux and SDXL/SD3 cannot reliably render specific text. If your piece
has brand wordmarks, specific phrases, or proper names that MUST be
spelled correctly, the right tool is:

1. **Ideogram V3** in-graph (`IdeogramV3` node) — Ideogram is the
   text-master model in this stack
2. PIL composite externally (post-processing) — guaranteed but layered
3. **Don't** trust Flux Pro Fill to render text inside an inpaint
   — it will produce garbled glyphs every time

The `text_card` fragment uses Ideogram for this reason. Don't substitute
Flux into it.

---

## 10. When the pattern breaks down

Honest limits:

- **One-shot exploration is faster without the indirection** — if
  you're just trying things, write raw workflow JSON or use
  `comfy workflow vary`.
- **External (non-Comfy) steps don't compose as cleanly** — Python
  post-processing like PIL composites or external file conversions
  need a separate step type in the composer. Keep them outside the
  Comfy graph.
- **Comfy version drift** — if ComfyUI's native subgraph support
  (v0.3+) stabilizes for API workflow JSON export, eventually migrate
  to native subgraphs. The JSON-composition approach is portable but
  reinvents what Comfy itself wants to provide.
- **Debugging composed workflows** — when something fails, you're
  looking at a generated workflow JSON, not your hand-written one.
  Keep the composer's intermediate output (`blueprint.compiled.json`)
  for inspection. Log the blueprint + fragment versions per run.

---

## 11. What NOT to do

- **Don't put model loading inside every fragment.** Load `CheckpointLoaderSimple`
  once in the blueprint's first step and pass `model`/`clip`/`vae` outputs by
  cross-step ref. Fragments are about reusable sub-regions; the shared model
  state belongs at the top.
- **Don't author huge fragments.** If a fragment has more than ~15 interior
  nodes, it's probably two fragments. Same for params — 15+ params means
  you should split.
- **Don't hide critical model choices in defaults.** If swapping
  `flux1-dev` for `sd3.5_large` would silently change the output
  character, expose it as a required param.
- **Don't compose at runtime via shell scripts.** The Python composer
  catches errors at compose time. Shell glue catches them after cloud spend.
- **Don't reach into another fragment's internals.** If you need access
  to a node deep inside a fragment, that node should be promoted to
  an output of the fragment's public interface, or the fragment should
  be split.
- **Don't skip validation** before submitting a composed workflow that
  uses a new fragment. Run `comfy workflow fragment validate <name>`
  first — it catches missing `binds` targets, malformed metadata, and
  orphan interior nodes locally.
- **Don't reuse aliases across steps.** Aliases must be unique within a
  blueprint; the composer rejects duplicates.

---

## 12. Failure modes and what they mean

| code | what's wrong | what to fix |
|---|---|---|
| `fragment_invalid` | The fragment file itself is malformed (bad `_fragment` header, missing fields, dangling `binds`) | Read the error message; fix the fragment JSON |
| `fragment_lib_not_found` | An explicit `--lib` path passed to `fragment ls` doesn't exist | Pass a valid `--lib <real_path>` or omit to use the bundled library |
| `blueprint_not_found` | The blueprint YAML path doesn't exist | Check the path |
| `blueprint_invalid_yaml` | The blueprint file isn't valid YAML | Run it through `yamllint` |
| `blueprint_invalid` | The blueprint semantically fails (missing fragment, missing input, unknown input key, duplicate alias) | Read the error — it names the offending step alias |

---

## Summary

| Without fragments | With fragments |
|---|---|
| 1500-line workflow JSON | 40-line blueprint + N small fragments |
| Edits hunt through node IDs | Edits change one blueprint param |
| Errors caught at cloud submission | Errors caught at compose time |
| Patterns get copy-pasted between projects | Patterns become reusable units |
| Gotchas re-discovered each project | Gotchas baked into fragment defaults |

Fragments treat workflows the way good code treats logic: small named
units, typed interfaces, defaults that encode wisdom, and a composer
that wires them up.
