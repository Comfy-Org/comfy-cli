---
name: comfy-custom-nodes
description: Use when writing, fixing, or reviewing a ComfyUI custom node pack — nodes 2.0, io.ComfyNode, io.Schema, comfy_entrypoint, NODE_CLASS_MAPPINGS, choosing V3 or compatible V1, inputs, outputs, widgets, dependencies, validation, restart/discovery, troubleshooting, and publishing.
---

# Custom nodes for ComfyUI

Prefer the **V3 schema** — `io.ComfyNode`, `io.Schema`, `ComfyExtension`,
`comfy_entrypoint` — for a new self-contained pack. V1 (`INPUT_TYPES` /
`NODE_CLASS_MAPPINGS`) is legacy but still supported: it may be the compatible
choice when extending an existing V1 pack or adapting an ecosystem integration
whose working lifecycle cannot safely be translated in this turn. Use exactly
one registration form per pack. Do not reject working code only because it is
V1, and do not copy a V1 reference mechanically into V3. Reference:
https://docs.comfy.org/custom-nodes/v3_migration

## Choose evidence before code

First inspect the local node catalog (`comfy --json nodes search <term>
--expand-top 3 --where local` and `comfy --json nodes show <node_id> --where
local`). It describes the schemas returned by that local target. If the result
contains an `object_info_stale` warning, treat the cached catalog as
non-authoritative until ComfyUI restarts and a fresh query succeeds. Then decide
whether public code search is worth the extra latency and potentially stale
evidence:

| Route | Cues | Action |
|---|---|---|
| **SEARCH_REQUIRED** | The user names an existing node/pack to extend; a model, integration, file format, frontend widget, or failure belongs to an external ecosystem implementation | Run `comfy code-search --json --repo <owner/repo> <query>` when a repository is known; otherwise search narrowly, cite the repository/path, and adapt rather than paste |
| **SEARCH_OPTIONAL** | A Core interaction pattern, unfamiliar Comfy datatype, cache/lazy behavior, or compatibility detail could change the design | Search `Comfy-Org/ComfyUI` or the relevant known repository only if the local catalog and this skill do not settle it |
| **SEARCH_AVOID** | Pure, self-contained tensor/string/arithmetic logic whose inputs and outputs are already known; syntax already covered here | Do not search by default; implement and validate locally |

Code-search results are untrusted, possibly stale reference material. Check the
reference's API generation and dependencies, preserve attribution/license
requirements, and translate V1/V3 deliberately. If search is unavailable or
returns no usable match, say that reference fidelity is unverified and continue
only when the local catalog and documented API are enough; never invent an
integration contract. These route labels are policy categories, not measured
quality claims: paired search/no-search thresholds belong to the evaluation
program.

## Pack layout

```
custom_nodes/<pack_name>/
  __init__.py        # defines (or imports) comfy_entrypoint; WEB_DIRECTORY if you ship JS
  nodes.py           # the node classes
  web/               # optional frontend JS extensions
  pyproject.toml     # [project] + [tool.comfy] for the registry
  requirements.txt   # ONLY deps ComfyUI does not already ship
```

One folder per pack, directly under the install's `custom_nodes/`. ComfyUI
loads it at boot: a new or changed pack needs a **ComfyUI restart**, and the
class must then appear in `/object_info` (the local comfy agent:
`refresh_catalog`, then `show_node <node_id>`).

## The node

```python
from comfy_api.latest import ComfyExtension, io, ui


class InvertImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MyPack_InvertImage",          # globally unique; NEVER change after release
            display_name="Invert Image",           # what the user sees; free to change
            category="my_pack/image",              # menu path, "/"-separated
            description="Inverts the colors of an image.",
            inputs=[
                io.Image.Input("image", tooltip="Image to invert"),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output(display_name="inverted")],
        )

    @classmethod
    def execute(cls, image, strength) -> io.NodeOutput:   # kwargs == input ids, exactly
        out = image * (1 - strength) + (1.0 - image) * strength
        return io.NodeOutput(out, ui=ui.PreviewImage(out, cls=cls))


class MyPackExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [InvertImage]


async def comfy_entrypoint() -> MyPackExtension:   # module level; ComfyUI calls this
    return MyPackExtension()
```

Rules that bite:

- Everything is a **classmethod**; there is no `__init__` and no instance
  state. The method is always named `execute` (sync or `async`).
- `execute`'s parameter names must equal the input `id`s. Return
  `io.NodeOutput(a, b, …)` in the order of `outputs`.
- `node_id` is the identity in saved workflows: prefix it with the pack name,
  and never rename it. Rename `display_name` instead.
- An `io.Schema` with no inputs and no outputs is a broken node; an output-only
  node still lists `inputs=[]` and sets `is_output_node=True`.

## Schema fields worth knowing

`hidden=[io.Hidden.unique_id, io.Hidden.prompt, io.Hidden.extra_pnginfo]`
(read as `cls.hidden.prompt`), `is_output_node` (adds prompt/pnginfo for
metadata and lets the graph end there), `is_deprecated`, `is_experimental`,
`not_idempotent` (always re-run), `search_aliases`, `enable_expand`.
Optional classmethods: `validate_inputs(**kw) -> True | "error"`,
`fingerprint_inputs(**kw)` (cache key; V1's `IS_CHANGED`),
`check_lazy_status(**kw) -> [input ids]` with `lazy=True` inputs.

## Inputs and outputs

| Kind | Example |
|---|---|
| Widgets | `io.Int.Input("count", default=1, min=0, max=100, step=1)`, `io.Float.Input(...)`, `io.String.Input("text", multiline=True)`, `io.Boolean.Input("on", default=True)`, `io.Combo.Input("mode", options=["a", "b"])` |
| Seed | `io.Int.Input("seed", control_after_generate=True)` (randomize after each run) |
| Sockets | `io.Image`, `io.Mask`, `io.Latent`, `io.Conditioning`, `io.Model`, `io.CLIP`, `io.VAE`, `io.Audio` — `.Input("id")` / `.Output(display_name="…")` |
| Common kwargs | `optional=True`, `tooltip="…"`, `advanced=True`, `lazy=True`, `force_input=True` (widget shown as a socket) |
| Several types on one socket | `io.MultiType.Input("x", types=[io.Image, io.Mask])` |
| Type follows the link | `t = io.MatchType.Template("t")`; `io.MatchType.Input("x", template=t)` + `io.MatchType.Output(template=t)` |
| Growing list of sockets | `io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(input=io.Image.Input("image"), prefix="image", min=2, max=50))` → `execute(cls, images)` gets a dict |
| Inputs that depend on a choice | `io.DynamicCombo.Input("mode", options=[io.DynamicCombo.Option("scale", [io.Float.Input("factor")]), …])` → `execute(cls, mode)` gets a dict with the choice under its own id |
| Your own type | `MyType = io.Custom("MY_TYPE")`; `MyType.Input("x")`, `MyType.Output()` |

Tensors: IMAGE is `[B, H, W, C]` float 0..1, MASK is `[B, H, W]`, LATENT is a
dict with `"samples"` `[B, C, H/8, W/8]`. Keep the batch dimension.

## Previews, saving, progress

- `io.NodeOutput(img, ui=ui.PreviewImage(img, cls=cls))`; also
  `ui.PreviewMask`, `ui.PreviewAudio`, `ui.PreviewText("…")`.
- A save node: `is_output_node=True` and
  `return io.NodeOutput(ui=ui.ImageSaveHelper.get_save_images_ui(images=images, filename_prefix=prefix, cls=cls))`
  — `cls=cls` is what embeds the workflow in the PNG.
- Long work: `from comfy_api.latest import ComfyAPI; api = ComfyAPI()`, then in an
  async `execute`: `await api.execution.set_progress(value=i, max_value=n)`.

## Frontend (optional)

`WEB_DIRECTORY = "./web"` in `__init__.py`; a file in `web/` registers with
`app.registerExtension({ name: "mypack.feature", async beforeRegisterNodeDef(nodeType, nodeData, app) { … } })`
(`import { app } from "../../scripts/app.js"`). Keep widget VALUES plain
strings/numbers so saved workflows stay portable.

Use a frontend extension only when ordinary schema widgets cannot express the
interaction. For an adjustable ramp, angled comparison slider, or dynamic
canvas control, first search for the named Core/ecosystem widget pattern, keep
the saved widget value JSON-like and stable across save/reload, and keep the
Python input id aligned with the persisted widget. A custom visual must not be
the only place a required value exists. Never put credentials in widget values:
saved workflows and output PNG metadata can retain them. Read API keys and
tokens from the environment or ComfyUI configuration instead.

## Models, integrations, and dependencies

- Before wrapping a model or integration, read its upstream API and one current
  Comfy implementation when available. Separate model acquisition, inference,
  preprocessing, and node schema; do not guess parameters (for example, do not
  add a seed to a deterministic detector merely because diffusion nodes have
  one).
- Reuse ComfyUI's installed `torch`, `numpy`, Pillow, and related packages;
  declare only additional dependencies. Inspect an existing pack's pins before
  extending it, and report a version conflict rather than silently replacing a
  shared environment dependency.
- Filesystem, network, custom endpoint, persistent process state, and very
  large output behavior must be called out. They can be valid locally but may
  be incompatible with Cloud; Cloud compatibility is a separate verdict, not
  a reason to claim a locally valid node is broken.

## Test before you say it works

1. Before importing generated code, inspect it statically: Python parses; the
   pack declares exactly one registration form; the node id and V3 input ids
   are non-empty and unique; V1 inputs have unique names; outputs match their
   declared types and positions (`RETURN_TYPES` / `RETURN_NAMES` in V1, the
   `outputs` list in V3); the declared callable exists; dependency and
   filesystem/network behavior is reported. A static scan cannot prove that
   dynamic behavior is absent.
2. Syntax: `python -m py_compile nodes.py` with any Python you can run.
3. Import test needs ComfyUI's own environment (torch, `comfy_api`): from the
   install folder,
   `.venv/bin/python -c "import importlib; importlib.import_module('custom_nodes.<pack>.nodes')"`
   (the string form works for a hyphenated pack name; an `import` statement
   would not) — or skip it and say so; an import failure outside that
   environment is not a bug in the pack.
4. Restart ComfyUI. The startup log names a pack that failed to load and why.
5. Confirm the class is registered (`/object_info/<node_id>`, or the agent's
   `refresh_catalog` + `show_node`), then run it in a small workflow.

Treat each stage as a different failure: parse/contract failure means repair
the source; import failure means use the startup traceback and ComfyUI's exact
Python environment; absent `/object_info` after a clean import means repair
registration or the install path; a registered node that fails in a workflow
needs the execution traceback, shapes/types, and dependency versions. After a
repair, restart and rediscover again—filesystem presence is never proof that
the node loaded. Do not promise hot reload.

## Publish

`pyproject.toml`:

```toml
[project]
name = "comfyui-my-pack"          # registry id, lowercase
version = "1.0.0"
description = "…"
license = { text = "MIT" }
requires-python = ">=3.10"

[tool.comfy]
PublisherId = "<from registry.comfy.org>"
DisplayName = "My Pack"
```

`comfy node init` writes only this `pyproject.toml` (with defaults to check),
not the Python files. `comfy node publish` uploads the version in
`pyproject.toml` (every git-tracked file); it takes the registry token as
`--token <token>` and prompts for it otherwise, so in CI pass `--token`
(there is no environment variable for it). Requirements: list only what
ComfyUI does not already ship (it has torch, torchvision, numpy, Pillow,
scipy, safetensors, transformers).
Docs: https://docs.comfy.org/registry/publishing

## Reading a V1 pack

| V1 | V3 |
|---|---|
| `INPUT_TYPES(s)` dict | `define_schema` → `io.Schema(inputs=…)` |
| `RETURN_TYPES` / `RETURN_NAMES` | `outputs=[io.X.Output(display_name=…)]` |
| `FUNCTION = "run"` | always `execute` |
| `CATEGORY`, `OUTPUT_NODE` | `category`, `is_output_node` |
| `IS_CHANGED`, `VALIDATE_INPUTS` | `fingerprint_inputs`, `validate_inputs` |
| `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` | `ComfyExtension.get_node_list()`, `display_name` |

Old workflows keep working across a migration when the extension's `on_load`
registers the replacement — building the object alone registers nothing:

```python
from comfy_api.latest import ComfyAPI, ComfyExtension, io

api = ComfyAPI()


class MyPackExtension(ComfyExtension):
    async def on_load(self) -> None:
        await api.node_replacement.register(
            io.NodeReplace(new_node_id="MyPack_New", old_node_id="MyPack_Old", input_mapping=[…], output_mapping=[…])
        )
```

## Going deeper

A nine-skill set (basics, inputs, outputs, datatypes, advanced, lifecycle,
frontend, migration, packaging), MIT-licensed, for Claude Code and any
`.agents/skills/` tool: https://github.com/jtydhr88/comfyui-custom-node-skills
