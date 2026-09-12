---
name: comfy-custom-nodes
description: Use when writing, fixing, or reviewing a ComfyUI custom node pack — the V3 node API, also called nodes 2.0 (io.ComfyNode, io.Schema, comfy_entrypoint), inputs, outputs, previews, the pack layout, testing without a restart loop, and publishing to the registry. V1 (NODE_CLASS_MAPPINGS) only for reading old packs.
---

# Custom nodes for ComfyUI (V3 API)

Write new packs against the **V3 schema** — `io.ComfyNode`, `io.Schema`,
`ComfyExtension`, `comfy_entrypoint`. It is the current API; V1
(`INPUT_TYPES` / `NODE_CLASS_MAPPINGS`) still loads but is legacy: read it,
do not write it. Reference: https://docs.comfy.org/custom-nodes/v3_migration

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

## Test before you say it works

1. Syntax: `python -m py_compile nodes.py` with any Python you can run.
2. Import test needs ComfyUI's own environment (torch, `comfy_api`): from the
   install folder, `.venv/bin/python -c "import custom_nodes.<pack>.nodes"` —
   or skip it and say so; an import failure outside that environment is not a
   bug in the pack.
3. Restart ComfyUI. The startup log names a pack that failed to load and why.
4. Confirm the class is registered (`/object_info/<node_id>`, or the agent's
   `refresh_catalog` + `show_node`), then run it in a small workflow.

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

`comfy node init` writes the skeleton; `comfy node publish` uploads the
version in `pyproject.toml` (every git-tracked file); set `COMFY_API_KEY` in
CI. Requirements: list only what ComfyUI does not already ship (it has torch,
torchvision, numpy, Pillow, scipy, safetensors, transformers).
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
registers `io.NodeReplace(new_node_id=…, old_node_id=…, input_mapping=…)`.

## Going deeper

A nine-skill set (basics, inputs, outputs, datatypes, advanced, lifecycle,
frontend, migration, packaging), MIT-licensed, for Claude Code and any
`.agents/skills/` tool: https://github.com/jtydhr88/comfyui-custom-node-skills
