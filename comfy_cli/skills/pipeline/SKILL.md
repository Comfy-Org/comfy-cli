---
name: comfy-pipeline
description: Multi-stage pipeline orchestration — chaining workflows, parallel fan-out, upload/download composition, project layout. The connective tissue between image, video, and audio skills.
---

This skill is the orchestration layer. It assumes `comfy` (core) is loaded
and teaches how to compose the other skills (`comfy-image`, `comfy-video`,
`comfy-audio`) into multi-stage pipelines. It's about the **glue**, not
the individual stages.

---

## 0. Graph-first: build large workflows, not many small ones

ComfyUI's power is the **workflow graph**. The execution engine
automatically parallelizes independent branches, manages dependencies,
and schedules nodes — you don't have to.

**Default to building one large workflow** that wires generation,
transformation, and assembly nodes together in a single graph. Only
split into separate workflows when:

- An intermediate result needs **human review/selection** before continuing
- Different stages need **different routing** (`--where local` vs `--where cloud`)
- The workflow would exceed server memory constraints

Anti-pattern: submitting many small workflows with upload/download between
each. Let ComfyUI's DAG engine handle parallelism.

```json
// ✅ GOOD — one workflow graph, ComfyUI handles the DAG
{
  "1":  {"class_type": "LoadImage", "inputs": {"image": "frame1.png"}},
  "2":  {"class_type": "LoadImage", "inputs": {"image": "frame2.png"}},
  "3":  {"class_type": "LoadImage", "inputs": {"image": "frame3.png"}},
  "4":  {"class_type": "KlingFirstLastFrameNode", "inputs": {
           "first_frame": ["1", 0], "end_frame": ["2", 0], "prompt": "...", "...": "..."}},
  "5":  {"class_type": "KlingFirstLastFrameNode", "inputs": {
           "first_frame": ["2", 0], "end_frame": ["3", 0], "prompt": "...", "...": "..."}},
  "6":  {"class_type": "GetVideoComponents", "inputs": {"video": ["4", 0]}},
  "7":  {"class_type": "GetVideoComponents", "inputs": {"video": ["5", 0]}},
  "8":  {"class_type": "ImageBatch", "inputs": {"image1": ["6", 0], "image2": ["7", 0]}},
  "9":  {"class_type": "SoniloTextToMusic", "inputs": {"prompt": "...", "duration": 10}},
  "10": {"class_type": "CreateVideo", "inputs": {"fps": ["6", 2], "images": ["8", 0], "audio": ["9", 0]}},
  "11": {"class_type": "SaveVideo", "inputs": {"video": ["10", 0], "filename_prefix": "final"}}
}
```

In the good example: Kling nodes 4 and 5 run **in parallel** automatically
(they share no dependency). Sonilo music node 9 also runs in parallel with
them. ComfyUI only blocks CreateVideo (node 10) until all its inputs are
ready. **One submit, zero uploads/downloads in between, zero manual
orchestration.**

### When to split workflows

| Split when... | Why |
|---|---|
| You need to **pick the best** of N generated images before animating | Human review gate |
| Generation is local GPU, animation is cloud API | Different `--where` routing |
| The graph has 50+ heavy nodes and the server OOMs | Resource constraint |
| You want to **reuse** the same base assets in multiple different pipelines | Shared asset library |

Even when splitting, minimize the number of workflows. Two or three large
graphs beat twenty small ones.

---

## 1. The pipe operator: `|`

The pipe pattern (`comfy run … --wait | comfy download`) and project
layout (`workflows/`, `inputs/`, `outputs/`) are documented in the core
`comfy` skill. Use those conventions here.

---

## 2. The three-phase pipeline pattern

Most non-trivial productions follow three phases:

1. **Generate** base assets in parallel (images, audio — all submitted simultaneously)
2. **Transform** — download Phase 1 outputs, upload as inputs, submit I2V / edit workflows in parallel
3. **Assemble** — download Phase 2 outputs, concat clips + sync audio, download final

Each phase fans out internally but gates on the prior phase completing.

---

## 3. Parallel fan-out / collect

Submit independent jobs simultaneously, then collect. See the core
`comfy` skill's "Async + parallel" section for the full pattern.

```bash
PIDS=()
for f in ./workflows/phase1_*.json; do
    PID=$(comfy --json run --workflow "$f" --where cloud | jq -r .data.prompt_id)
    PIDS+=("$PID")
done

for p in "${PIDS[@]}"; do
    comfy --json jobs watch "$p" --where cloud | comfy download --where cloud
done
```

---

## 4. Stage handoff: download → upload → reference

The bridge between pipeline stages:

```bash
# 1. Download Phase 1 output
comfy --json jobs watch "$IMG_PID" --where cloud | comfy download --where cloud

# 2. Upload the downloaded image as input for Phase 2
CLOUD_NAME=$(comfy --json upload ./outputs/abc12345_000.png --where cloud \
    | jq -r '.data.uploads[0].cloud_name')

# 3. Build Phase 2 workflow referencing the uploaded file
# In the workflow JSON: {"class_type": "LoadImage", "inputs": {"image": "$CLOUD_NAME"}}
```

This is the key pattern that replaces manual `curl` + API key extraction.
`comfy upload` and `comfy download` handle auth internally.

---

## 5. Assembly workflow pattern

Video assembly (concatenating clips + audio sync) is documented in full
in the `comfy-video` skill. Use `GetVideoComponents` → `ImageBatch` →
`CreateVideo` → `SaveVideo`, always wiring fps from source.

---

## 6. Pipeline failure recovery

When one job in a parallel batch fails: re-submit only the failed workflow.
Use `comfy --json jobs status <id>` to identify which failed. Patch the
assembly workflow to skip the missing input or re-wire around it.

---

## 7. What NOT to do

- **Don't create many small workflows when one large graph works** — ComfyUI's
  execution engine parallelizes independent branches automatically. Build the
  full pipeline as a single workflow graph; only split when you need human
  review, different routing, or hit resource limits. (See §0.)
- Don't scatter files across `/tmp` — use the project layout from the core skill.
- Don't block on sequential jobs when they're independent — fan out, then
  collect.
- Don't forget to upload between stages — cloud workflows can't see your
  local files.
- Don't manually upload/download between nodes that could be wired together
  in the same workflow — every upload/download cycle is latency and fragility
  you don't need.
- If a single workflow JSON grows past ~200 lines, see the `comfy-subgraphs`
  skill.