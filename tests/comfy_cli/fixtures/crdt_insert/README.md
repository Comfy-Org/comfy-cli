# crdt_insert fixtures

Canvases as they look AFTER the doc host (cmp) applied an `insert_workflow` op:
every node, link, group and subgraph-definition id deterministically remapped
from the op id.

Generated with comfy-multi-player `src/remap.ts` → `remapInsertedWorkflowIds`
at commit `622865b`, op id `1ed4449ae23f3bcc8b599de88f69fd6a`, over the
sibling fixtures:

| fixture | source |
| --- | --- |
| `image_z_image_turbo.cmp_inserted.json` | `../gallery/image_z_image_turbo.json` |
| `sd15_ui_workflow.cmp_inserted.json` | `../sd15_ui_workflow.json` |

Refresh (from a comfy-multi-player checkout, Node ≥ 23):

```ts
// run.ts — beside src/remap.ts (import "./digest.ts" instead of "./digest.js")
import { readFileSync } from "node:fs";
import { remapInsertedWorkflowIds } from "./remap.ts";
const [, , file, opId] = process.argv;
process.stdout.write(JSON.stringify(remapInsertedWorkflowIds(JSON.parse(readFileSync(file, "utf8")), opId), null, 2) + "\n");
```

`node run.ts <source.json> 1ed4449ae23f3bcc8b599de88f69fd6a > <fixture>`
