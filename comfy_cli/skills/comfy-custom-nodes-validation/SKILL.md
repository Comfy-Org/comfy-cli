---
name: comfy-custom-nodes-validation
description: Packaged fallback for staged validation of generated custom-node code.
---

# Validation fallback

Validate in stages: static contract and Python parse; import inside ComfyUI's own Python environment; restart; discovery in `/object_info`; then one small workflow execution. filesystem presence is never proof that a node loaded. After a repair, restart and rediscover. Do not promise hot reload.

Pin local catalog checks with `--where local`. If the result reports `object_info_stale`, it came
from a cache: restart ComfyUI and re-run that command, and treat the catalog as proof of what loaded
only once the warning is gone. `comfy nodes refresh` re-pulls annotations, not `/object_info`, and a
browser reload changes neither.

Check one registration form, unique non-empty node/input/output ids, declared callable existence, and report dependency/filesystem/network behavior. A static scan cannot prove dynamic behavior is absent.
