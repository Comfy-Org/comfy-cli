---
name: comfy-custom-nodes-migration
description: Packaged fallback for migrating a custom-node pack from V1 to V3 registration.
---

# Migration fallback

A module loads through exactly one registration route and `NODE_CLASS_MAPPINGS` wins: when present and not `None`, loading stops there and a `comfy_entrypoint` in the same module is never called. An empty mapping still wins; only absent or `None` falls through. A half-migrated module therefore presents as "the new nodes did not appear", with no error. Migrate a module whole, or keep the two generations in separate packs.

Changing a node id breaks every saved workflow that used it, so migrate identity explicitly rather than renaming.

Deprecating and replacing are two stages, not one recipe. `is_deprecated` marks a node that is **still registered**, steering new users away while old workflows keep running. A replacement takes over only once that class stops being registered. Deprecate first, register the replacement in the same release, and let the release that finally drops the old class carry the migration.

Build `io.NodeReplace(new_node_id=…, old_node_id=…, old_widget_ids=…, input_mapping=…, output_mapping=…)` and await `node_replacement.register(...)`. It fires only when the old class is **absent** from the loaded catalog; only the first replacement registered for an old id is used; unmapped inputs are dropped and widget values match positionally, which is what `old_widget_ids` orders; and moving a downstream link to a different output position needs an `output_mapping` entry. A runtime that does not expose `node_replacement` has no replacement path — state that limit rather than working around it.
