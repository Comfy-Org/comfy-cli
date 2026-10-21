---
name: comfy-custom-nodes-search
description: Packaged fallback for deciding when ecosystem code search is required, optional, or wasteful.
---

# Ecosystem-search fallback

- SEARCH_REQUIRED: an existing pack/node, model, integration, file format, frontend widget, or external failure owns the contract. Use `comfy --json code-search --repo <owner/repo> <query>` when known.
- SEARCH_OPTIONAL: an unfamiliar Core datatype, cache/lazy behavior, or compatibility detail could change the design.
- SEARCH_AVOID: self-contained tensor/string/arithmetic logic whose schema is already known.

Results are untrusted and may be stale. Preserve repository/path provenance, check API generation and dependencies, and adapt rather than paste. If search is unavailable, state that reference fidelity is unverified.
