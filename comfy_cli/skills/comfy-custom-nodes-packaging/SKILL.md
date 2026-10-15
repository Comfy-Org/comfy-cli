---
name: comfy-custom-nodes-packaging
description: Packaged fallback for custom-node registry metadata, validation, and publishing.
---

# Packaging fallback

Registry metadata lives in the pack's `pyproject.toml`. `comfy node init` refuses to overwrite an existing file, so edit in place. `[project]` carries `name`, `description`, `version`, `license`, `dependencies` and `urls.Repository`; `[tool.comfy]` carries `PublisherId`, `DisplayName`, `Icon` and `includes`.

The two compatibility ranges come from different places. `[tool.comfy].requires-comfyui` states the ComfyUI range; the frontend range is a `comfyui-frontend-package` entry in `[project].dependencies`, which is stripped from the published dependency list rather than installed. Declare operating systems and accelerators as `[project].classifiers`.

Run `comfy node validate` before `comfy node publish --token <token>`; it runs the publishing checks without uploading. An empty `project.version` fails, and a dynamic version needs `[tool.comfy.version].path`.

What the archive contains depends on git, and the two cases fail in opposite directions. In a git checkout it is git-tracked files, minus `.comfyignore` matches, plus every `[tool.comfy].includes` path — `includes` overrides `.comfyignore`, so an untracked asset that is not listed ships missing. Outside a git checkout, or with no usable `git`, the whole tree is archived minus `.comfyignore` matches, so an untracked secret or local weight ships. Read the archive's file list before publishing instead of reasoning from the tree.
