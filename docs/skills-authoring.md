# Skills Authoring Guide

> How to write, validate, and install a domain skill — so future `anime-video`,
> `3d`, `audio-styles`, and image-editing skills all follow the same pattern.

---

## 1. Skill Format Contract

A skill is a directory named `<skill-name>` containing a single `SKILL.md` file.

**Directory name rules:** the name must match the slug pattern `[A-Za-z0-9][A-Za-z0-9_-]*`
(no leading dots, no path separators). This becomes the install subdirectory and is
validated to prevent traversal attacks.

**`SKILL.md` structure:**

```
---
name: <dir-name>
description: One-sentence summary shown in `comfy skills list`.
---

# Skill title

Body — the teaching content delivered verbatim to the agent.
```

Both `name:` and `description:` are required. The `name:` value must exactly match the
containing directory name (when installing from a directory path).

**Content conventions (mirror the bundled skills):**

- Lead with a concise **mechanism map** — what commands exist and when to reach for each.
- Every fenced `comfy` example must be a real, runnable command. Never invent flags.
- Teach **discovery-first**: never hardcode model or asset names — show the agent how
  to find them via `comfy models search`, `comfy nodes show`, and `comfy workflow fragment ls`.
- Keep section headings short; agents scan headings to navigate large skills.

---

## 2. Authoring Loop

```bash
# 1. Scaffold your skill directory
mkdir my-anime-video
cat > my-anime-video/SKILL.md <<'EOF'
---
name: my-anime-video
description: Anime-style image and video generation with style-model / lora discovery.
---

# My Anime Video Skill

...
EOF

# 2. Validate before installing
comfy skills validate ./my-anime-video

# 3. Install into Claude Code (project scope during development)
comfy skills install --skill ./my-anime-video --target claude-code --scope project

# 4. Iterate: edit SKILL.md, re-install
#    Re-installing backs up any agent edits as SKILL.md.<timestamp>.bak
comfy skills install --skill ./my-anime-video --target claude-code --scope project

# 5. Check status
comfy skills status --scope project
```

`comfy skills validate` exits 0 and emits `{valid: true, name: "..."}` on success,
or exits 1 with `error.code = skill_invalid` on any format violation.

---

## 3. Bundled vs Third-Party Skills

- **Bundled skills** (`comfy`, `comfy-fragments`, `comfy-debug`, `comfy-relay`) ship with
  the CLI and are always available by bare name: `comfy skills install --skill comfy`.
- **Third-party / local skills** are installed via a directory path: `--skill ./my-skill`.
  The path token is recognized by the presence of `/`, `./`, `~/`, or because the path exists.
- **Local `./fragments/<name>.json` shadows a bundled fragment of the same name** — the
  same shadowing rule applies to skills: a local path install does not replace a bundled skill
  unless you explicitly uninstall the bundled one first.

### Install Manifest

Every successful `claude-code` and `cursor` install records a provenance entry in
`<config-dir>/skills-manifest.json`:

```json
{
  "/path/to/.claude/skills/comfy-debug/SKILL.md": {
    "skill": "comfy-debug",
    "sha256": "<sha256 of installed content>",
    "cli_version": "1.2.3"
  }
}
```

`comfy skills uninstall` removes entries. `comfy skills status` uses the manifest to
compute a **state** for each installed skill:

| State | Meaning |
|---|---|
| `current` | File matches the current bundled content exactly. |
| `stale` | File matches what was installed (user has not edited) but the bundled version has moved on — reinstall to get the latest. |
| `modified` | Agent or user has edited the file since it was installed. |
| `missing` | Expected file does not exist on disk. |
| `unmanaged` | File exists but was never recorded by the installer (e.g. hand-placed). |

`prune_retired` only deletes a retired-name file when the manifest records it as
comfy-managed; unrecorded files at retired-name paths are left untouched so user-authored
skills are never silently destroyed.

---

## 4. Domain-Skill Pattern — Worked Example: `anime-video`

The same recipe applies to any new domain: anime, 3D, audio styles, image editing.

### Step 1 — Author fragments for the domain

Create `./fragments/anime_t2i.json` (or PR it into `comfy_cli/fragments_lib/`):

```json
{
  "_fragment": {
    "name": "anime_t2i",
    "description": "Anime-style text-to-image. Requires an anime checkpoint and optional lora.",
    "params": {
      "ckpt_name":   { "required": true,  "description": "discover via: comfy models search --type checkpoint --text anime" },
      "lora_name":   { "required": false, "description": "discover via: comfy models search --type lora --text style" },
      "lora_weight": { "default": 0.75 },
      "prompt":      { "required": true },
      "negative":    { "default": "lowres, bad anatomy" },
      "seed":        { "default": 0 },
      "steps":       { "default": 28 },
      "width":       { "default": 832 },
      "height":      { "default": 1216 }
    },
    "outputs": {
      "image": { "type": "IMAGE" }
    }
  }
}
```

Add a second fragment `./fragments/anime_i2v.json` wiring `image` input to a video node
(e.g. using `kling_i2v` or a new Wan i2v partner node).

All asset names (`ckpt_name`, `lora_name`) are **required params discovered at runtime** —
never hardcode them. This is the "never hardcode model names" rule in artifact form.

### Step 2 — Write blueprint patterns

```yaml
# anime_video.yaml
pipeline:
  - fragment: anime_t2i
    alias: frame
    params:
      prompt: "fennec fox shrine maiden, sakura petals, golden hour"
      ckpt_name: "{{ ckpt_name }}"   # agent fills from `comfy models search` output
  - fragment: anime_i2v
    alias: clip
    inputs: { image: $frame.image }
    params:
      duration: 5
```

For batch scenes use `foreach` with `chunk: N` — compose splits into N-item batches and
writes numbered files (`<stem>.000.json`, `<stem>.001.json`, …). Script against
`data.written`, not `data.out`, when chunking.

### Step 3 — Write the `SKILL.md`

The skill body teaches the agent:

1. **When to reach for this domain** (anime-style output: what signals trigger it).
2. **Discovery commands** for domain-specific asset types:
   ```bash
   comfy --json models search --type checkpoint --text "anime" --where cloud
   comfy --json models search --type lora --text "flat color style" --where cloud
   comfy --json nodes show CLIPTextEncode --where cloud   # verify input names
   ```
3. **The worked blueprint** from Step 2 — the agent copies and fills params verbatim
   from the discovery output.
4. **Compose and run:**
   ```bash
   comfy workflow compose anime_video.yaml -o anime_video.json
   comfy run --workflow anime_video.json --where cloud
   ```

### Step 4 — Validate and install

```bash
comfy skills validate ./anime-video
comfy skills install --skill ./anime-video --target claude-code
```

**The same recipe extends to any domain.** 3D, audio, and image-editing skills follow
identical structure — the CLI machinery (fragments resolver, compose, discovery, skills
installer) is domain-agnostic by design. New domains need only fragments + a blueprint
pattern + a SKILL.md that teaches discovery.

---

## 5. Quick Reference

| Command | Purpose |
|---|---|
| `comfy skills validate ./my-skill` | Check format before installing |
| `comfy skills install --skill ./my-skill` | Install third-party skill (all targets) |
| `comfy skills install --skill ./my-skill --target claude-code` | Install to one target |
| `comfy skills status` | Show state of all installed skills |
| `comfy skills list` | List bundled skills |
| `comfy workflow fragment ls` | List all fragments (bundled + local) |
| `comfy models search --type lora --text "anime"` | Discover assets before wiring |
