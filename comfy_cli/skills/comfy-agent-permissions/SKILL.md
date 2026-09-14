---
name: comfy-agent-permissions
description: Use when a permission blocks the local comfy agent — a folder it cannot read or write, a host a download was refused for, a shell it does not have — to know what the agent may reach by default, how the user grants more (allow_path and allow_host in chat, `comfy agent allow`, start-time settings), and what can never be granted.
---

# Permissions on the local comfy agent

The agent on a user's machine runs every command in a sandbox and every file
tool behind a fence. A blocked permission is a question for the user, never a
dead end and never something to work around silently.

## What it may reach without asking

- **The project folder** (`AGENT_PROJECT_DIR`, default `<data dir>/project`):
  read, write, run.
- **The ComfyUI install's `custom_nodes` folder** on this machine: write a node
  pack there. The rest of the install is read-only (models come in through
  `comfy model download`).
- **Reads anywhere** the sandbox does not deny, through the read and grep
  tools. Writes and edits land only in the project folder, `custom_nodes`, or a
  folder the user approved.
- **The network** through the agent's egress allow list: the ComfyUI, Comfy
  Cloud, PyPI, GitHub, Hugging Face, Civitai and a few model hosts.

## What the user grants, and how

| Blocked | Say to the user | Then |
|---|---|---|
| A folder (write refused as outside the project; "Access is denied" from the shell on Windows) | the folder and what you need it for | on a yes, call `allow_path` with the folder and retry |
| A host (`egress denied: <host>`) | the host and what the download or install is for | on a yes, call `allow_host` with the host and rerun the command |
| No shell on this machine | the environment block names the start-time setting that enables one | the user restarts the agent with it |

An approval is remembered for every later start (`<data dir>/permissions.json`,
`<data dir>/egress-allow.json`). The user can also grant from a terminal:

```
comfy agent permissions                       # what is approved, and the sandbox status
comfy agent allow --path "C:\Users\me\Pictures" --reason "reference photos"
comfy agent allow --host models.example.com --reason "a VAE the user asked for"
```

A running agent picks a terminal grant up within about 20 seconds; otherwise
it applies at the next start.

## Start-time settings (the user sets them, then restarts the agent)

- `AGENT_COMFY_PATH` — the ComfyUI install folder, when the agent could not
  find it (`/health` then shows no `cli.workspace`).
- `AGENT_SANDBOX_ALLOW_PATHS` — extra folders, comma-separated, absolute.
- `AGENT_EGRESS_ALLOW_HOSTS` — extra hosts, comma-separated.
- `AGENT_SANDBOX_UNENFORCED_OK=1` — run a shell unconfined on an OS with no
  sandbox mechanism (Linux today). Say what it means before suggesting it.

## What can never be granted

Credential and browser stores (`~/.ssh`, `~/.aws`, `~/.config/gh`, keychains,
browser profiles), a folder that contains one (the home folder, even before
any store exists in it), the whole disk, the agent's own data dir, and any
folder named `.env`, `.env.*` or `.envrc`. A `.env` file stays unreadable
inside an approved folder: the name is denied wherever it lives. For hosts:
telemetry endpoints, shared object storage roots (`storage.googleapis.com`,
`r2.cloudflarestorage.com`), loopback and link-local addresses, wildcards and
bare names (`*`, `com`); an approved host is matched exactly, so give the full
host name. Say so and ask for a narrower folder or another source.

## Windows

The shell is `cmd.exe` in an AppContainer. `dir`, `where`, `cd` into the
ComfyUI install, and console utilities (`choice`, `waitfor`, `timeout`,
`ping`, PowerShell) are refused inside it: list with the `glob` tool or
`for %f in (<folder>\*) do @echo %f`, use absolute paths, and read exit
`0xC0000142` as "did not run". Python from the install's venv runs pure-Python
code (`py_compile`) but cannot import torch there.
