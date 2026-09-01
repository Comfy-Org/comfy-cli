---
name: comfy-build-pins
description: "Reference skill cited by comfy-build. Read it with `comfy skills show comfy-build-pins` when deciding what belongs in a comfy-build.yaml's pipDependencies, or when a build failed on a dependency conflict. Covers which pins to delete and which to keep, the package pairs that must move together, and how to predict a dependency conflict off the install's requirements files before spending a build on it."
---

# comfy-build-pins

The depth behind `comfy-build`'s pins rule. The parent skill already carries the
default — **cut the first build with `pipDependencies` emptied** — and that is
the right answer most of the time. Read this when you need to justify keeping a
line, or when a build failed on a conflict.

## Why an inherited freeze is the usual first failure

`init` fills `pipDependencies` with the whole pip freeze, behind a two-line
comment header naming the platform it was captured on. The builder applies every
line as a pip **override**, and an override *replaces* what every package
declared rather than capping it — a user line for a package the torch stack also
names replaces it in place. Left alone, a freeze taken on macOS with Python 3.13
forces those exact versions onto a linux Python 3.12 build.

So the freeze is evidence about the environment the workflow ran in, not a set of
pins for the environment being built. Emptying it lets the build resolve the
packs' own requirements against the base image's torch, which is what you want.

**Empty is the default, not a rule that outranks what you can already see.** The
reading below exists to avoid buying a conflict, and cutting empty after finding
one buys it anyway. A conflict you can state in a sentence goes into cut one,
disclosed.

## What to delete

Delete rather than curate: `torch`, `torchvision`, `torchaudio`, `triton`,
`xformers`, every `nvidia-*`, `comfyui-frontend-package`, `comfyui-manager`,
`comfyui-embedded-docs`, and any wheel that only exists on the source OS
(`pywin32`, `pyobjc*`).

A torch pin is the worst of these: pinning one member of that stack replaces the
base image's line for it and releases the other two, so one pin silently
un-pins two.

## What to keep, and the rules for anything you keep

Keep a line only when you can name why — a pack's own docs demand a version and
nothing else supplies it, or a named failure in `comfy-build-failures` tells you
to. Then:

- **`numpy` and `scipy` are one axis.** Pin one and you have chosen for the
  other, so pin both, to versions released for each other.
- **Two packages providing one import are one axis too.** `opencv-python` and
  `opencv-python-headless` both install `cv2`, so pin both to the same version
  number. Resolve the competing names together to get that number rather than
  recalling one:

  ```shell
  printf 'opencv-python\nopencv-python-headless\n' > pair.txt
  <install>/.venv/bin/uv pip compile pair.txt --python-version <py> --python-platform linux
  ```
- **An override forces a version, it never adds a package.** Pinning something
  nothing requires installs nothing, and the build says so.

## Predict the conflict instead of buying it

A build takes minutes to tell you two packages disagree. Most of that answer is
sitting in text files, so look before you cut.

```shell
cat <install>/requirements.txt <install>/custom_nodes/*/requirements.txt > declared.txt 2>/dev/null
```

**A hand-authored definition can do this too — clone the packs.** The
requirements files are not on this machine, but every pack in the definition
names a `repository` or a registry slug that resolves to one, and concatenating
their `requirements.txt` with core's reproduces what the builder resolves. It
found the real blocker in one port before a single build was spent:

```shell
# core, plus one shallow clone per pack in customNodes
git clone --depth 1 -b <baseComfyVersion> https://github.com/comfyanonymous/ComfyUI core

# per pack, when gitRef is a branch or tag:
git clone --depth 1 -b <gitRef> https://github.com/<org>/<repo> packs/<name>

# per pack, when gitRef is a commit SHA — `--branch` does NOT accept one
# ("fatal: Remote branch <sha> not found in upstream origin"), so fetch it:
git init -q packs/<name>
git -C packs/<name> fetch -q --depth 1 https://github.com/<org>/<repo> <gitRef>
git -C packs/<name> checkout -q FETCH_HEAD

cat core/requirements.txt packs/*/requirements.txt > declared.txt
```

**Clone each pack at the ref the definition pins, not its HEAD** — the same
distinction `comfy-build-authoring` draws about `latest_version`: a `gitRef`
entry means that commit, and HEAD's requirements are a different pack's. For a
`registryVersion` entry there is no ref to clone; take that pack's requirements
from its published archive, or say in the disclosure that its constraints went
unread.

Then strip the deletions listed above (`torch`, `torchvision`, `torchaudio`,
`triton`, `xformers`, `nvidia-*`, `comfyui-frontend-package`, …) and resolve.

**Pass the platform tag the build actually uses, because the tag decides which
conflict surfaces first.** On the same input, the default `manylinux_2_28`
reported an `open3d` wheel-tag problem — a red herring — while `manylinux_2_34`
reported the real blocker:

```shell
uv pip compile declared_clean.txt --python-version 3.12 --python-platform x86_64-manylinux_2_34
```

That blocker was `Imath>=3.1.0`, unsatisfiable: PyPI's `imath` has only 0.0.1
and 0.0.2, and the `Imath` *module* ships inside the `OpenEXR` package the same
pack already requires. Which is the second thing worth knowing:

**`pip install -r req.txt || true` in a source script is a smell.** It hides
every constraint failure the builder will enforce, so a source environment can
run happily for months carrying a requirement that has never once been
satisfied — which is exactly how that `Imath` line survived unnoticed.

**`requirements.txt` is the only file the build reads.** It resolves ComfyUI's
own plus one per pack, so a dependency declared only in a `pyproject.toml` is
never installed on its account, and a pyproject constraint you find on disk is
not one the build applies: one real pack asked for a bare `timm` in
`requirements.txt` and `timm==0.6.13` in its `pyproject.toml`. A pack shipping no
`requirements.txt` declares nothing and gets whatever the others pulled in, which
is the shape behind `declared custom nodes failed to import` naming a module
nothing asked for.

Three shapes in that text are worth a build each:

- **Two names for one import.** `opencv-python` and `opencv-python-headless` both
  install `cv2`; `pillow` and the abandoned `pil` both answer to `PIL`. A real
  install had four packs asking for both `cv2` names. One loses, and whichever
  loses, something breaks. A failing import names the module, never the pip
  package, so pick between them from what the packs declare and not from the log.
- **A ceiling on a shared package.** A line like
  `opencv-python-headless[ffmpeg]<=4.7.0.72` holds everyone at a 2023 build. That
  single line is the most common cause of a failed first build here, because that
  wheel predates NumPy 2 and aborts at import under it.
- **A pack pinning far below what the install runs.** Compare a pin against the
  freeze `init` captured. `timm==0.6.13` under an install running `1.0.28` is a
  pack untouched in two years, and that gap is the pin to write.

Ignore `torch`, `torchvision` and `torchaudio` in all of this. The build owns them
and they always differ.

### When a resolver is available, confirm it

The transitive answer needs one, and `uv` is usually already in the install:

```shell
<install>/.venv/bin/uv pip compile declared.txt --python-version <py> --python-platform linux -o resolved.txt
```

`<py>` is the base image's Python, which `comfy build refs base-images` names —
read it rather than assuming, since the catalog moves. That command needs the
sign-in, and so does the cut, so this is the point to sign in. If the user would
rather not yet, resolve against the install's own Python and say in the disclosure
that the build's Python is unconfirmed. A refusal to resolve is the clearest
possible finding: the error names both sides. Plain `pip` cannot do this reliably
for another platform, so do not force it.

**A warning is a finding too.** `uv` reporting that a package has no extra by the
name a pack asked for — say `[ffmpeg]` on a pinned wheel — corroborates that the
pin is old enough to have moved on. Read warnings, not only the exit status.

**A clean resolve is not an all-clear.** The ceiling case satisfies every
constraint: the six-pack install that failed resolves to `numpy==2.5.2` with
`opencv-python-headless==4.7.0.72` without complaint. Take a refusal as a finding
and a success as nothing learned about the three shapes above.

**When there is no resolver**, offer to install one and say plainly what it is
for. If the user would rather not, say the check was the reading above only, and
that the build is now the first thing that will disagree with you.

### What none of this can see

- **A binary compiled against another version.** Every constraint is satisfied
  and the pack still aborts with `numpy.core.multiarray failed to import`.
- **Install scripts.** Packs run their own at build time, outside the lock, so
  the final environment is not the one you resolved.

---

Back to `comfy skills show comfy-build` for the disclosure and the cut.
`comfy skills show comfy-build-failures` reads a build that already failed.
