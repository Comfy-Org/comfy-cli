"""PATH resolution for the `comfy build` command tree, and later for deploy.

Every build command takes one optional positional `PATH` that answers two
questions at once — *where is the spec* and *where is the install* — because a
ComfyUI install and its spec normally live in the same folder (build design
lines 291-312):

===================== ========================== ==================
You pass              Spec resolves to           Install root
===================== ========================== ==================
nothing               ``./comfy-build.yaml``     ``.``
a directory           ``<dir>/comfy-build.yaml`` ``<dir>``
a ``.yaml``/``.json`` that file                  the file's parent
===================== ========================== ==================

``--models-dir``, ``--custom-nodes-dir`` and ``--python`` override the INSTALL
half only — never the spec half. That is what makes split and Desktop layouts,
where the code and the data directory sit apart, expressible with a single
positional.

`ls` and the `refs` command groups are workspace-level and involve no spec, so
they ignore PATH entirely: they simply never call this module.

Deploy reuses the spec half verbatim and raises the *same*
``build_spec_not_found`` code rather than a second one (deploy design lines
251-271).

This module is pure — no Typer, no renderer, no filesystem writes. A missing
spec surfaces as :class:`BuildSpecNotFoundError`, which the command layer
renders through ``renderer.error(code=e.code, message=str(e), details=e.details)``
the way ``workflow_ops``' typed errors are rendered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from comfy_cli.command.build_spec import BuildSpecInvalidError
from comfy_cli.constants import DEFAULT_COMFY_MODEL_PATH

#: The canonical spec filename a directory PATH expands to.
SPEC_FILENAME = "comfy-build.yaml"

#: Suffixes that mark PATH as a spec FILE rather than a directory. Fixed by the
#: build design's resolution table. Classification is by suffix ALONE, never by
#: what is on disk, so one argument means one thing whether or not the file
#: exists yet — `init` resolves a spec it is about to create.
SPEC_SUFFIXES: frozenset[str] = frozenset({".yaml", ".json"})

#: ``custom_nodes/`` has no shared constant; ``command/build.py:80`` keeps its
#: own copy. Duplicated rather than imported so this module stays free of the
#: heavy command module it will be imported *by*.
CUSTOM_NODES_DIRNAME = "custom_nodes"


class BuildSpecNotFoundError(ValueError):
    """No spec at the path PATH resolved to.

    Every build command except `init` fails with this (build design lines
    307-309); `init` is the one command that resolves with
    ``require_spec=False``, because creating the spec is its whole job.

    Subclasses ``ValueError`` — matching the house typed errors in
    ``workflow_ops`` — deliberately rather than ``FileNotFoundError``, so a
    broad ``except OSError`` around neighbouring file IO cannot swallow it.
    """

    code = "build_spec_not_found"
    hint = "create a comfy-build.yaml at that path, or pass the PATH that holds the spec"

    def __init__(self, spec_file: Path) -> None:
        self.spec_file = spec_file
        # `details.path` is the EXACT absolute path probed, so an agent can
        # retry against it without re-deriving the resolution rules.
        self.details: dict[str, str] = {"path": str(spec_file)}
        super().__init__(f"no build spec at {spec_file}")


@dataclass(frozen=True, slots=True)
class InstallOverrides:
    """The three flags that redirect the install half of PATH.

    ``None`` means "not overridden — derive it from the install root".
    """

    models_dir: Path | None = None
    custom_nodes_dir: Path | None = None
    python: Path | None = None

    @classmethod
    def from_options(
        cls,
        models_dir: str | None = None,
        custom_nodes_dir: str | None = None,
        python: str | None = None,
    ) -> InstallOverrides:
        """Parse the raw Typer option strings once, at the boundary."""
        return cls(
            models_dir=Path(models_dir) if models_dir else None,
            custom_nodes_dir=Path(custom_nodes_dir) if custom_nodes_dir else None,
            python=Path(python) if python else None,
        )


@dataclass(frozen=True, slots=True)
class BuildPaths:
    """Both halves of a resolved PATH. Every path is absolute.

    ``python`` is the only optional member: there is no filesystem-free default
    for the interpreter, so ``None`` means "not overridden — detect it from
    ``install_root``".
    """

    spec_file: Path
    install_root: Path
    models_dir: Path
    custom_nodes_dir: Path
    python: Path | None


def resolve_local_path(scan_root: Path, local_path: str, *, entry: str) -> Path:
    """Resolve one spec localPath lexically against its own scan root."""
    root = scan_root.resolve()
    path = PurePosixPath(local_path)
    drive_qualified = re.match(r"^[A-Za-z]:", local_path) is not None
    if "\\" in local_path or drive_qualified or path.is_absolute() or ".." in path.parts:
        raise BuildSpecInvalidError(f"{entry}: invalid localPath {local_path!r}")
    return root.joinpath(*path.parts)


def _absolute(path: Path) -> Path:
    """Anchor ``path`` at the cwd without resolving symlinks or collapsing ``..``.

    ``Path.resolve()`` would rewrite a symlinked PATH into wherever it points,
    so a spec the caller named as ``link/comfy-build.yaml`` would be probed —
    and reported in ``build_spec_not_found`` — under a directory they never
    typed. Anchoring stays lexical so the resolved paths are the caller's own.
    """
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def resolve_build_paths(
    path: str | Path | None = None,
    *,
    overrides: InstallOverrides | None = None,
    require_spec: bool = True,
) -> BuildPaths:
    """Map the positional ``PATH`` to its spec file and its install root.

    Args:
        path: the positional ``PATH``. ``None`` (or empty) is the cwd.
        overrides: ``--models-dir``/``--custom-nodes-dir``/``--python``. These
            touch the install half only; ``spec_file`` is unaffected by them.
        require_spec: raise when the resolved spec is absent. `init` — the only
            command allowed to proceed without one — passes ``False``.

    Raises:
        BuildSpecNotFoundError: ``require_spec`` and no spec at the resolved
            path. Its ``details["path"]`` is the exact absolute path probed.
    """
    given = Path(path) if path else None
    # Classify the path the CALLER gave, not the cwd it may stand in for: a bare
    # `comfy build` run inside a directory literally named `foo.yaml` is still
    # row one, not the spec-file row.
    is_spec_file = given is not None and given.suffix in SPEC_SUFFIXES
    anchored = _absolute(given) if given is not None else Path.cwd()

    spec_file = anchored if is_spec_file else anchored / SPEC_FILENAME
    install_root = anchored.parent if is_spec_file else anchored

    if require_spec and not spec_file.is_file():
        raise BuildSpecNotFoundError(spec_file)

    opts = overrides or InstallOverrides()
    models_dir = _absolute(opts.models_dir) if opts.models_dir else install_root / DEFAULT_COMFY_MODEL_PATH
    if opts.custom_nodes_dir:
        custom_nodes_dir = _absolute(opts.custom_nodes_dir)
    elif opts.models_dir:
        # Split layout: `--models-dir` names the data root, and `custom_nodes/`
        # is its sibling — the rule `scan` already uses (build.py:783-784).
        custom_nodes_dir = models_dir.parent / CUSTOM_NODES_DIRNAME
    else:
        custom_nodes_dir = install_root / CUSTOM_NODES_DIRNAME

    return BuildPaths(
        spec_file=spec_file,
        install_root=install_root,
        models_dir=models_dir,
        custom_nodes_dir=custom_nodes_dir,
        python=_absolute(opts.python) if opts.python else None,
    )
