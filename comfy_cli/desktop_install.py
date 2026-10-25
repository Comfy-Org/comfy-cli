"""Find a Comfy Desktop 2 install from the records Desktop keeps on disk.

Desktop 2 splits one install across up to three places, and none of them is
where comfy-cli's own default workspace lives:

* the **code dir** — a ComfyUI checkout at ``<installPath>/ComfyUI``;
* the **data dir** — ``models/``, ``custom_nodes/``, ``user/`` and the venv.
  For an install adopted from Desktop 1 that is ``adoptedBaseDir`` (Desktop
  launches it with ``--base-directory``); otherwise it is the code dir;
* **model roots** — the data dir's ``models/`` plus the shared library in
  ``settings.json`` → ``modelsDirs`` (unless the install opted out with
  ``useSharedModels: false``) plus the install's own ``modelDirs``.

Everything is read from files Desktop maintains in its app-data dir
(``installations.json``, ``settings.json``). Nothing here starts Desktop,
talks to a server, or writes a file.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from comfy_cli.constants import OS
from comfy_cli.utils import get_os

#: Desktop 2's Electron ``userData`` folder name.
APP_DIR_NAME = "Comfy Desktop"
MODERN_APP_DIR_NAME = "comfyui-desktop-2"
#: Desktop 1's ``userData`` folder, which holds ``config.json`` → ``basePath``.
LEGACY_APP_DIR_NAME = "ComfyUI"

_BARE_RELEASE_RE = re.compile(r"^\d+(?:\.\d+)+$")
_VERSION_MARKER_RE = re.compile(r"""^__version__\s*=\s*['"]([^'"\n]+)['"]""", re.MULTILINE)


def desktop_app_data_dir() -> Path | None:
    """Desktop 2's app-data dir on this OS (where ``installations.json`` lives)."""
    current = get_os()
    if current == OS.MACOS:
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    if current == OS.WINDOWS:
        appdata = os.environ.get("APPDATA")
        return Path(appdata) / APP_DIR_NAME if appdata else None
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME


def legacy_config_dir() -> Path | None:
    """Desktop 1's app-data dir (``config.json`` → ``basePath``); none on Linux."""
    current = get_os()
    if current == OS.MACOS:
        return Path.home() / "Library" / "Application Support" / LEGACY_APP_DIR_NAME
    if current == OS.WINDOWS:
        appdata = os.environ.get("APPDATA")
        return Path(appdata) / LEGACY_APP_DIR_NAME if appdata else None
    return None


@dataclass(frozen=True)
class ModelRoot:
    path: str
    #: ``data`` (the install's own ``models/``), ``shared`` (``settings.json``
    #: ``modelsDirs``) or ``install`` (the install record's own ``modelDirs``).
    source: str
    exists: bool

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "source": self.source, "exists": self.exists}


@dataclass(frozen=True)
class DesktopInstall:
    install_id: str | None
    name: str | None
    code_dir: str
    data_dir: str
    python: str | None
    model_roots: tuple[ModelRoot, ...]
    comfyui_version: str | None
    comfyui_commit: str | None
    adopted: bool
    status: str | None
    last_launched_ms: int | None
    #: The code dir holds a ComfyUI checkout (``main.py``) — only such an
    #: install can be selected.
    exists: bool

    @property
    def last_launched_at(self) -> str | None:
        if not self.last_launched_ms:
            return None
        try:
            stamp = datetime.fromtimestamp(self.last_launched_ms / 1000, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
        return stamp.isoformat(timespec="seconds").replace("+00:00", "Z")

    def summary(self, *, selected: bool) -> dict[str, Any]:
        """One row of ``desktop.installs``: enough to tell the installs apart."""
        return {
            "install_id": self.install_id,
            "name": self.name,
            "code_dir": self.code_dir,
            "data_dir": self.data_dir,
            "comfyui_version": self.comfyui_version,
            "last_launched_at": self.last_launched_at,
            "status": self.status,
            "exists": self.exists,
            "selected": selected,
        }


@dataclass(frozen=True)
class DesktopDetection:
    app_data_dir: str
    installs: tuple[DesktopInstall, ...] = ()
    selected: DesktopInstall | None = None
    selected_reason: str | None = None
    #: Records that exist but could not be read, as ``"<file>: <reason>"``.
    problems: tuple[str, ...] = field(default_factory=tuple)
    input_dir: str | None = None
    output_dir: str | None = None
    output_dir_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The ``desktop`` block of ``comfy which`` / ``comfy env``.

        The selected install's fields sit at the top level so an agent reads
        ``desktop.data_dir`` directly; ``installs`` lists every local install.
        """
        chosen = self.selected
        block: dict[str, Any] = {
            "install_id": chosen.install_id if chosen else None,
            "name": chosen.name if chosen else None,
            "code_dir": chosen.code_dir if chosen else None,
            "data_dir": chosen.data_dir if chosen else None,
            "custom_nodes_dir": os.path.join(chosen.data_dir, "custom_nodes") if chosen else None,
            "python": chosen.python if chosen else None,
            "model_roots": [root.to_dict() for root in chosen.model_roots] if chosen else [],
            "comfyui_version": chosen.comfyui_version if chosen else None,
            "comfyui_commit": chosen.comfyui_commit if chosen else None,
            "selected_reason": self.selected_reason,
            "installs": [install.summary(selected=install is chosen) for install in self.installs],
            "app_data_dir": self.app_data_dir,
        }
        if self.input_dir is not None:
            block["input_dir"] = self.input_dir
        if self.output_dir is not None:
            block["output_dir"] = self.output_dir
            block["output_dir_source"] = self.output_dir_source
        return block


def _same_path(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _venv_python(venv: Path) -> Path | None:
    for candidate in (venv / "bin" / "python3", venv / "bin" / "python", venv / "Scripts" / "python.exe"):
        if candidate.is_file():
            return candidate
    return None


def _code_dir(install_path: Path) -> Path:
    """Mirror Desktop's ``resolveComfyDir``: ``<installPath>/ComfyUI``, or the
    install path itself when the checkout sits there directly."""
    nested = install_path / "ComfyUI"
    if not nested.is_dir() and (install_path / "main.py").is_file():
        return install_path
    return nested


def _version_from_marker(code_dir: Path) -> str | None:
    try:
        match = _VERSION_MARKER_RE.search((code_dir / "comfyui_version.py").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None
    return match.group(1) if match else None


def _as_tag(version: str) -> str:
    version = version.strip()
    return "v" + version if _BARE_RELEASE_RE.match(version) else version


def _str_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str) and item] if isinstance(value, list) else []


def _model_roots(data_dir: Path, record: dict, shared_dirs: list[str]) -> tuple[ModelRoot, ...]:
    """Every root ComfyUI searches for this install, in search order, de-duplicated.

    Desktop writes the shared and per-install dirs into an
    ``--extra-model-paths-config`` YAML at launch; ComfyUI always searches the
    data dir's own ``models/`` as well.
    """
    candidates: list[tuple[str, str]] = [(str(data_dir / "models"), "data")]
    if record.get("useSharedModels") is not False:
        candidates += [(os.path.abspath(os.path.expanduser(d)), "shared") for d in shared_dirs]
    candidates += [(os.path.abspath(os.path.expanduser(d)), "install") for d in _str_list(record.get("modelDirs"))]
    roots: list[ModelRoot] = []
    for path, source in candidates:
        if any(_same_path(path, root.path) for root in roots):
            continue
        roots.append(ModelRoot(path=path, source=source, exists=os.path.isdir(path)))
    return tuple(roots)


def _install_from_record(record: dict, shared_dirs: list[str], legacy_base_path: str | None) -> DesktopInstall | None:
    install_path = record.get("installPath")
    if not isinstance(install_path, str) or not install_path:
        return None  # a remote entry (Comfy Cloud) has no local files
    install_root = Path(os.path.expanduser(install_path))
    code_dir = _code_dir(install_root)
    adopted = record.get("adopted") is True
    base_dir = record.get("adoptedBaseDir") if adopted else None
    if adopted and not (isinstance(base_dir, str) and base_dir):
        base_dir = legacy_base_path
    data_dir = Path(os.path.expanduser(base_dir)) if isinstance(base_dir, str) and base_dir else code_dir

    python: Path | None = None
    adopted_python = record.get("adoptedPythonPath") if adopted else None
    if isinstance(adopted_python, str) and Path(adopted_python).is_file():
        python = Path(adopted_python)
    if python is None:
        python = _venv_python(data_dir / ".venv")

    version_record = record.get("comfyVersion") if isinstance(record.get("comfyVersion"), dict) else {}
    base_tag = version_record.get("baseTag")
    version = base_tag if isinstance(base_tag, str) and base_tag else _version_from_marker(code_dir)
    commit = version_record.get("commit")
    last_launched = record.get("lastLaunchedAt")

    return DesktopInstall(
        install_id=record.get("id") if isinstance(record.get("id"), str) else None,
        name=record.get("name") if isinstance(record.get("name"), str) else None,
        code_dir=str(code_dir),
        data_dir=str(data_dir),
        python=str(python) if python else None,
        model_roots=_model_roots(data_dir, record, shared_dirs),
        comfyui_version=_as_tag(version) if version else None,
        comfyui_commit=commit if isinstance(commit, str) and commit else None,
        adopted=adopted,
        status=record.get("status") if isinstance(record.get("status"), str) else None,
        last_launched_ms=last_launched if isinstance(last_launched, (int, float)) and last_launched > 0 else None,
        exists=(code_dir / "main.py").is_file(),
    )


def _select(installs: tuple[DesktopInstall, ...]) -> tuple[DesktopInstall | None, str | None]:
    """Pick the install an agent should assume, and say why."""
    usable = [install for install in installs if install.exists]
    if not usable:
        return None, None
    if len(usable) == 1:
        return usable[0], "the only Comfy Desktop install with a ComfyUI checkout on disk"
    launched = [install for install in usable if install.last_launched_at]
    if launched:
        chosen = max(launched, key=lambda install: install.last_launched_ms or 0)
        return chosen, (
            f"the most recently launched of {len(usable)} Comfy Desktop installs (last launched "
            f"{chosen.last_launched_at}); pass --workspace <code_dir> to pick another"
        )
    return usable[0], (
        f"the first of {len(usable)} Comfy Desktop installs in installations.json (none has been launched); "
        "pass --workspace <code_dir> to pick another"
    )


def app_data_dirs() -> tuple[Path, ...]:
    """Known registry folders, keeping an explicit/test override isolated."""
    primary = desktop_app_data_dir()
    if primary is None:
        return ()
    if primary.name not in (APP_DIR_NAME, MODERN_APP_DIR_NAME):
        return (primary,)
    return tuple(dict.fromkeys((primary, primary.parent / MODERN_APP_DIR_NAME, primary.parent / APP_DIR_NAME)))


def _argv_values(argv: list[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(argv):
        if item.startswith(flag + "="):
            values.append(item.split("=", 1)[1])
        elif item == flag and index + 1 < len(argv):
            for value in argv[index + 1 :]:
                if value.startswith("--"):
                    break
                values.append(value)
                if flag != "--extra-model-paths-config":
                    break
    return values


def server_launch_paths(stats: dict[str, Any]) -> dict[str, list[str]]:
    """Return only known filesystem launch arguments, excluding unrelated credential flags."""
    system = stats.get("system") or {}
    argv = system.get("argv") if isinstance(system, dict) else None
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        return {}
    return {
        name: values
        for name in ("base_directory", "input_directory", "output_directory", "extra_model_paths_config")
        if (values := _argv_values(argv, "--" + name.replace("_", "-")))
    }


def _for_server(found: DesktopDetection, stats: dict[str, Any]) -> DesktopDetection:
    """Select using launch identity, never a remembered last-launched record."""
    system = stats.get("system") or {}
    if not isinstance(system, dict):
        system = {}
    argv = system.get("argv") or []
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        argv = []
    bases = _argv_values(argv, "--base-directory")
    configs = _argv_values(argv, "--extra-model-paths-config")
    scripts = [arg for arg in argv if Path(arg).is_absolute() and Path(arg).name == "main.py"]
    matches = [
        install
        for install in found.installs
        if install.exists
        and (
            any(_same_path(base, install.data_dir) for base in bases)
            or any(_same_path(str(Path(script).parent), install.code_dir) for script in scripts)
            or any(Path(config).stem == install.install_id for config in configs)
        )
    ]
    if len(matches) != 1:
        detail = "does not identify one recorded Desktop installation"
        return replace(
            found,
            selected=None,
            selected_reason=None,
            problems=(*found.problems, f"running server {detail}; pass explicit model/input roots"),
        )
    chosen = matches[0]
    outputs = _argv_values(argv, "--output-directory")
    if any(
        not Path(path).expanduser().is_absolute()
        for path in (*bases, *_argv_values(argv, "--input-directory"), *outputs)
    ):
        return replace(
            found,
            selected=None,
            selected_reason=None,
            problems=(
                *found.problems,
                "running server uses relative data/input/output paths; verify its working directory",
            ),
        )
    data_dir = str(Path(bases[-1]).expanduser()) if bases else chosen.data_dir
    roots = [root for root in chosen.model_roots if root.source != "data"] if not configs else []
    roots.insert(0, ModelRoot(str(Path(data_dir) / "models"), "data", (Path(data_dir) / "models").is_dir()))
    problems = list(found.problems)
    for config in configs:
        config_path = Path(config).expanduser()
        if not config_path.is_absolute():
            problems.append(f"running server model config is relative: {config}; verify its working directory")
            continue
        try:
            import yaml

            contents = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(contents, dict):
                raise ValueError("expected a mapping")
            for row in contents.values():
                if not isinstance(row, dict):
                    continue
                base = row.get("base_path")
                base_path = None
                if isinstance(base, str) and base:
                    base_path = Path(os.path.expandvars(base)).expanduser()
                    if not base_path.is_absolute():
                        base_path = config_path.parent / base_path
                    base_path = base_path.resolve()
                paths = [base_path] if base_path is not None else []
                for kind, value in row.items():
                    if kind in ("base_path", "is_default"):
                        continue
                    if not isinstance(value, str):
                        raise ValueError(f"{kind}: expected newline-separated model paths")
                    for entry in value.splitlines():
                        if not entry:
                            continue
                        path = Path(entry)
                        if not path.is_absolute():
                            path = (base_path or config_path.parent) / path
                        path = path.resolve()
                        # A base covers its descendant type folders. Keep a
                        # separate root for absolute paths outside that base.
                        if base_path is None or not path.is_relative_to(base_path):
                            paths.append(path)
                for path in paths:
                    if not any(_same_path(str(path), root.path) for root in roots):
                        roots.append(ModelRoot(str(path), "install", path.is_dir()))
        except (OSError, ValueError, yaml.YAMLError) as error:
            problems.append(f"{config}: {error}")
    python = system.get("python")
    version = system.get("comfyui_version")
    chosen = replace(
        chosen,
        comfyui_version=_as_tag(version) if isinstance(version, str) and version else chosen.comfyui_version,
        data_dir=data_dir,
        model_roots=tuple(roots),
        python=python if isinstance(python, str) and Path(python).is_file() else chosen.python,
    )
    inputs = _argv_values(argv, "--input-directory")
    input_dir = str(Path(inputs[-1]).expanduser()) if inputs else str(Path(data_dir) / "input")
    return replace(
        found,
        selected=chosen,
        selected_reason="matches the running server launch arguments",
        installs=tuple(chosen if item is matches[0] else item for item in found.installs),
        input_dir=input_dir,
        output_dir=str(Path(outputs[-1]).expanduser()) if outputs else str(Path(data_dir) / "output"),
        output_dir_source="server_argv" if outputs else "server_default",
        problems=tuple(problems),
    )


def detect(
    app_data_dir: Path | None = None, legacy_dir: Path | None = None, *, server_stats: dict[str, Any] | None = None
) -> DesktopDetection | None:
    """Read Desktop 2's install records. ``None`` means Desktop 2 is not installed.

    The app-data dirs default to this OS's; tests pass fixture dirs instead.
    """
    if app_data_dir is None:
        detections = [found for directory in app_data_dirs() if (found := detect(directory, legacy_dir)) is not None]
        if not detections:
            return None
        installs = tuple(install for found in detections for install in found.installs)
        selected, reason = _select(installs)
        origin = next(
            (found.app_data_dir for found in detections if selected in found.installs), detections[0].app_data_dir
        )
        found = DesktopDetection(
            origin, installs, selected, reason, tuple(problem for item in detections for problem in item.problems)
        )
        return _for_server(found, server_stats) if server_stats is not None else found
    app_dir = app_data_dir
    if app_dir is None:
        return None
    installations_file = app_dir / "installations.json"
    if not installations_file.is_file():
        return None

    problems: list[str] = []
    try:
        records = _read_json(installations_file)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return DesktopDetection(app_data_dir=str(app_dir), problems=(f"{installations_file}: {error}",))
    if not isinstance(records, list):
        return DesktopDetection(
            app_data_dir=str(app_dir), problems=(f"{installations_file}: expected a list of installs",)
        )

    shared_dirs: list[str] = []
    settings_file = app_dir / "settings.json"
    if settings_file.is_file():
        try:
            settings = _read_json(settings_file)
            shared_dirs = _str_list(settings.get("modelsDirs")) if isinstance(settings, dict) else []
        except (OSError, UnicodeDecodeError, ValueError) as error:
            problems.append(f"{settings_file}: {error} (shared model roots are missing from model_roots)")

    legacy_base_path: str | None = None
    legacy = legacy_dir if legacy_dir is not None else legacy_config_dir()
    if legacy is not None and (legacy / "config.json").is_file():
        try:
            config = _read_json(legacy / "config.json")
            base = config.get("basePath") if isinstance(config, dict) else None
            legacy_base_path = base if isinstance(base, str) and base else None
        except (OSError, UnicodeDecodeError, ValueError) as error:
            legacy_base_path = None
            problems.append(f"{legacy / 'config.json'}: {error}")

    if legacy_base_path is None and any(
        isinstance(record, dict)
        and record.get("adopted") is True
        and not (isinstance(record.get("adoptedBaseDir"), str) and record["adoptedBaseDir"])
        for record in records
    ):
        problems.append(
            "an adopted Desktop installation has no readable data path; its data_dir and derived model roots "
            "fall back to the code directory and are unverified"
        )

    installs = tuple(
        install
        for install in (
            _install_from_record(record, shared_dirs, legacy_base_path)
            for record in records
            if isinstance(record, dict)
        )
        if install is not None
    )
    selected, reason = _select(installs)
    found = DesktopDetection(
        app_data_dir=str(app_dir),
        installs=installs,
        selected=selected,
        selected_reason=reason,
        problems=tuple(problems),
    )
    return _for_server(found, server_stats) if server_stats is not None else found


def install_for_dir(path: str | os.PathLike[str], detection: DesktopDetection | None = None) -> DesktopInstall | None:
    """The Desktop install whose code dir or data dir is ``path``, if any."""
    found = detection if detection is not None else detect()
    if found is None:
        return None
    target = os.fspath(path)
    for install in found.installs:
        if _same_path(target, install.data_dir) or _same_path(target, install.code_dir):
            return install
    return None
