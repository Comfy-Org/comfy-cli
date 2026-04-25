import os
import pathlib
import re
import subprocess
from urllib.parse import urlparse, urlunparse

import tomlkit
import tomlkit.exceptions
import typer

from comfy_cli import ui
from comfy_cli.registry.types import (
    ComfyConfig,
    License,
    Model,
    ProjectConfig,
    PyProjectConfig,
    URLs,
)

# Mirrors pip's requirements-file comment rule: `#` only starts a comment when
# preceded by whitespace, so VCS URL fragments (`#subdirectory=`, `#egg=`) and
# direct-URL hashes (`#sha256=`) survive.
_inline_comment_re: re.Pattern[str] = re.compile(r"(^|\s+)#.*$")

# For `dynamic = ["version"]`: match a top-level `__version__` or `VERSION` assignment in a source file. Anchored
# to start-of-line (MULTILINE) so single-line comments are skipped. Horizontal whitespace only — no cross-line
# matching. Supports an optional PEP 526 type annotation. Straight quotes only. Backslash is excluded from the
# value class — escape sequences (`\n`, `\t`, `\"`, ...) cause the regex to fail to match, surfacing as a
# "could not find" warning rather than being silently misinterpreted. PEP 440 versions are ASCII-only so this
# is a clean fail-closed contract; users with auto-generated `__version__` containing escapes must clean up
# their source. The single-alternative character class also makes catastrophic backtracking impossible.
#
# Recommended user layout: a dedicated `_version.py` / `__version__.py` (NOT the package's `__init__.py`), so
# nothing in the file — module docstrings, assignments referenced in text, etc. — can collide with this regex.
# This matches hatch/setuptools convention for dynamic-version source files.
_VERSION_RE: re.Pattern[str] = re.compile(
    r"""^(?P<name>__version__|VERSION)
        (?:[\t ]*:[\t ]*[^=\n]+)?
        [\t ]*=[\t ]*
        (?:"(?P<dq>[^"\\\n]*)"|'(?P<sq>[^'\\\n]*)')
        """,
    re.MULTILINE | re.VERBOSE,
)


def create_comfynode_config():
    # Create the initial structure of the TOML document
    document = tomlkit.document()

    project = tomlkit.table()
    project["name"] = ""
    project["description"] = ""
    project["version"] = "1.0.0"
    project["dependencies"] = tomlkit.aot()
    project["license"] = "MIT"

    urls = tomlkit.table()
    urls["Repository"] = ""

    project.add("urls", urls)
    document.add("project", project)

    # Create the tool table
    tool = tomlkit.table()
    document.add(tomlkit.comment(" Used by Comfy Registry https://registry.comfy.org"))

    comfy = tomlkit.table()
    comfy["PublisherId"] = ""
    comfy["DisplayName"] = "ComfyUI-AIT"
    comfy["Icon"] = ""
    comfy["includes"] = tomlkit.array()

    # Add uncommentable hint for ComfyUI version compatibility, below of "[tool.comfy].includes" field.
    comfy["includes"].comment("""
# "requires-comfyui" = ">=1.0.0"  # ComfyUI version compatibility
""")

    tool.add("comfy", comfy)
    document.add("tool", tool)

    # Add the default model
    # models = tomlkit.array()
    # model = tomlkit.inline_table()
    # model["location"] = "/checkpoints/model.safetensor"
    # model["model_url"] = "https://example.com/model.zip"
    # models.append(model)
    # comfy["Models"] = models

    # Write the TOML document to a file
    try:
        with open("pyproject.toml", "w") as toml_file:
            toml_file.write(tomlkit.dumps(document))
    except OSError as e:
        raise Exception("Failed to write 'pyproject.toml'") from e


def sanitize_node_name(name: str) -> str:
    """Remove common ComfyUI-related prefixes from a string.

    Args:
        name: The string to process

    Returns:
        The string with any ComfyUI-related prefix removed
    """
    name = name.lower()
    prefixes = [
        "comfyui-",
        "comfyui_",
        "comfy-",
        "comfy_",
        "comfy",
        "comfyui",
    ]

    for prefix in prefixes:
        name = name.removeprefix(prefix)
    return name


def validate_and_extract_os_classifiers(classifiers: list) -> list:
    os_classifiers = [c for c in classifiers if c.startswith("Operating System :: ")]
    if not os_classifiers:
        return []

    os_values = [c[len("Operating System :: ") :] for c in os_classifiers]
    valid_os_prefixes = {"Microsoft", "POSIX", "MacOS", "OS Independent"}

    for os_value in os_values:
        if not any(os_value.startswith(prefix) for prefix in valid_os_prefixes):
            typer.echo(
                'Warning: Invalid Operating System classifier found. Operating System classifiers must start with one of: "Microsoft", "POSIX", "MacOS", "OS Independent". '
                'Examples: "Operating System :: Microsoft :: Windows", "Operating System :: POSIX :: Linux", "Operating System :: MacOS", "Operating System :: OS Independent". '
                "No OS information will be populated."
            )
            return []

    return os_values


def validate_and_extract_accelerator_classifiers(classifiers: list) -> list:
    accelerator_classifiers = [c for c in classifiers if c.startswith("Environment ::")]
    if not accelerator_classifiers:
        return []

    accelerator_values = [c[len("Environment :: ") :] for c in accelerator_classifiers]

    valid_accelerators = {
        "GPU :: NVIDIA CUDA",
        "GPU :: AMD ROCm",
        "GPU :: Intel Arc",
        "NPU :: Huawei Ascend",
        "GPU :: Apple Metal",
    }

    for accelerator_value in accelerator_values:
        if accelerator_value not in valid_accelerators:
            typer.echo(
                "Warning: Invalid Environment classifier found. Environment classifiers must be one of: "
                '"Environment :: GPU :: NVIDIA CUDA", "Environment :: GPU :: AMD ROCm", "Environment :: GPU :: Intel Arc", '
                '"Environment :: NPU :: Huawei Ascend", "Environment :: GPU :: Apple Metal". '
                "No accelerator information will be populated."
            )
            return []

    return accelerator_values


def validate_version(version: str, field_name: str) -> str:
    if not version:
        return version

    version_pattern = r"^(?:(==|>=|<=|!=|~=|>|<|<>|=)\s*)?(\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+)?)?$"

    version_parts = [part.strip() for part in version.split(",")]
    for part in version_parts:
        if not re.match(version_pattern, part):
            typer.echo(
                f'Warning: Invalid {field_name} format: "{version}". '
                f"Each version part must follow the pattern: [operator][version] where operator is optional (==, >=, <=, !=, ~=, >, <, <>, =) "
                f"and version is in format major.minor.patch[-suffix]. "
                f"Multiple versions can be comma-separated. "
                f'Examples: ">=1.0.0", "==2.1.0-beta", "1.5.2", ">=1.0.0,<2.0.0". '
                f"No {field_name} will be populated."
            )
            return ""

    return version


def _strip_url_credentials(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https") and (parsed.username or parsed.password):
        netloc = parsed.hostname or ""
        if ":" in netloc:
            netloc = f"[{netloc}]"
        if parsed.port:
            netloc += f":{parsed.port}"
        return urlunparse(parsed._replace(netloc=netloc))
    return url


def initialize_project_config():
    create_comfynode_config()

    with open("pyproject.toml") as file:
        document = tomlkit.parse(file.read())

    # Get the current git remote URL
    try:
        git_remote_url = subprocess.check_output(["git", "remote", "get-url", "origin"]).decode().strip()
        git_remote_url = _strip_url_credentials(git_remote_url)
    except subprocess.CalledProcessError as e:
        raise Exception("Could not retrieve Git remote URL. Are you in a Git repository?") from e

    # Convert SSH URL to HTTPS if needed
    if git_remote_url.startswith("git@github.com:"):
        git_remote_url = git_remote_url.replace("git@github.com:", "https://github.com/")

    # Ensure the URL ends with `.git` and remove it to obtain the plain URL
    repo_name = git_remote_url.rsplit("/", maxsplit=1)[-1].replace(".git", "")
    git_remote_url = git_remote_url.replace(".git", "")

    project = document.get("project", tomlkit.table())
    urls = project.get("urls", tomlkit.table())
    urls["Repository"] = git_remote_url
    urls["Documentation"] = git_remote_url + "/wiki"
    urls["Bug Tracker"] = git_remote_url + "/issues"

    project["urls"] = urls
    project["name"] = sanitize_node_name(repo_name)
    project["description"] = ""
    project["version"] = "1.0.0"

    # Use PEP 639 SPDX license identifier
    project["license"] = "MIT"

    # [project].classifiers Classifiers uncommentable hint for OS/GPU support
    # Attach classifiers comments to the project, below of "license" field.
    # will generate a comment like this:
    #
    # [project]
    # ...
    # license = "MIT"
    # # classifiers = [
    # #     # For OS-independent nodes (works on all operating systems)
    # ...

    project["license"].comment("""
# classifiers = [
#     # For OS-independent nodes (works on all operating systems)
#     "Operating System :: OS Independent",
#
#     # OR for OS-specific nodes, specify the supported systems:
#     "Operating System :: Microsoft :: Windows",  # Windows specific
#     "Operating System :: POSIX :: Linux",  # Linux specific
#     "Operating System :: MacOS",  # macOS specific
#
#     # GPU Accelerator support. Pick the ones that are supported by your extension.
#     "Environment :: GPU :: NVIDIA CUDA",    # NVIDIA CUDA support
#     "Environment :: GPU :: AMD ROCm",       # AMD ROCm support
#     "Environment :: GPU :: Intel Arc",      # Intel Arc support
#     "Environment :: NPU :: Huawei Ascend",  # Huawei Ascend support
#     "Environment :: GPU :: Apple Metal",    # Apple Metal support
# ]
""")

    tool = document.get("tool", tomlkit.table())
    comfy = tool.get("comfy", tomlkit.table())
    comfy["DisplayName"] = repo_name
    tool["comfy"] = comfy
    document["tool"] = tool

    # Handle dependencies
    if os.path.exists("requirements.txt"):
        with open("requirements.txt") as req_file:
            dependencies: list[str] = []
            for raw in req_file:
                # Strip inline/full-line comments, then skip pip-requirements-file
                # options (-r, -e, -c, --index-url, ...) which are not valid
                # PEP 508 deps and would break downstream build tooling.
                line = _inline_comment_re.sub("", raw).strip()
                if not line:
                    continue
                if line.startswith("-"):
                    print(
                        f"Warning: skipping pip-only option from requirements.txt (not valid as PEP 508 dep): {line!r}"
                    )
                    continue
                dependencies.append(line)
        project["dependencies"] = dependencies
    else:
        print("Warning: 'requirements.txt' not found. No dependencies will be added.")

    # Write the updated config to a new file in the current directory
    try:
        with open("pyproject.toml", "w") as toml_file:
            toml_file.write(tomlkit.dumps(document))
        print("pyproject.toml has been created successfully in the current directory.")
    except OSError as e:
        raise OSError("Failed to write 'pyproject.toml'") from e


def _resolve_dynamic_version(pyproject_dir: pathlib.Path, rel_path: str) -> str:
    """Read a version from a source file referenced by `[tool.comfy.version].path`.

    No Python execution — just text I/O and a regex, matching the contract
    agreed in issue #294. Returns empty string on any failure and emits a
    user-visible warning so scanning contexts degrade gracefully.
    """
    # Reject paths that are absolute under either POSIX or Windows rules —
    # `pathlib.Path.is_absolute()` alone is OS-specific (e.g., `/etc/foo` is
    # not considered absolute on Windows because it has no drive), and we
    # want identical rejection behavior regardless of the host OS.
    if pathlib.PurePosixPath(rel_path).is_absolute() or pathlib.PureWindowsPath(rel_path).is_absolute():
        typer.echo(
            f"Warning: `[tool.comfy.version].path` must be relative to pyproject.toml "
            f"(got `{rel_path}`). No version will be populated."
        )
        return ""
    path_obj = pathlib.Path(rel_path)

    pyproject_dir = pyproject_dir.resolve()
    resolved = (pyproject_dir / path_obj).resolve()
    try:
        resolved.relative_to(pyproject_dir)
    except ValueError:
        typer.echo(
            f"Warning: `[tool.comfy.version].path` must point inside the project directory "
            f"(got `{rel_path}`). No version will be populated."
        )
        return ""

    try:
        # `utf-8-sig` transparently strips a leading BOM — some Windows editors
        # add one, and it would defeat the `^__version__` anchor otherwise.
        text = resolved.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        typer.echo(f"Warning: could not read version file `{rel_path}`: {e}. No version will be populated.")
        return ""

    match = _VERSION_RE.search(text)
    if not match:
        typer.echo(
            f"Warning: could not find `__version__` or `VERSION` in `{rel_path}`. "
            f'The version file must contain a line like `__version__ = "1.2.3"`. '
            f"No version will be populated."
        )
        return ""

    # Exactly one of `dq` / `sq` was consumed by the regex. An empty capture
    # (`""` / `''`) is a valid match the regex accepts; if followed by another
    # quote the concat check below intercepts it (this also covers the
    # triple-quoted `"""..."""` case), otherwise we return "" and the
    # publish-layer guard surfaces it.
    raw = match.group("dq") if match.group("dq") is not None else match.group("sq")

    # Python concatenates adjacent string literals: `__version__ = "1." "2.3"`
    # (with or without whitespace between, quote styles freely mixed) evaluates
    # to "1.2.3". The regex captures only the first literal, so silently
    # returning `"1."` would POST a wrong version. Look ahead on the same line:
    # if the first non-whitespace char is a quote, reject the concatenation.
    # `;` (statement separator) and `#` (comment) are preserved because neither
    # starts with a quote.
    rest_of_line = text[match.end() :].split("\n", 1)[0]
    stripped_rest = rest_of_line.lstrip(" \t")
    if stripped_rest and stripped_rest[0] in ('"', "'"):
        typer.echo(
            f"Warning: `{match.group('name')}` in `{rel_path}` uses adjacent-string-literal "
            f"concatenation, which is not supported. Use a single assignment like "
            f'`{match.group("name")} = "1.2.3"`. No version will be populated.'
        )
        return ""

    return raw.strip()


def _parse_dynamic_fields(project_data) -> list[str]:
    """Return the `project.dynamic` field as a list of strings.

    Warns and returns `[]` if `dynamic` is present but has the wrong shape
    (e.g. a scalar string — a common PEP 621 misconfiguration).
    """
    dynamic_raw = project_data.get("dynamic", [])
    # tomlkit.Array inherits from list, so valid arrays (including empty `[]`)
    # pass through. Everything else is a misconfiguration.
    if not isinstance(dynamic_raw, (list, tuple)):
        typer.echo(
            "Warning: `project.dynamic` must be an array of strings. "
            'Use `dynamic = ["version"]` instead. '
            "No dynamic fields will be honored."
        )
        return []
    return [str(d) for d in dynamic_raw]


def _extract_version(project_data, comfy_data, pyproject_dir: pathlib.Path) -> str:
    """Return the project version, honoring PEP 621 `dynamic = ["version"]`.

    - Static `project.version` wins if present.
    - If absent and `"version"` is in `project.dynamic`, resolve via
      `[tool.comfy.version].path` (text-read + regex, no Python execution).
    - Otherwise return empty (existing behavior).
    """
    static_version = project_data.get("version", "")
    dynamic_fields = _parse_dynamic_fields(project_data)
    # Type-check runs BEFORE the truthy check so falsy non-strings (`version = 0`,
    # `version = 0.0`, `version = false`, `version = []`, `version = {}`) produce
    # the same named "must be a string" warning as truthy non-strings (`version = 1`,
    # `version = ["1","2"]`, `version = { path = "_v.py" }`). With the order reversed,
    # they would silently fall through to the dynamic branch and the user would
    # only see the downstream "project version is empty" error at publish time.
    if not isinstance(static_version, str):
        typer.echo("Warning: `project.version` must be a string. No version will be populated.")
        return ""
    if static_version:
        # Strip so `version = "  1.0.0  "` doesn't get POSTed with surrounding
        # whitespace. A whitespace-only `version = "   "` becomes "" and the
        # publish-layer guard surfaces it as "project version is empty".
        return static_version.strip()

    if "version" not in dynamic_fields:
        return ""

    version_cfg = comfy_data.get("version")
    if version_cfg is None:
        typer.echo(
            'Warning: `dynamic = ["version"]` declared but `[tool.comfy.version].path` is not set. '
            "See https://docs.comfy.org/registry/specifications for dynamic-version setup. "
            "No version will be populated."
        )
        return ""
    if not isinstance(version_cfg, dict):
        # A non-table value under `[tool.comfy].version` — the user likely
        # wrote `version = "x"` scalar (or any other type) instead of a nested
        # table.
        typer.echo(
            "Warning: `[tool.comfy].version` must be a table with a `path` key. "
            'Use `[tool.comfy.version]` with `path = "..."` instead. '
            "No version will be populated."
        )
        return ""
    # Order matters: check type BEFORE falsy-ness so that `path = 0` / `false`
    # / `[]` / `{}` produce a type warning, not a misleading "not set" warning.
    path_value = version_cfg.get("path")
    if path_value is not None and not isinstance(path_value, str):
        typer.echo("Warning: `[tool.comfy.version].path` must be a string. No version will be populated.")
        return ""
    if not path_value:
        typer.echo(
            "Warning: `[tool.comfy.version].path` is not set. "
            "See https://docs.comfy.org/registry/specifications for dynamic-version setup. "
            "No version will be populated."
        )
        return ""

    return _resolve_dynamic_version(pyproject_dir, path_value)


def extract_node_configuration(
    path: str = os.path.join(os.getcwd(), "pyproject.toml"),
) -> PyProjectConfig | None:
    if not os.path.isfile(path):
        ui.display_error_message("No pyproject.toml file found in the current directory.")
        return None

    try:
        # `utf-8-sig` strips a leading BOM if present — Windows editors sometimes
        # write one, and tomlkit would otherwise report `Empty key at line 1 col 0`.
        # `UnicodeDecodeError` must be in the except tuple: it is a `ValueError`,
        # not an `OSError`, and would otherwise escape and crash the caller.
        with open(path, encoding="utf-8-sig") as file:
            data = tomlkit.load(file)
    except (OSError, UnicodeDecodeError, tomlkit.exceptions.TOMLKitError) as e:
        ui.display_error_message(f"Could not parse `{path}`: {e}")
        return None

    project_data = data.get("project", {})
    if not isinstance(project_data, dict):
        # Degenerate TOML like `project = "hello"` at the root. Keep scanning
        # contexts alive by treating it as "no project metadata".
        typer.echo("Warning: `project` in pyproject.toml must be a table. Using defaults.")
        project_data = {}
    urls_data = project_data.get("urls", {})
    if not isinstance(urls_data, dict):
        urls_data = {}
    tool_data = data.get("tool", {})
    comfy_data = tool_data.get("comfy", {}) if isinstance(tool_data, dict) else {}
    if not isinstance(comfy_data, dict):
        comfy_data = {}

    dependencies = project_data.get("dependencies", [])
    supported_comfyui_frontend_version = ""
    for dep in dependencies:
        if isinstance(dep, str) and dep.startswith("comfyui-frontend-package"):
            supported_comfyui_frontend_version = dep.removeprefix("comfyui-frontend-package")
            break

    # Remove the ComfyUI-frontend dependency from the dependencies list
    dependencies = [
        dep for dep in dependencies if not (isinstance(dep, str) and dep.startswith("comfyui-frontend-package"))
    ]

    supported_comfyui_version = comfy_data.get("requires-comfyui", "")

    classifiers = project_data.get("classifiers", [])
    supported_os = validate_and_extract_os_classifiers(classifiers)
    supported_accelerators = validate_and_extract_accelerator_classifiers(classifiers)
    supported_comfyui_version = validate_version(supported_comfyui_version, "requires-comfyui")
    supported_comfyui_frontend_version = validate_version(
        supported_comfyui_frontend_version, "comfyui-frontend-package"
    )

    license_data = project_data.get("license", {})
    if isinstance(license_data, str):
        license = License(text=license_data)
    elif isinstance(license_data, dict):
        if "file" in license_data or "text" in license_data:
            license = License(file=license_data.get("file", ""), text=license_data.get("text", ""))
        else:
            typer.echo(
                'Warning: License should be in one of these two formats: license = {file = "LICENSE"} OR license = {text = "MIT License"}. Please check the documentation: https://docs.comfy.org/registry/specifications.'
            )
            license = License()
    else:
        license = License()
        typer.echo(
            'Warning: License should be in one of these two formats: license = {file = "LICENSE"} OR license = {text = "MIT License"}. Please check the documentation: https://docs.comfy.org/registry/specifications.'
        )

    pyproject_dir = pathlib.Path(path).parent
    version = _extract_version(project_data, comfy_data, pyproject_dir)

    project = ProjectConfig(
        name=project_data.get("name", ""),
        description=project_data.get("description", ""),
        version=version,
        requires_python=project_data.get("requires-python", ""),
        dependencies=dependencies,
        license=license,
        urls=URLs(
            homepage=urls_data.get("Homepage", ""),
            documentation=urls_data.get("Documentation", ""),
            repository=urls_data.get("Repository", ""),
            issues=urls_data.get("Issues", ""),
        ),
        supported_os=supported_os,
        supported_accelerators=supported_accelerators,
        supported_comfyui_version=supported_comfyui_version,
        supported_comfyui_frontend_version=supported_comfyui_frontend_version,
    )

    comfy = ComfyConfig(
        publisher_id=comfy_data.get("PublisherId", ""),
        display_name=comfy_data.get("DisplayName", ""),
        icon=comfy_data.get("Icon", ""),
        models=[Model(location=m["location"], model_url=m["model_url"]) for m in comfy_data.get("Models", [])],
        includes=comfy_data.get("includes", []),
        banner_url=comfy_data.get("Banner", ""),
        web=comfy_data.get("web", ""),
    )

    return PyProjectConfig(project=project, tool_comfy=comfy)
