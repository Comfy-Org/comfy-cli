import os
import pathlib
import re
import subprocess
from urllib.parse import urlparse, urlunparse

import tomlkit
import tomlkit.exceptions
import typer
from tomlkit.items import Comment, Trivia

from comfy_cli import ui
from comfy_cli._safe_exec import resolve_required_binary
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


# Hint blocks the generated pyproject.toml carries as commented-out TOML for the user to
# uncomment. Each line is the comment BODY — tomlkit renders the leading "# ". They are
# emitted one comment item per line because tomlkit >= 0.15 rejects a comment containing
# line breaks (`ValueError: Comment cannot contain line breaks`).
_REQUIRES_COMFYUI_HINT = '"requires-comfyui" = ">=1.0.0"  # ComfyUI version compatibility'

_CLASSIFIERS_HINT = """\
classifiers = [
    # For OS-independent nodes (works on all operating systems)
    "Operating System :: OS Independent",

    # OR for OS-specific nodes, specify the supported systems:
    "Operating System :: Microsoft :: Windows",  # Windows specific
    "Operating System :: POSIX :: Linux",  # Linux specific
    "Operating System :: MacOS",  # macOS specific

    # GPU Accelerator support. Pick the ones that are supported by your extension.
    "Environment :: GPU :: NVIDIA CUDA",    # NVIDIA CUDA support
    "Environment :: GPU :: AMD ROCm",       # AMD ROCm support
    "Environment :: GPU :: Intel Arc",      # Intel Arc support
    "Environment :: NPU :: Huawei Ascend",  # Huawei Ascend support
    "Environment :: GPU :: Apple Metal",    # Apple Metal support
]"""


def _append_hint(table: tomlkit.items.Table, hint: str) -> None:
    """Append a hint block to `table` as one comment item per line."""
    for line in hint.split("\n"):
        # Blank separators are emitted as a bare "#" so the block stays one contiguous
        # comment. tomlkit.comment("") would render "# " and leave trailing whitespace
        # in the file we hand the user.
        table.add(tomlkit.comment(line) if line.strip() else Comment(Trivia(comment_ws="", comment="#")))


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
    _append_hint(comfy, _REQUIRES_COMFYUI_HINT)

    tool.add("comfy", comfy)
    document.add("tool", tool)

    # No default [tool.comfy].Models entry is scaffolded: a placeholder model
    # location/URL pair is not valid for any real node, and publishing rejects
    # it, so an empty section is a better starting point than a broken example.

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


# PEP 508 names are ASCII alphanumerics plus ``.``, ``-`` and ``_``, and must
# begin and end with an alphanumeric. A folder name is bound by none of that, so
# a name derived from one has to be normalized before it reaches
# ``[project].name``: ``comfy node init`` reports success either way and an
# invalid name only surfaces much later, at build or publish time.
_PROJECT_NAME_DISALLOWED_RE: re.Pattern[str] = re.compile(r"[^a-z0-9._-]+")

# Used when normalization leaves nothing usable — a folder named ``ComfyUI-``,
# one whose name is entirely non-ASCII, or the filesystem root. An obviously
# placeholder name the user is warned about beats an empty one that passes
# ``init`` and then fails at publish.
FALLBACK_PROJECT_NAME = "my-custom-nodes"


def normalize_project_name(name: str) -> str:
    """Coerce ``name`` into a valid PEP 508 / Registry ``[project].name``.

    Runs of disallowed characters collapse to a single ``-``, and leading and
    trailing separators are dropped so the result begins and ends with an
    alphanumeric. Returns :data:`FALLBACK_PROJECT_NAME`, with a warning, when
    nothing usable survives.

    Only ``[project].name`` is normalized. ``[tool.comfy].DisplayName`` keeps the
    original text, which is free-form and is what a human reads.
    """
    normalized = _PROJECT_NAME_DISALLOWED_RE.sub("-", name.lower()).strip("._-")
    if not normalized:
        print(
            f'Warning: could not derive a valid project name from "{name}". '
            f'Using "{FALLBACK_PROJECT_NAME}" — set `[project].name` in '
            "pyproject.toml before publishing."
        )
        return FALLBACK_PROJECT_NAME
    return normalized


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


# ``git remote get-url <name>`` exits 2 when there is no such remote — git's
# documented signal that the config key is simply absent. That is the ONLY
# non-zero remote exit treated as "you have no remote yet". Every other one is
# git declining to do its job (a repository-ownership refusal, a corrupt
# ``.git/config``, a permission error), and falling back there would overwrite a
# real repository's identity with folder-derived metadata while discarding git's
# own diagnostic.
_GIT_NO_SUCH_REMOTE_EXIT_CODE = 2

# git's wording for "there is no repository here, or above here". Matched as an
# allowlist rather than a denylist: a failure this does not recognise propagates,
# so a git diagnostic we have never seen fails loudly instead of silently
# scaffolding the wrong metadata. One deliberate consequence is that a bare
# repository ("this operation must be run in a work tree") raises with git's
# message rather than being guessed at.
_GIT_ABSENT_REPOSITORY_STDERR = "not a git repository"

# git reads these from the environment to decide WHICH repository it is talking
# about, so inheriting them makes the lookups in :func:`_resolve_origin_url`
# answer about a different one. ``GIT_DIR`` is the load-bearing case: set
# without ``GIT_WORK_TREE`` — which is exactly how git invokes a hook, and so
# how pre-commit invokes pytest — git treats the CURRENT directory as the top of
# the work tree, so ``rev-parse --show-toplevel`` answers with this folder, the
# rooted-here guard below passes, and ``remote get-url origin`` then returns the
# origin of the repository ``GIT_DIR`` points at. That is the same
# somebody-else's-URLs leak the guard exists to prevent, arriving by another
# route.
#
# The two discovery variables are stripped for the same reason rather than a
# different one: ``GIT_CEILING_DIRECTORIES`` can hide a repository that really is
# rooted here, and ``GIT_DISCOVERY_ACROSS_FILESYSTEM`` changes how far the search
# walks.
#
# Command-scope config — ``GIT_CONFIG_COUNT`` with its ``GIT_CONFIG_KEY_<n>`` /
# ``GIT_CONFIG_VALUE_<n>`` pairs — is deliberately NOT stripped, even though it
# does override config files and can carry ``remote.origin.url``. It is
# *protected* configuration in git's own terms, which is why granting
# ``safe.directory`` through it is the standard container recipe, and dropping it
# would turn an ownership refusal back into the hard error above for those users.
# What makes keeping it safe is that ``git remote get-url`` requires the remote to
# be configured in the REPOSITORY, so an injected ``remote.origin.url`` is visible
# to ``git config --get`` and still leaves ``get-url`` exiting 2 — the folder
# fallback, not somebody else's URL. Pinned by
# ``test_initialize_project_config_ignores_an_env_injected_origin_url``, measured
# on git 2.43.0.
_GIT_LOCATION_ENV_VARS = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_WORK_TREE",
    }
)


def _run_git(git_bin: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git, capturing both streams. Never raises on a non-zero exit.

    stderr is captured rather than discarded: it is the only thing that
    distinguishes "no repository here" from a refusal, and it is what the user
    needs to see when the refusal is the answer.

    The C locale is forced because that stderr is then *matched* against
    :data:`_GIT_ABSENT_REPOSITORY_STDERR`. git's messages are gettext-localized
    and Git for Windows and most distributions ship the translations, so without
    this a French or Chinese user's "not a git repository" would miss the
    allowlist and the fallback this function exists to enable would turn into a
    hard error. ``LANGUAGE`` is cleared as well: it takes precedence over
    ``LC_ALL`` in gettext's lookup.

    Inherited repository-location variables are dropped — see
    :data:`_GIT_LOCATION_ENV_VARS` — so the answer describes the current
    directory and not whatever repository the caller happened to be inside.

    The output is decoded as UTF-8 explicitly. ``text=True`` alone decodes with
    the parent's locale encoding, which on Windows is a legacy code page while
    git writes paths and config values as UTF-8 — so a pack whose path or origin
    URL contains non-ASCII characters either decoded to different characters,
    silently failing the repository-root comparison in
    :func:`_resolve_origin_url` and dropping valid URLs, or hit an unmapped byte
    and crashed ``comfy node init`` with a
    ``UnicodeDecodeError``. ``errors="replace"`` keeps a genuinely undecodable
    diagnostic from doing the same: a garbled message the user can still read
    beats a traceback.
    """
    env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATION_ENV_VARS}
    env.update({"LC_ALL": "C", "LANGUAGE": ""})
    return subprocess.run(
        [git_bin, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )


def _git_failure(invocation: str, completed: subprocess.CompletedProcess[str]) -> Exception:
    detail = completed.stderr.strip() or f"exited with status {completed.returncode}"
    return Exception(f"`git {invocation}` failed and the result is not usable:\n{detail}")


def _resolve_origin_url() -> str:
    """The ``origin`` URL of the git repository rooted at the current directory.

    Returns ``""`` — with a warning naming which case it was — for the three
    shapes a first publication legitimately has:

    * no git repository at all,
    * a repository that is not rooted at this folder, and
    * a repository with no ``origin`` remote yet.

    ``[project.urls]`` is optional for the Registry route, so those are
    scaffolding conditions rather than errors.

    Raises when git could not answer at all: absent or refused
    (``BinaryNotFoundError``, an ``OSError`` subclass), an ownership refusal, a
    corrupt config, a permission error. Those must not be mistaken for "no
    remote yet" — see :data:`_GIT_NO_SUCH_REMOTE_EXIT_CODE`.
    """
    try:
        git_bin = resolve_required_binary("git")

        # Resolve the repository root first. It answers "is there a repository,
        # and is it this folder" in one call, and it is what makes the remote
        # lookup below safe to trust.
        toplevel = _run_git(git_bin, "rev-parse", "--show-toplevel")
    except OSError as e:
        # ``BinaryNotFoundError`` subclasses ``OSError``: git absent, or found
        # and refused because the only match was CWD-anchored.
        raise Exception(f"Could not run git to read the remote URL. ({e})") from e

    if toplevel.returncode != 0:
        if _GIT_ABSENT_REPOSITORY_STDERR not in toplevel.stderr:
            raise _git_failure("rev-parse --show-toplevel", toplevel)
        print(
            "Note: this folder is not a git repository, so there is no `origin` remote to "
            "read. Leaving `[project.urls]` empty. Setting `Repository` in pyproject.toml is "
            "recommended — the Registry links it from your pack's page — but it is not required "
            "to publish."
        )
        return ""

    # ``git remote get-url`` searches PARENT directories, so the usual new-pack
    # layout — a plain folder under ``ComfyUI/custom_nodes/`` inside a cloned
    # ComfyUI — answers with ComfyUI's own origin. Naming the pack after that
    # would scaffold, and allow publishing, a pack whose Repository, Docs and
    # Bug Tracker all point at somebody else's project. So only a repository
    # rooted at THIS folder is allowed to describe this pack.
    repository_root = toplevel.stdout.strip()
    if pathlib.Path(repository_root).resolve() != pathlib.Path.cwd().resolve():
        print(
            f"Note: the enclosing git repository is {repository_root}, not this folder, so its "
            "`origin` remote does not describe this pack. Leaving `[project.urls]` empty. Run "
            "`git init` here and add an `origin` remote, or set `Repository` in pyproject.toml."
        )
        return ""

    remote = _run_git(git_bin, "remote", "get-url", "origin")
    if remote.returncode == _GIT_NO_SUCH_REMOTE_EXIT_CODE:
        print(
            "Note: this git repository has no `origin` remote yet. Leaving `[project.urls]` "
            "empty. Setting `Repository` in pyproject.toml is recommended — the Registry links "
            "it from your pack's page — but it is not required to publish."
        )
        return ""
    if remote.returncode != 0:
        raise _git_failure("remote get-url origin", remote)

    return _strip_url_credentials(remote.stdout.strip())


def initialize_project_config():
    # Resolve git BEFORE create_comfynode_config() writes pyproject.toml.
    #
    # ``comfy node init`` refuses to overwrite an existing pyproject.toml, so
    # anything that raises *after* that first write leaves a half-initialized
    # project with `name = ""` that the user cannot retry — which is the state
    # this function used to leave behind whenever git was absent or refused.
    # Resolving first means a git failure aborts with nothing written and
    # re-running after the fix works.
    #
    # A first publication is often made from a plain folder with no git
    # repository, or from one with no `origin` remote yet. That is not an error
    # for the Registry route: publishing needs `project.name` and
    # `project.version`, and `[project.urls]` is optional (``URLs`` defaults
    # every field to ""), so those scaffold without the URLs rather than
    # aborting. A git that cannot answer still aborts — see
    # :func:`_resolve_origin_url`.
    git_remote_url = _resolve_origin_url()

    create_comfynode_config()

    with open("pyproject.toml") as file:
        document = tomlkit.parse(file.read())

    project = document.get("project", tomlkit.table())
    urls = project.get("urls", tomlkit.table())

    if git_remote_url:
        # Convert SSH URL to HTTPS if needed
        if git_remote_url.startswith("git@github.com:"):
            git_remote_url = git_remote_url.replace("git@github.com:", "https://github.com/")

        # Strip a trailing `.git` to obtain the plain URL. `removesuffix`, not
        # `replace`: `replace` removes EVERY occurrence, so a host or path that
        # contains `.git` — `https://my.gitea.host/u/my.github.io.git` — came out
        # corrupted and pointing at a different host.
        repo_name = git_remote_url.rsplit("/", maxsplit=1)[-1].removesuffix(".git")
        git_remote_url = git_remote_url.removesuffix(".git")

        urls["Repository"] = git_remote_url
        urls["Documentation"] = git_remote_url + "/wiki"
        urls["Bug Tracker"] = git_remote_url + "/issues"
    else:
        # No remote to name the pack after, so the containing folder is the only
        # thing that does. ``Path.cwd().name`` is "" at the filesystem root.
        repo_name = pathlib.Path.cwd().name

    project_name = normalize_project_name(sanitize_node_name(repo_name))
    project["name"] = project_name
    project["description"] = ""
    project["version"] = "1.0.0"

    # Use PEP 639 SPDX license identifier
    project["license"] = "MIT"

    # [project].classifiers uncommentable hint for OS/GPU support, below of "license"
    # field. `urls` is removed and re-added so the hint lands between the two rather
    # than after the [project.urls] sub-table, which is where an append would put it.
    if "urls" in project:
        del project["urls"]
    _append_hint(project, _CLASSIFIERS_HINT)
    project["urls"] = urls

    tool = document.get("tool", tomlkit.table())
    comfy = tool.get("comfy", tomlkit.table())
    # DisplayName keeps the original text — it is free-form and is what a human
    # reads. It only borrows the normalized name when there was no text at all
    # (``comfy node init`` run at the filesystem root).
    comfy["DisplayName"] = repo_name or project_name
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
