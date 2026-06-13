import contextlib
import os
import pathlib
import time
from typing import Annotated
from urllib.parse import parse_qs, unquote, urlparse

import requests
import typer
import yaml
from rich import print
from rich.markup import escape

from comfy_cli import constants, tracking, ui
from comfy_cli.config_manager import ConfigManager
from comfy_cli.constants import DEFAULT_COMFY_MODEL_PATH
from comfy_cli.extra_model_paths import collect_extra_paths, paths_for_category
from comfy_cli.file_utils import DownloadException, check_unauthorized, download_file
from comfy_cli.workspace_manager import WorkspaceManager

app = typer.Typer()

workspace_manager = WorkspaceManager()
config_manager = ConfigManager()

_CIVITAI_SUBDOMAIN_SUFFIXES = tuple(f".{h}" for h in constants.CIVITAI_ALLOWED_HOSTS)


model_path_map = {
    "lora": "loras",
    "hypernetwork": "hypernetworks",
    "checkpoint": "checkpoints",
    "textualinversion": "embeddings",
    "controlnet": "controlnet",
}


def get_workspace() -> pathlib.Path:
    return pathlib.Path(workspace_manager.workspace_path)


def _resolve_default_relative_path(category: str | None, basemodel: str, extras: list) -> str:
    """Pick the destination subdir for a typed download.

    Returns an absolute path string when ``extras`` configures the category
    (pathlib's ``/`` operator discards the workspace prefix in that case).
    Otherwise returns the workspace-relative ``models/<category>/<basemodel>``
    form preserved from comfy-cli's existing behavior.
    """
    if category and extras:
        configured = paths_for_category(extras, category)
        if configured:
            return str(configured[0] / basemodel) if basemodel else str(configured[0])
    return os.path.join(DEFAULT_COMFY_MODEL_PATH, category or "", basemodel)


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""
    rounded = round(seconds, 1)
    if rounded < 60:
        return f"{rounded:.1f}s"
    minutes, secs = divmod(int(rounded), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


def potentially_strip_param_url(path_name: str) -> str:
    return path_name.split("?")[0]


def check_huggingface_url(url: str) -> tuple[bool, str | None, str | None, str | None, str | None]:
    """
    Check if the given URL is a Hugging Face URL and extract relevant information.

    Args:
        url (str): The URL to check.

    Returns:
        Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str]]:
            - is_huggingface_url (bool): True if it's a Hugging Face URL, False otherwise.
            - repo_id (Optional[str]): The repository ID if it's a Hugging Face URL, None otherwise.
            - filename (Optional[str]): The filename if present, None otherwise.
            - folder_name (Optional[str]): The folder name if present, None otherwise.
            - branch_name (Optional[str]): The git branch name if present, None otherwise.
    """
    parsed_url = urlparse(url)

    if parsed_url.netloc != "huggingface.co" and parsed_url.netloc != "huggingface.com":
        return False, None, None, None, None

    path_parts = [p for p in parsed_url.path.split("/") if p]

    if len(path_parts) < 5 or (path_parts[2] != "resolve" and path_parts[2] != "blob"):
        return False, None, None, None, None
    repo_id = f"{path_parts[0]}/{path_parts[1]}"
    branch_name = path_parts[3]

    remaining_path = "/".join(path_parts[4:])
    folder_name = os.path.dirname(remaining_path) if "/" in remaining_path else None
    filename = os.path.basename(remaining_path)

    # URL decode the filename
    filename = unquote(filename)

    return True, repo_id, filename, folder_name, branch_name


def check_civitai_url(url: str) -> tuple[bool, bool, int | None, int | None]:
    """
    Returns:
        is_civitai_model_url: True if the url is a civitai *web* model url (e.g. /models/12345)
        is_civitai_api_url: True if the url is a civitai *api* url useful for resolving downloads
        model_id: The model id (for /models/*), else None
        version_id: The version id (for /api/download/models/* or ?modelVersionId=), else None
    """
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host not in constants.CIVITAI_ALLOWED_HOSTS and not host.endswith(_CIVITAI_SUBDOMAIN_SUFFIXES):
            return False, False, None, None
        p_parts = [p for p in parsed.path.split("/") if p]
        query = parse_qs(parsed.query)

        if len(p_parts) >= 4 and p_parts[0] == "api":
            # Case 1: /api/download/models/<version_id>
            # e.g. https://civitai.com/api/download/models/1617665?type=Model&format=SafeTensor
            if p_parts[1] == "download" and p_parts[2] == "models":
                try:
                    version_id = int(p_parts[3])
                    return False, True, None, version_id
                except ValueError:
                    return False, True, None, None

            # Case 2: /api/v1/model-versions/<version_id>
            if p_parts[1] == "v1" and p_parts[2] in ("model-versions", "modelVersions"):
                try:
                    version_id = int(p_parts[3])
                    return False, True, None, version_id
                except ValueError:
                    return False, True, None, None

        # Case 3: /models/<model_id>[/*] with optional ?modelVersionId=<id>
        # e.g. https://civitai.com/models/43331
        #      https://civitai.com/models/43331/majicmix-realistic?modelVersionId=485088
        if len(p_parts) >= 2 and p_parts[0] == "models":
            try:
                model_id = int(p_parts[1])
            except ValueError:
                return False, False, None, None
            version_id = None
            mv = query.get("modelVersionId")
            if mv and len(mv) > 0:
                with contextlib.suppress(ValueError):
                    version_id = int(mv[0])
            if version_id is None:
                mv = query.get("version")
                if mv and len(mv) > 0:
                    with contextlib.suppress(ValueError):
                        version_id = int(mv[0])
            return True, False, model_id, version_id

        return False, False, None, None

    except Exception:
        print("Error parsing CivitAI model URL")
        return False, False, None, None


def request_civitai_model_version_api(version_id: int, headers: dict | None = None):
    # Make a request to the CivitAI API to get the model information
    response = requests.get(
        f"https://civitai.com/api/v1/model-versions/{version_id}",
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()  # Raise an error for bad status codes

    model_data = response.json()
    for file in model_data["files"]:
        if file.get("primary", False):  # Assuming we want the primary file
            model_name = file["name"]
            download_url = file["downloadUrl"]
            model_type = model_data["model"]["type"].lower()
            basemodel = model_data["baseModel"].replace(" ", "")
            return model_name, download_url, model_type, basemodel


def request_civitai_model_api(model_id: int, version_id: int = None, headers: dict | None = None):
    # Make a request to the CivitAI API to get the model information
    response = requests.get(f"https://civitai.com/api/v1/models/{model_id}", headers=headers, timeout=10)
    response.raise_for_status()  # Raise an error for bad status codes

    model_data = response.json()

    # If version_id is None, use the first version
    if version_id is None:
        version_id = model_data["modelVersions"][0]["id"]

    # Find the version with the specified version_id
    for version in model_data["modelVersions"]:
        if version["id"] == version_id:
            # Get the model name and download URL from the files array
            for file in version["files"]:
                if file.get("primary", False):  # Assuming we want the primary file
                    model_name = file["name"]
                    download_url = file["downloadUrl"]
                    model_type = model_data["type"].lower()
                    basemodel = version["baseModel"].replace(" ", "")
                    return model_name, download_url, model_type, basemodel

    # If the specified version_id is not found, raise an error
    raise ValueError(f"Version ID {version_id} not found for model ID {model_id}")


@app.command(help="Download model file from url")
@tracking.track_command("model")
def download(
    _ctx: typer.Context,
    url: Annotated[
        str,
        typer.Option(help="The URL from which to download the model.", show_default=False),
    ],
    relative_path: Annotated[
        str | None,
        typer.Option(
            help="The relative path from the current workspace to install the model.",
            show_default=True,
        ),
    ] = None,
    filename: Annotated[
        str | None,
        typer.Option(
            help="The filename to save the model.",
            show_default=True,
        ),
    ] = None,
    set_civitai_api_token: Annotated[
        str | None,
        typer.Option(
            "--set-civitai-api-token",
            help="Set the CivitAI API token to use for model downloading.",
            show_default=False,
        ),
    ] = None,
    set_hf_api_token: Annotated[
        str | None,
        typer.Option(
            "--set-hf-api-token",
            help="Set the Hugging Face API token to use for model downloading.",
            show_default=False,
        ),
    ] = None,
    downloader: Annotated[
        str | None,
        typer.Option(
            "--downloader",
            help="Download backend: 'httpx' (default) or 'aria2' (requires aria2 RPC server).",
            show_default=False,
        ),
    ] = None,
    extra_model_paths_config: Annotated[
        list[pathlib.Path] | None,
        typer.Option(
            "--extra-model-paths-config",
            help="Additional extra_model_paths.yaml file(s) to honor. Repeatable.",
            show_default=False,
        ),
    ] = None,
    extra_model_paths: Annotated[
        bool,
        typer.Option(
            "--extra-model-paths/--no-extra-model-paths",
            help="Honor extra_model_paths.yaml from the workspace and any --extra-model-paths-config files.",
            show_default=False,
        ),
    ] = True,
):
    if relative_path is not None:
        relative_path = os.path.expanduser(relative_path)

    extras: list = []
    if extra_model_paths:
        try:
            extras = collect_extra_paths(get_workspace(), extra_model_paths_config or [])
        except yaml.YAMLError as e:
            print(f"[yellow]Warning: extra_model_paths YAML is invalid; ignoring extras ({escape(str(e))})[/yellow]")

    local_filename = None
    headers = None

    civitai_api_token = config_manager.get_or_override(
        constants.CIVITAI_API_TOKEN_ENV_KEY, constants.CIVITAI_API_TOKEN_KEY, set_civitai_api_token
    )
    hf_api_token = config_manager.get_or_override(
        constants.HF_API_TOKEN_ENV_KEY, constants.HF_API_TOKEN_KEY, set_hf_api_token
    )

    resolved_downloader = downloader or config_manager.get(constants.CONFIG_KEY_DEFAULT_DOWNLOADER) or "httpx"

    is_civitai_model_url, is_civitai_api_url, model_id, version_id = check_civitai_url(url)
    is_huggingface_url, repo_id, hf_filename, hf_folder_name, hf_branch_name = check_huggingface_url(url)

    if is_civitai_model_url or is_civitai_api_url:
        headers = {
            "Content-Type": "application/json",
        }
        if civitai_api_token is not None:
            headers["Authorization"] = f"Bearer {civitai_api_token}"

    if is_civitai_model_url:
        local_filename, url, model_type, basemodel = request_civitai_model_api(model_id, version_id, headers)

        model_path = model_path_map.get(model_type)

        if relative_path is None:
            if model_path is None:
                model_path = ui.prompt_input("Enter model type path (e.g. loras, checkpoints, ...)", default="")

            relative_path = _resolve_default_relative_path(model_path, basemodel, extras)
    elif is_civitai_api_url:
        local_filename, url, model_type, basemodel = request_civitai_model_version_api(version_id, headers)

        model_path = model_path_map.get(model_type)

        if relative_path is None:
            if model_path is None:
                model_path = ui.prompt_input("Enter model type path (e.g. loras, checkpoints, ...)", default="")

            relative_path = _resolve_default_relative_path(model_path, basemodel, extras)
    elif is_huggingface_url:
        model_id = "/".join(url.split("/")[-2:])

        local_filename = potentially_strip_param_url(url.split("/")[-1])

        if relative_path is None:
            model_path = ui.prompt_input("Enter model type path (e.g. loras, checkpoints, ...)", default="")
            basemodel = ui.prompt_input("Enter base model (e.g. SD1.5, SDXL, ...)", default="")
            relative_path = _resolve_default_relative_path(model_path, basemodel, extras)
    else:
        print("Model source is unknown")

    if filename is None:
        if local_filename is None:
            local_filename = ui.prompt_input("Enter filename to save model as")
        else:
            local_filename = ui.prompt_input("Enter filename to save model as", default=local_filename)
    else:
        local_filename = filename

    if relative_path is None:
        relative_path = DEFAULT_COMFY_MODEL_PATH

    if local_filename is None:
        raise typer.Exit(code=1)
    if local_filename == "":
        raise DownloadException("Filename cannot be empty")

    local_filepath = get_workspace() / relative_path / local_filename

    if local_filepath.exists():
        print(f"[bold red]File already exists: {local_filepath}[/bold red]")
        return

    start_time = time.monotonic()

    if is_huggingface_url and check_unauthorized(url, headers):
        if hf_api_token is None:
            print(
                f"Unauthorized access to Hugging Face model. Please set the Hugging Face API token using `comfy model download --set-hf-api-token` or via the `{constants.HF_API_TOKEN_ENV_KEY}` environment variable"
            )
            return
        else:
            try:
                import huggingface_hub
            except ImportError:
                print("huggingface_hub not found. Installing...")
                import subprocess

                from comfy_cli.resolve_python import resolve_workspace_python

                python = resolve_workspace_python(str(get_workspace()))
                subprocess.check_call([python, "-m", "pip", "install", "huggingface_hub"])
                import huggingface_hub

            print(f"Downloading model {model_id} from Hugging Face...")
            output_path = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=hf_filename,
                subfolder=hf_folder_name,
                revision=hf_branch_name,
                token=hf_api_token,
                local_dir=get_workspace() / relative_path,
                cache_dir=get_workspace() / relative_path,
            )
            print(f"Model downloaded successfully to: {output_path}")
    else:
        print(f"Start downloading URL: {url} into {local_filepath}")
        try:
            download_file(url, local_filepath, headers, downloader=resolved_downloader)
        except DownloadException as e:
            # escape() so a dynamic error message containing "[/]" or similar
            # rich-markup syntax doesn't trigger MarkupError or get mis-rendered.
            print(f"[bold red]{escape(str(e))}[/bold red]")
            raise typer.Exit(code=1) from None

    elapsed = time.monotonic() - start_time
    print(f"Done in {_format_elapsed(elapsed)}")


@app.command()
@tracking.track_command("model")
def remove(
    ctx: typer.Context,
    relative_path: str = typer.Option(
        DEFAULT_COMFY_MODEL_PATH,
        help="The relative path from the current workspace where the models are stored.",
        show_default=True,
    ),
    model_names: list[str] | None = typer.Option(
        None,
        help="List of model filenames to delete, separated by spaces",
        show_default=False,
    ),
    confirm: bool = typer.Option(
        False,
        help="Confirm for deletion and skip the prompt",
        show_default=False,
    ),
    extra_model_paths_config: list[pathlib.Path] | None = typer.Option(
        None,
        "--extra-model-paths-config",
        help="Additional extra_model_paths.yaml file(s) to honor. Repeatable.",
        show_default=False,
    ),
    extra_model_paths: bool = typer.Option(
        True,
        "--extra-model-paths/--no-extra-model-paths",
        help="Honor extra_model_paths.yaml from the workspace and any --extra-model-paths-config files.",
        show_default=False,
    ),
):
    """Remove one or more downloaded models, either by specifying them directly or through an interactive selection."""
    primary = get_workspace() / relative_path
    extras = _load_extras_safely(extra_model_paths, extra_model_paths_config)
    roots = _enumerate_search_roots(primary, extras)
    scanned = _scan_all_roots(roots)

    if not scanned:
        typer.echo("No models found to remove.")
        return

    resolved_roots: list[pathlib.Path] = []
    for root, _ in roots:
        try:
            resolved_roots.append(root.resolve())
        except OSError:
            continue

    to_delete: list[pathlib.Path] = []
    if model_names:
        missing_models = []
        ambiguous = []
        for name in model_names:
            valid_matches: set[pathlib.Path] = set()
            any_outside = False
            for root, _ in roots:
                try:
                    candidate = (root / name).resolve()
                except OSError:
                    continue
                if any(candidate.is_relative_to(r) for r in resolved_roots):
                    if candidate.is_file():
                        valid_matches.add(candidate)
                else:
                    any_outside = True

            if not valid_matches:
                if any_outside:
                    typer.echo(f"Invalid model path: {name}")
                else:
                    missing_models.append(name)
                continue

            if len(valid_matches) > 1:
                ambiguous.append((name, sorted(valid_matches)))
                continue

            to_delete.append(valid_matches.pop())

        if ambiguous:
            for name, paths in ambiguous:
                typer.echo(f"Ambiguous model name '{name}'; matches multiple paths:")
                for p in paths:
                    typer.echo(f"  {p}")
            typer.echo("Specify a more specific path to disambiguate.")
            if not to_delete:
                return

        if missing_models:
            typer.echo("The following models were not found and cannot be removed: " + ", ".join(missing_models))
            if not to_delete:
                return
    else:
        if len(roots) == 1:
            single_root = roots[0][0]
            labels_to_paths = {str(file.relative_to(single_root)): file for file, _, _ in scanned}
        else:
            labels_to_paths = {str(file): file for file, _, _ in scanned}

        selections = ui.prompt_multi_select("Select models to delete:", list(labels_to_paths.keys()))
        if not selections:
            typer.echo("No models selected for deletion.")
            return
        to_delete = [labels_to_paths[sel] for sel in selections]

    if to_delete and (
        confirm or ui.prompt_confirm_action("Are you sure you want to delete the selected files?", False)
    ):
        for model_path in to_delete:
            model_path.unlink()
            typer.echo(f"Deleted: {model_path}")
    else:
        typer.echo("Deletion canceled.")


def list_models(path: pathlib.Path) -> list[pathlib.Path]:
    """List all model files recursively in the specified directory."""
    if not path.is_dir():
        return []
    return sorted(f for f in path.rglob("*") if f.is_file())


def _load_extras_safely(use_extras: bool, extra_configs: list[pathlib.Path] | None) -> list:
    if not use_extras:
        return []
    try:
        return collect_extra_paths(get_workspace(), extra_configs or [])
    except yaml.YAMLError as e:
        print(f"[yellow]Warning: extra_model_paths YAML is invalid; ignoring extras ({escape(str(e))})[/yellow]")
        return []


def _enumerate_search_roots(primary_root: pathlib.Path, extras: list) -> list[tuple[pathlib.Path, str | None]]:
    """Return ``(root, category)`` pairs to scan, longest-first.

    The primary root carries ``category=None`` so list rendering preserves
    today's "category from path" behavior. Extras roots carry their canonical
    category name for the Type-column prefix. Roots are deduplicated by
    realpath; unresolvable roots (e.g., circular symlinks) are skipped with
    a warning. Sorting longest-first ensures a file under nested roots is
    assigned to the most specific one.
    """
    candidates: list[tuple[pathlib.Path, str | None]] = [(primary_root, None)]
    for ep in extras:
        candidates.append((ep.path, ep.category))

    seen_resolved: set[pathlib.Path] = set()
    unique: list[tuple[pathlib.Path, str | None]] = []
    for root, category in candidates:
        try:
            resolved = root.resolve()
        except OSError as e:
            print(f"[yellow]Warning: skipping {root}: {e}[/yellow]")
            continue
        if resolved in seen_resolved:
            continue
        seen_resolved.add(resolved)
        unique.append((root, category))

    unique.sort(key=lambda rc: len(rc[0].parts), reverse=True)
    return unique


def _scan_all_roots(
    roots: list[tuple[pathlib.Path, str | None]],
) -> list[tuple[pathlib.Path, pathlib.Path, str | None]]:
    """Return ``(file, root, category)`` tuples, each file assigned to its
    deepest containing root. Output is sorted by file path."""
    seen_files: set[pathlib.Path] = set()
    result: list[tuple[pathlib.Path, pathlib.Path, str | None]] = []
    for root, category in roots:
        for file in list_models(root):
            try:
                resolved = file.resolve()
            except OSError:
                continue
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            result.append((file, root, category))
    result.sort(key=lambda x: x[0])
    return result


def _format_type_column(file: pathlib.Path, root: pathlib.Path, category: str | None) -> str:
    """Compute Type column text. For extras roots the canonical category is
    prepended so output is consistent with the workspace listing where the
    category is implicit in the on-disk subdir."""
    rel = file.relative_to(root)
    parent = str(rel.parent) if len(rel.parts) > 1 else ""
    if category is None:
        return parent
    if not parent or parent == ".":
        return category
    return f"{category}/{parent}"


@app.command("list")
@tracking.track_command("model")
def list_command(
    ctx: typer.Context,
    relative_path: str = typer.Option(
        DEFAULT_COMFY_MODEL_PATH,
        help="The relative path from the current workspace where the models are stored.",
        show_default=True,
    ),
    extra_model_paths_config: list[pathlib.Path] | None = typer.Option(
        None,
        "--extra-model-paths-config",
        help="Additional extra_model_paths.yaml file(s) to honor. Repeatable.",
        show_default=False,
    ),
    extra_model_paths: bool = typer.Option(
        True,
        "--extra-model-paths/--no-extra-model-paths",
        help="Honor extra_model_paths.yaml from the workspace and any --extra-model-paths-config files.",
        show_default=False,
    ),
):
    """Display a list of all models currently downloaded in a table format."""
    primary = get_workspace() / relative_path
    extras = _load_extras_safely(extra_model_paths, extra_model_paths_config)
    roots = _enumerate_search_roots(primary, extras)
    scanned = _scan_all_roots(roots)

    if not scanned:
        typer.echo("No models found.")
        return

    show_source = len({r for _, r, _ in scanned}) > 1
    data = []
    for file, root, category in scanned:
        type_str = _format_type_column(file, root, category)
        size_str = f"{file.stat().st_size // 1024} KB"
        if show_source:
            data.append((file.name, type_str, size_str, str(root)))
        else:
            data.append((file.name, type_str, size_str))

    columns = ["Model Name", "Type", "Size"]
    if show_source:
        columns.append("Source")
    ui.display_table(data, columns)
