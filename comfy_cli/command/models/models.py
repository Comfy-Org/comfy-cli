import os
import pathlib
import sys
from typing import List, Optional, Tuple
from urllib.parse import unquote, urlparse

import requests
import typer
from rich import print
from typing_extensions import Annotated

from comfy_cli import constants, tracking, ui
from comfy_cli.config_manager import ConfigManager
from comfy_cli.constants import DEFAULT_COMFY_MODEL_PATH
from comfy_cli.file_utils import DownloadException, check_unauthorized, download_file
from comfy_cli.workspace_manager import WorkspaceManager

app = typer.Typer()

workspace_manager = WorkspaceManager()
config_manager = ConfigManager()


model_path_map = {
    "lora": "loras",
    "hypernetwork": "hypernetworks",
    "checkpoint": "checkpoints",
    "textualinversion": "embeddings",
    "controlnet": "controlnet",
}


def get_workspace() -> pathlib.Path:
    return pathlib.Path(workspace_manager.workspace_path)


def potentially_strip_param_url(path_name: str) -> str:
    path_name = path_name.split("?")[0]
    return path_name


def check_huggingface_url(url: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str]]:
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


def check_civitai_url(url: str) -> Tuple[bool, bool, int, int]:
    """
    Returns:
        is_civitai_model_url: True if the url is a civitai model url
        is_civitai_api_url: True if the url is a civitai api url
        model_id: The model id or None if it's api url
        version_id: The version id or None if it doesn't have version id info
    """
    prefix = "civitai.com"
    try:
        if prefix in url:
            # URL is civitai api download url: https://civitai.com/api/download/models/12345
            if "civitai.com/api/download" in url:
                # This is a direct download link
                version_id = url.strip("/").split("/")[-1]
                return False, True, None, int(version_id)

            # URL is civitai web url (e.g.
            #   - https://civitai.com/models/43331
            #   - https://civitai.com/models/43331/majicmix-realistic
            subpath = url[url.find(prefix) + len(prefix) :].strip("/")
            url_parts = subpath.split("?")
            if len(url_parts) > 1:
                model_id = url_parts[0].split("/")[1]
                version_id = url_parts[1].split("=")[1]
                return True, False, int(model_id), int(version_id)
            else:
                model_id = subpath.split("/")[1]
                return True, False, int(model_id), None
    except (ValueError, IndexError):
        print("Error parsing CivitAI model URL")

    return False, False, None, None


def request_civitai_model_version_api(version_id: int, headers: Optional[dict] = None):
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


def request_civitai_model_api(model_id: int, version_id: int = None, headers: Optional[dict] = None):
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
        Optional[str],
        typer.Option(
            help="The relative path from the current workspace to install the model.",
            show_default=True,
        ),
    ] = None,
    filename: Annotated[
        Optional[str],
        typer.Option(
            help="The filename to save the model.",
            show_default=True,
        ),
    ] = None,
    set_civitai_api_token: Annotated[
        Optional[str],
        typer.Option(
            "--set-civitai-api-token",
            help="Set the CivitAI API token to use for model downloading.",
            show_default=False,
        ),
    ] = None,
    set_hf_api_token: Annotated[
        Optional[str],
        typer.Option(
            "--set-hf-api-token",
            help="Set the Hugging Face API token to use for model downloading.",
            show_default=False,
        ),
    ] = None,
):
    if relative_path is not None:
        relative_path = os.path.expanduser(relative_path)

    local_filename = None
    headers = None

    civitai_api_token = config_manager.get_or_override(
        constants.CIVITAI_API_TOKEN_ENV_KEY, constants.CIVITAI_API_TOKEN_KEY, set_civitai_api_token
    )
    hf_api_token = config_manager.get_or_override(
        constants.HF_API_TOKEN_ENV_KEY, constants.HF_API_TOKEN_KEY, set_hf_api_token
    )

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

            relative_path = os.path.join(DEFAULT_COMFY_MODEL_PATH, model_path, basemodel)
    elif is_civitai_api_url:
        local_filename, url, model_type, basemodel = request_civitai_model_version_api(version_id, headers)

        model_path = model_path_map.get(model_type)

        if relative_path is None:
            if model_path is None:
                model_path = ui.prompt_input("Enter model type path (e.g. loras, checkpoints, ...)", default="")

            relative_path = os.path.join(DEFAULT_COMFY_MODEL_PATH, model_path, basemodel)
    elif is_huggingface_url:
        model_id = "/".join(url.split("/")[-2:])

        local_filename = potentially_strip_param_url(url.split("/")[-1])

        if relative_path is None:
            model_path = ui.prompt_input("Enter model type path (e.g. loras, checkpoints, ...)", default="")
            basemodel = ui.prompt_input("Enter base model (e.g. SD1.5, SDXL, ...)", default="")
            relative_path = os.path.join(DEFAULT_COMFY_MODEL_PATH, model_path, basemodel)
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

                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
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
        download_file(url, local_filepath, headers)


@app.command()
@tracking.track_command("model")
def remove(
    ctx: typer.Context,
    relative_path: str = typer.Option(
        DEFAULT_COMFY_MODEL_PATH,
        help="The relative path from the current workspace where the models are stored.",
        show_default=True,
    ),
    model_names: Optional[List[str]] = typer.Option(
        None,
        help="List of model filenames to delete, separated by spaces",
        show_default=False,
    ),
    confirm: bool = typer.Option(
        False,
        help="Confirm for deletion and skip the prompt",
        show_default=False,
    ),
):
    """Remove one or more downloaded models, either by specifying them directly or through an interactive selection."""
    model_dir = get_workspace() / relative_path
    available_models = list_models(model_dir)

    if not available_models:
        typer.echo("No models found to remove.")
        return

    to_delete = []
    # Scenario #1: User provided model names to delete
    if model_names:
        # Validate and filter models to delete based on provided names
        missing_models = []
        for name in model_names:
            model_path = model_dir / name
            if model_path.exists():
                to_delete.append(model_path)
            else:
                missing_models.append(name)

        if missing_models:
            typer.echo("The following models were not found and cannot be removed: " + ", ".join(missing_models))
            if not to_delete:
                return  # Exit if no valid models were found

    # Scenario #2: User did not provide model names, prompt for selection
    else:
        selections = ui.prompt_multi_select("Select models to delete:", [model.name for model in available_models])
        if not selections:
            typer.echo("No models selected for deletion.")
            return
        to_delete = [model_dir / selection for selection in selections]

    # Confirm deletion
    if to_delete and (
        confirm or ui.prompt_confirm_action("Are you sure you want to delete the selected files?", False)
    ):
        for model_path in to_delete:
            model_path.unlink()
            typer.echo(f"Deleted: {model_path}")
    else:
        typer.echo("Deletion canceled.")


@app.command()
@tracking.track_command("model")
def list(
    ctx: typer.Context,
    relative_path: str = typer.Option(
        DEFAULT_COMFY_MODEL_PATH,
        help="The relative path from the current workspace where the models are stored.",
        show_default=True,
    ),
):
    """Display a list of all models currently downloaded in a table format."""
    model_dir = get_workspace() / relative_path
    models = list_models(model_dir)

    if not models:
        typer.echo("No models found.")
        return

    # Prepare data for table display
    data = [(model.name, f"{model.stat().st_size // 1024} KB") for model in models]
    column_names = ["Model Name", "Size"]
    ui.display_table(data, column_names)


def list_models(path: pathlib.Path) -> list:
    """List all models in the specified directory."""
    return [file for file in path.iterdir() if file.is_file()]
