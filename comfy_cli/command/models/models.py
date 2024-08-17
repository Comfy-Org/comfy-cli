import os
import pathlib
from typing import List, Optional, Tuple

import requests
import typer
from rich import print
from typing_extensions import Annotated

from comfy_cli import constants, tracking, ui
from comfy_cli.config_manager import ConfigManager
from comfy_cli.constants import DEFAULT_COMFY_MODEL_PATH
from comfy_cli.file_utils import DownloadException, download_file
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


# Convert relative path to absolute path based on the current working
# directory
def check_huggingface_url(url: str) -> bool:
    return "huggingface.co" in url


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
        print("Error parsing Civitai model URL")

    return False, False, None, None


def request_civitai_model_version_api(version_id: int, headers: Optional[dict] = None):
    # Make a request to the Civitai API to get the model information
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
    # Make a request to the Civitai API to get the model information
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
        typer.Option(help="The URL from which to download the model", show_default=False),
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
            help="Set the CivitAI API token to use for model listing.",
            show_default=False,
        ),
    ] = None,
):
    if relative_path is not None:
        relative_path = os.path.expanduser(relative_path)

    local_filename = None
    headers = None
    civitai_api_token = None

    if set_civitai_api_token is not None:
        config_manager.set(constants.CIVITAI_API_TOKEN_KEY, set_civitai_api_token)
        civitai_api_token = set_civitai_api_token

    else:
        civitai_api_token = config_manager.get(constants.CIVITAI_API_TOKEN_KEY)

    if civitai_api_token is not None:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {civitai_api_token}",
        }

    is_civitai_model_url, is_civitai_api_url, model_id, version_id = check_civitai_url(url)

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
    elif check_huggingface_url(url):
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

    # Check if the file already exists
    if local_filepath.exists():
        print(f"[bold red]File already exists: {local_filepath}[/bold red]")
        return

    # File does not exist, proceed with download
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
