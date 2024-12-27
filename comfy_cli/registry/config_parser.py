import os
import subprocess
from typing import Optional

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


def create_comfynode_config():
    # Create the initial structure of the TOML document
    document = tomlkit.document()

    project = tomlkit.table()
    project["name"] = ""
    project["description"] = ""
    project["version"] = "1.0.0"
    project["dependencies"] = tomlkit.aot()
    project["license"] = "LICENSE"

    urls = tomlkit.table()
    urls["Repository"] = ""

    project.add("urls", urls)
    document.add("project", project)

    # Create the tool table
    tool = tomlkit.table()
    document.add(tomlkit.comment(" Used by Comfy Registry https://comfyregistry.org"))

    comfy = tomlkit.table()
    comfy["PublisherId"] = ""
    comfy["DisplayName"] = "ComfyUI-AIT"
    comfy["Icon"] = ""

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
    except IOError as e:
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


def initialize_project_config():
    create_comfynode_config()

    with open("pyproject.toml", "r") as file:
        document = tomlkit.parse(file.read())

    # Get the current git remote URL
    try:
        git_remote_url = subprocess.check_output(["git", "remote", "get-url", "origin"]).decode().strip()
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
    project["urls"] = urls
    project["name"] = sanitize_node_name(repo_name)
    project["description"] = ""
    project["version"] = "1.0.0"

    # Update the license field to comply with pyproject.toml spec
    license_table = tomlkit.inline_table()
    license_table["file"] = "LICENSE"
    project["license"] = license_table

    tool = document.get("tool", tomlkit.table())
    comfy = tool.get("comfy", tomlkit.table())
    comfy["DisplayName"] = repo_name
    tool["comfy"] = comfy
    document["tool"] = tool

    # Handle dependencies
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as req_file:
            dependencies = [line.strip() for line in req_file if line.strip()]
        project["dependencies"] = dependencies
    else:
        print("Warning: 'requirements.txt' not found. No dependencies will be added.")

    # Write the updated config to a new file in the current directory
    try:
        with open("pyproject.toml", "w") as toml_file:
            toml_file.write(tomlkit.dumps(document))
        print("pyproject.toml has been created successfully in the current directory.")
    except IOError as e:
        raise IOError("Failed to write 'pyproject.toml'") from e


def extract_node_configuration(
    path: str = os.path.join(os.getcwd(), "pyproject.toml"),
) -> Optional[PyProjectConfig]:
    if not os.path.isfile(path):
        ui.display_error_message("No pyproject.toml file found in the current directory.")
        return None

    with open(path, "r") as file:
        data = tomlkit.load(file)

    project_data = data.get("project", {})
    urls_data = project_data.get("urls", {})
    comfy_data = data.get("tool", {}).get("comfy", {})

    license_data = project_data.get("license", {})
    if isinstance(license_data, str):
        license = License(text=license_data)
        typer.echo(
            'Warning: License should be in one of these two formats: license = {file = "LICENSE"} OR license = {text = "MIT License"}. Please check the documentation: https://docs.comfy.org/registry/specifications.'
        )
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

    project = ProjectConfig(
        name=project_data.get("name", ""),
        description=project_data.get("description", ""),
        version=project_data.get("version", ""),
        requires_python=project_data.get("requires-python", ""),
        dependencies=project_data.get("dependencies", []),
        license=license,
        urls=URLs(
            homepage=urls_data.get("Homepage", ""),
            documentation=urls_data.get("Documentation", ""),
            repository=urls_data.get("Repository", ""),
            issues=urls_data.get("Issues", ""),
        ),
    )

    comfy = ComfyConfig(
        publisher_id=comfy_data.get("PublisherId", ""),
        display_name=comfy_data.get("DisplayName", ""),
        icon=comfy_data.get("Icon", ""),
        models=[Model(location=m["location"], model_url=m["model_url"]) for m in comfy_data.get("Models", [])],
    )

    return PyProjectConfig(project=project, tool_comfy=comfy)
