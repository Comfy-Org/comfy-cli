import os
import re
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
    document.add(tomlkit.comment(" Used by Comfy Registry https://registry.comfy.org"))

    comfy = tomlkit.table()
    comfy["PublisherId"] = ""
    comfy["DisplayName"] = "ComfyUI-AIT"
    comfy["Icon"] = ""
    comfy["includes"] = tomlkit.array()

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

    supported_comfyui_version = data.get("tool", {}).get("comfy", {}).get("requires-comfyui", "")

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
    )

    return PyProjectConfig(project=project, tool_comfy=comfy)
