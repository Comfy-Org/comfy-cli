import os

import tomlkit.exceptions
from comfy_cli.registry.types import (
    PyProjectConfig,
    ProjectConfig,
    URLs,
    Model,
    ComfyConfig,
)
import tomlkit
import subprocess


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

    tool = tomlkit.table()
    comfy = tomlkit.table()
    comfy["PublisherId"] = ""
    comfy["DisplayName"] = ""
    comfy["Icon"] = ""

    # Add the default model
    models = tomlkit.array()
    model = tomlkit.inline_table()
    model["location"] = "/checkpoints/model.safetensor"
    model["model_url"] = "https://example.com/model.zip"
    models.append(model)
    comfy["Models"] = models

    tool.add("comfy", comfy)
    document.add("tool", tool)

    # Write the TOML document to a file
    try:
        with open("pyproject.toml", "w") as toml_file:
            toml_file.write(tomlkit.dumps(document))
    except IOError as e:
        raise Exception(f"Failed to write 'pyproject.toml': {str(e)}")


def initialize_project_config():
    create_comfynode_config()

    with open("pyproject.toml", "r") as file:
        document = tomlkit.parse(file.read())

    # Get the current git remote URL
    try:
        git_remote_url = (
            subprocess.check_output(["git", "remote", "get-url", "origin"])
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        raise Exception(
            "Could not retrieve Git remote URL. Are you in a Git repository?"
        )

    # Convert SSH URL to HTTPS if needed
    if git_remote_url.startswith("git@github.com:"):
        git_remote_url = git_remote_url.replace(
            "git@github.com:", "https://github.com/"
        )

    # Ensure the URL ends with `.git` and remove it to obtain the plain URL
    repo_name = git_remote_url.split("/")[-1].replace(".git", "")
    git_remote_url = git_remote_url.replace(".git", "")

    project = document.get("project", tomlkit.table())
    urls = project.get("urls", tomlkit.table())
    urls["Repository"] = git_remote_url
    project["urls"] = urls
    project["name"] = repo_name.lower()
    project["description"] = ""
    project["version"] = "1.0.0"
    project["license"] = "LICENSE"

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
        raise IOError(f"Failed to write 'pyproject.toml': {str(e)}")


def extract_node_configuration(
    path: str = os.path.join(os.getcwd(), "pyproject.toml"),
) -> PyProjectConfig:
    import tomlkit

    with open(path, "r") as file:
        data = tomlkit.load(file)

    project_data = data.get("project", {})
    urls_data = project_data.get("urls", {})
    comfy_data = data.get("tool", {}).get("comfy", {})

    project = ProjectConfig(
        name=project_data.get("name", ""),
        description=project_data.get("description", ""),
        version=project_data.get("version", ""),
        requires_python=project_data.get("requires-pyton", ""),
        dependencies=project_data.get("dependencies", []),
        license=project_data.get("license", ""),
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
        models=[
            Model(location=m["location"], model_url=m["model_url"])
            for m in comfy_data.get("Models", [])
        ],
    )

    return PyProjectConfig(project=project, tool_comfy=comfy)
