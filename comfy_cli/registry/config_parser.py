import os
from comfy_cli.registry.types import (
    PyProjectConfig,
    ProjectConfig,
    URLs,
    Model,
    ComfyConfig,
)


def extract_node_configuration(
    path: str = os.path.join(os.getcwd(), "comfynode.toml"),
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
