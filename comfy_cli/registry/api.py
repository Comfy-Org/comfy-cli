import json
import logging
import os

import requests

# Reduced global imports from comfy_cli.registry
from comfy_cli.registry.types import (
    License,
    Node,
    NodeVersion,
    PublishNodeVersionResponse,
    PyProjectConfig,
)


class RegistryAPI:
    def __init__(self):
        self.base_url = self.determine_base_url()

    def determine_base_url(self):
        env = os.getenv("ENVIRONMENT")
        if env == "dev":
            return "http://localhost:8080"
        elif env == "staging":
            return "https://stagingapi.comfy.org"
        else:
            return "https://api.comfy.org"

    def publish_node_version(self, node_config: PyProjectConfig, token) -> PublishNodeVersionResponse:
        """
        Publishes a new version of a node.

        Args:
          node_config (PyProjectConfig): The node configuration.
          token (str): The token to authenticate with the API server.

        Returns:
        PublishNodeVersionResponse: The response object from the API server.
        """
        # Local import to prevent circular dependency
        if not node_config.tool_comfy.publisher_id:
            raise Exception("Publisher ID is required in pyproject.toml to publish a node version")

        if not node_config.project.name:
            raise Exception("Project name is required in pyproject.toml to publish a node version")
        license_json = serialize_license(node_config.project.license)
        request_body = {
            "personal_access_token": token,
            "node": {
                "id": node_config.project.name,
                "description": node_config.project.description,
                "icon": node_config.tool_comfy.icon,
                "name": node_config.tool_comfy.display_name,
                "license": license_json,
                "repository": node_config.project.urls.repository,
            },
            "node_version": {
                "version": node_config.project.version,
                "dependencies": node_config.project.dependencies,
            },
        }
        print(request_body)
        url = f"{self.base_url}/publishers/{node_config.tool_comfy.publisher_id}/nodes/{node_config.project.name}/versions"
        headers = {"Content-Type": "application/json"}
        body = request_body

        response = requests.post(url, headers=headers, data=json.dumps(body))

        if response.status_code == 201:
            data = response.json()
            return PublishNodeVersionResponse(
                node_version=map_node_version(data["node_version"]),
                signedUrl=data["signedUrl"],
            )
        else:
            raise Exception(f"Failed to publish node version: {response.status_code} {response.text}")

    def list_all_nodes(self):
        """
        Retrieves a list of all nodes and maps them to Node dataclass instances.

        Returns:
          list: A list of Node instances.
        """
        url = f"{self.base_url}/nodes"
        response = requests.get(url)
        if response.status_code == 200:
            raw_nodes = response.json()["nodes"]
            mapped_nodes = [map_node_to_node_class(node) for node in raw_nodes]
            return mapped_nodes
        else:
            raise Exception(f"Failed to retrieve nodes: {response.status_code} - {response.text}")

    def install_node(self, node_id, version=None):
        """
        Retrieves the node version for installation.

        Args:
          node_id (str): The unique identifier of the node.
          version (str, optional): Specific version of the node to retrieve. If omitted, the latest version is returned.

        Returns:
          NodeVersion: Node version data or error message.
        """
        if version is None:
            url = f"{self.base_url}/nodes/{node_id}/install"
        else:
            url = f"{self.base_url}/nodes/{node_id}/install?version={version}"

        response = requests.get(url)
        if response.status_code == 200:
            # Convert the API response to a NodeVersion object
            logging.debug(f"RegistryAPI install_node response: {response.json()}")
            return map_node_version(response.json())
        else:
            raise Exception(f"Failed to install node: {response.status_code} - {response.text}")


def map_node_version(api_node_version):
    """
    Maps node version data from API response to NodeVersion dataclass.

    Args:
        api_data (dict): The 'node_version' part of the API response.

    Returns:
        NodeVersion: An instance of NodeVersion dataclass populated with data from the API.
    """
    return NodeVersion(
        changelog=api_node_version.get("changelog", ""),  # Provide a default value if 'changelog' is missing
        dependencies=api_node_version.get(
            "dependencies", []
        ),  # Provide a default empty list if 'dependencies' is missing
        deprecated=api_node_version.get("deprecated", False),  # Assume False if 'deprecated' is not specified
        id=api_node_version["id"],  # 'id' should be mandatory; raise KeyError if missing
        version=api_node_version["version"],  # 'version' should be mandatory; raise KeyError if missing
        download_url=api_node_version.get("downloadUrl", ""),  # Provide a default value if 'downloadUrl' is missing
    )


def map_node_to_node_class(api_node_data):
    """
    Maps node data from API response to Node dataclass.

    Args:
        api_node_data (dict): The node data from the API.

    Returns:
        Node: An instance of Node dataclass populated with API data.
    """
    return Node(
        id=api_node_data["id"],
        name=api_node_data["name"],
        description=api_node_data["description"],
        author=api_node_data.get("author"),
        license=api_node_data.get("license"),
        icon=api_node_data.get("icon"),
        repository=api_node_data.get("repository"),
        tags=api_node_data.get("tags", []),
        latest_version=(
            map_node_version(api_node_data["latest_version"]) if "latest_version" in api_node_data else None
        ),
    )


def serialize_license(license: License) -> str:
    if license.file:
        return json.dumps({"file": license.file})
    if license.text:
        return json.dumps({"text": license.text})
    return "{}"
