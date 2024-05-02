import requests
import json
from comfy_cli import constants
from comfy_cli.registry.types import NodeConfiguration


def publish_node_version(node_config: NodeConfiguration, token: str):
    """
    Publishes a new version of a node.

    Args:
    node_config (NodeConfiguration): The node configuration.
    token (str): Personal access token for authentication.

    Returns:
    dict: JSON response from the API server.
    """
    url = f"{constants.COMFY_REGISTRY_URL_ROOT}/publishers/{node_config.publisher_id}/nodes/{node_config.node_id}/versions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "personal_access_token": token,
        "node": {
            "id": node_config.node_id,
            "description": node_config.description,
            "name": node_config.display_name,
            "author": node_config.author,
            "license": node_config.license,
            "tags": node_config.tags,
        },
        "node_version": {
            "version": node_config.version,
        },
    }

    response = requests.post(url, headers=headers, data=json.dumps(body))

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to publish node version: {response.status_code} {response.text}"
        )
