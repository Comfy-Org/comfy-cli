import os

import requests
import json
from comfy_cli.registry.types import (
  PyProjectConfig,
  PublishNodeVersionResponse,
  NodeVersion,
)


class RegistryAPI:
  def __init__(self):
    self.base_url = self.determine_base_url()

  def determine_base_url(self):
    if os.getenv("ENVIRONMENT") == "dev":
      return "http://localhost:8080"
    else:
      return "https://api-frontend-dev-qod3oz2v2q-uc.a.run.app"

  def publish_node_version(self, node_config: PyProjectConfig, token: str) -> PublishNodeVersionResponse:
    """
    Publishes a new version of a node.

    Args:
      node_config (PyProjectConfig): The node configuration.
      token (str): The token to authenticate with the API server.

    Returns:
    PublishNodeVersionResponse: The response object from the API server.
    """
    url = f"{self.base_url}/publishers/{node_config.tool_comfy.publisher_id}/nodes/{node_config.project.name}/versions"
    headers = {"Content-Type": "application/json"}
    body = {
      "personal_access_token": token,
      "node": {
        "id": node_config.project.name,
        "description": node_config.project.description,
        "name": node_config.tool_comfy.display_name,
        "license": node_config.project.license,
        "repository": node_config.project.urls.repository,
      },
      "node_version": {
        "version": node_config.project.version,
        "dependencies": node_config.project.dependencies,
      },
    }

    response = requests.post(url, headers=headers, data=json.dumps(body))

    if response.status_code == 201:
      data = response.json()
      node_version = NodeVersion(
        changelog=data["node_version"]["changelog"],
        dependencies=data["node_version"]["dependencies"],
        deprecated=data["node_version"]["deprecated"],
        id=data["node_version"]["id"],
        version=data["node_version"]["version"],
      )
      return PublishNodeVersionResponse(node_version=node_version, signedUrl=data["signedUrl"])
    else:
      raise Exception(f"Failed to publish node version: {response.status_code} {response.text}")


def upload_file_to_signed_url(signed_url: str, file_path: str):
  try:
    with open(file_path, "rb") as f:
      headers = {"Content-Type": "application/gzip"}
      response = requests.put(signed_url, data=f, headers=headers)

      # Simple success check
      if response.status_code == 200:
        print("Upload successful.")
      else:
        # Print a generic error message with status code and response text
        print(
          f"Upload failed with status code: {response.status_code}. Error: {response.text}"
        )

  except requests.exceptions.RequestException as e:
    # Print error related to the HTTP request
    print(f"An error occurred during the upload: {str(e)}")
  except FileNotFoundError:
    # Print file not found error
    print(f"Error: The file {file_path} does not exist.")
