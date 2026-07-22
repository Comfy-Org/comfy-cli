import contextlib
import io
import json
import unittest
from unittest.mock import MagicMock, patch

from comfy_cli.registry import PyProjectConfig
from comfy_cli.registry.api import RegistryAPI
from comfy_cli.registry.types import ComfyConfig, License, ProjectConfig, URLs


class TestRegistryAPI(unittest.TestCase):
    def setUp(self):
        self.registry_api = RegistryAPI()
        self.node_config = PyProjectConfig(
            project=ProjectConfig(
                name="test_node",
                description="A test node",
                version="0.1.0",
                requires_python=">= 3.9",
                dependencies=["dep1", "dep2"],
                license=License(file="LICENSE"),
                urls=URLs(repository="https://github.com/test/test_node"),
            ),
            tool_comfy=ComfyConfig(
                publisher_id="123",
                display_name="Test Node",
                icon="https://example.com/icon.png",
            ),
        )
        self.token = "dummy_token"

    @patch("os.getenv")
    def test_determine_base_url_dev(self, mock_getenv):
        mock_getenv.return_value = "dev"
        self.assertEqual(self.registry_api.determine_base_url(), "http://localhost:8080")

    @patch("os.getenv")
    def test_determine_base_url_prod(self, mock_getenv):
        mock_getenv.return_value = "prod"
        self.assertEqual(self.registry_api.determine_base_url(), "https://api.comfy.org")

    @patch("requests.post")
    def test_publish_node_version_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "node_version": {
                "id": "test_node",
                "version": "0.1.0",
                "changelog": "",
                "dependencies": ["dep1", "dep2"],
                "deprecated": False,
                "downloadUrl": "https://example.com/download",
            },
            "signedUrl": "https://example.com/signed",
        }
        mock_post.return_value = mock_response

        response = self.registry_api.publish_node_version(self.node_config, self.token)
        self.assertEqual(response.node_version.id, "test_node")
        self.assertEqual(response.node_version.version, "0.1.0")
        self.assertEqual(response.signedUrl, "https://example.com/signed")

    @patch("requests.post")
    def test_publish_node_version_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        with self.assertRaises(Exception) as context:
            self.registry_api.publish_node_version(self.node_config, self.token)
        self.assertIn("Failed to publish node version", str(context.exception))

    def _mock_publish_response(self, changelog=""):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "node_version": {
                "id": "7f2d0a4e-0000-4000-8000-000000000001",
                "version": "0.1.0",
                "changelog": changelog,
                "dependencies": ["dep1", "dep2"],
                "deprecated": False,
                "downloadUrl": "https://example.com/download",
            },
            "signedUrl": "https://example.com/signed",
        }
        return mock_response

    @patch("requests.post")
    def test_publish_node_version_sends_changelog_verbatim(self, mock_post):
        changelog = "## 0.1.0\n\n- Fixed flux capacitor ⚡\n- Added docs"
        mock_post.return_value = self._mock_publish_response(changelog=changelog)

        response = self.registry_api.publish_node_version(self.node_config, self.token, changelog=changelog)

        sent_body = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(sent_body["node_version"]["changelog"], changelog)
        self.assertEqual(response.node_version.changelog, changelog)

    @patch("requests.post")
    def test_publish_node_version_omits_changelog_when_not_given(self, mock_post):
        mock_post.return_value = self._mock_publish_response()

        self.registry_api.publish_node_version(self.node_config, self.token)

        sent_body = json.loads(mock_post.call_args[1]["data"])
        self.assertNotIn("changelog", sent_body["node_version"])

    @patch("requests.post")
    def test_publish_node_version_omits_changelog_when_empty(self, mock_post):
        mock_post.return_value = self._mock_publish_response()

        self.registry_api.publish_node_version(self.node_config, self.token, changelog="")

        sent_body = json.loads(mock_post.call_args[1]["data"])
        self.assertNotIn("changelog", sent_body["node_version"])

    @patch("requests.post")
    def test_publish_node_version_does_not_print_token(self, mock_post):
        mock_post.return_value = self._mock_publish_response()

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.registry_api.publish_node_version(self.node_config, self.token, changelog="notes")

        self.assertNotIn(self.token, captured.getvalue())

    @patch("requests.get")
    def test_list_all_nodes_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "nodes": [
                {
                    "id": "node1",
                    "name": "Node 1",
                    "description": "First node",
                    "author": "Author 1",
                    "license": "MIT",
                    "icon": "https://example.com/icon1.png",
                    "repository": "https://github.com/test/node1",
                    "tags": ["tag1", "tag2"],
                    "latest_version": {
                        "id": "node1",
                        "version": "1.0.0",
                        "changelog": "",
                        "dependencies": ["dep1"],
                        "deprecated": False,
                        "downloadUrl": "https://example.com/download1",
                    },
                }
            ]
        }
        mock_get.return_value = mock_response

        nodes = self.registry_api.list_all_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].id, "node1")
        self.assertEqual(nodes[0].name, "Node 1")

    @patch("requests.get")
    def test_list_all_nodes_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as context:
            self.registry_api.list_all_nodes()
        self.assertIn("Failed to retrieve nodes", str(context.exception))

    @patch("requests.get")
    def test_install_node_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "node1",
            "version": "1.0.0",
            "changelog": "",
            "dependencies": ["dep1"],
            "deprecated": False,
            "downloadUrl": "https://example.com/download1",
        }
        mock_get.return_value = mock_response

        node_version = self.registry_api.install_node("node1")
        self.assertEqual(node_version.id, "node1")
        self.assertEqual(node_version.version, "1.0.0")

    @patch("requests.get")
    def test_install_node_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as context:
            self.registry_api.install_node("node1")
        self.assertIn("Failed to install node", str(context.exception))

    @patch("requests.get")
    def test_get_node_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "node1",
            "name": "Node One",
            "description": "A node",
            "latest_version": {
                "id": "nv1",
                "version": "1.2.3",
            },
        }
        mock_get.return_value = mock_response

        node = self.registry_api.get_node("node1")
        self.assertEqual(node.id, "node1")
        self.assertEqual(node.latest_version.version, "1.2.3")
        # Read-only endpoint — must hit /nodes/{id}, never /nodes/{id}/install.
        called_url = mock_get.call_args[0][0]
        self.assertTrue(called_url.endswith("/nodes/node1"))

    @patch("requests.get")
    def test_get_node_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as context:
            self.registry_api.get_node("node1")
        self.assertIn("Failed to retrieve node", str(context.exception))
