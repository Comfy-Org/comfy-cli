from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests
import semver

from comfy_cli.command.install import (
    GithubRelease,
    fetch_github_releases,
    parse_releases,
    select_version,
    validate_version,
)


def test_validate_version_nightly():
    assert validate_version("nightly") == "nightly"
    assert validate_version("NIGHTLY") == "nightly"


def test_validate_version_latest():
    assert validate_version("latest") == "latest"
    assert validate_version("LATEST") == "latest"


def test_validate_version_valid_semver():
    assert validate_version("1.2.3") == "1.2.3"
    assert validate_version("v1.2.3") == "1.2.3"
    assert validate_version("1.2.3-alpha") == "1.2.3-alpha"


def test_validate_version_invalid():
    with pytest.raises(ValueError):
        validate_version("invalid_version")


def test_validate_version_empty():
    with pytest.raises(ValueError):
        validate_version("")


# Tests for fetch_github_releases function
@patch("requests.get")
def test_fetch_releases_success(mock_get):
    # Mock the response
    mock_response = MagicMock()
    mock_response.json.return_value = [{"id": 1, "tag_name": "v1.0.0"}, {"id": 2, "tag_name": "v1.1.0"}]
    mock_get.return_value = mock_response

    releases = fetch_github_releases("owner", "repo")

    assert len(releases) == 2
    assert releases[0]["tag_name"] == "v1.0.0"
    assert releases[1]["tag_name"] == "v1.1.0"
    mock_get.assert_called_once_with("https://api.github.com/repos/owner/repo/releases", headers={}, timeout=5)


@patch("requests.get")
def test_fetch_releases_empty(mock_get):
    # Mock an empty response
    mock_response = MagicMock()
    mock_response.json.return_value = []
    mock_get.return_value = mock_response

    releases = fetch_github_releases("owner", "repo")

    assert len(releases) == 0


@patch("requests.get")
def test_fetch_releases_error(mock_get):
    # Mock a request exception
    mock_get.side_effect = requests.RequestException("API error")

    with pytest.raises(requests.RequestException):
        fetch_github_releases("owner", "repo")


def test_parse_releases_with_semver():
    input_releases = [
        {"tag_name": "v1.2.3", "zipball_url": "https://api.github.com/repos/owner/repo/zipball/v1.2.3"},
        {"tag_name": "2.0.0", "zipball_url": "https://api.github.com/repos/owner/repo/zipball/2.0.0"},
    ]

    result = parse_releases(input_releases)

    assert len(result) == 2
    assert result[0]["version"] == semver.VersionInfo.parse("1.2.3")
    assert result[0]["tag"] == "v1.2.3"
    assert result[0]["download_url"] == "https://api.github.com/repos/owner/repo/zipball/v1.2.3"
    assert result[1]["version"] == semver.VersionInfo.parse("2.0.0")
    assert result[1]["tag"] == "2.0.0"


def test_parse_releases_with_special_tags():
    input_releases = [
        {"tag_name": "latest", "zipball_url": "https://api.github.com/repos/owner/repo/zipball/latest"},
        {"tag_name": "nightly", "zipball_url": "https://api.github.com/repos/owner/repo/zipball/nightly"},
    ]

    result = parse_releases(input_releases)

    assert len(result) == 2
    assert result[0]["version"] is None
    assert result[0]["tag"] == "latest"
    assert result[1]["version"] is None
    assert result[1]["tag"] == "nightly"


def test_parse_releases_mixed():
    input_releases = [
        {"tag_name": "v1.0.0", "zipball_url": "https://api.github.com/repos/owner/repo/zipball/v1.0.0"},
        {"tag_name": "latest", "zipball_url": "https://api.github.com/repos/owner/repo/zipball/latest"},
        {"tag_name": "2.0.0-beta", "zipball_url": "https://api.github.com/repos/owner/repo/zipball/2.0.0-beta"},
    ]

    result = parse_releases(input_releases)

    assert len(result) == 3
    assert result[0]["version"] == semver.VersionInfo.parse("1.0.0")
    assert result[1]["version"] is None
    assert result[1]["tag"] == "latest"
    assert result[2]["version"] == semver.VersionInfo.parse("2.0.0-beta")


def test_parse_releases_empty_list():
    input_releases: List[Dict[str, str]] = []

    result = parse_releases(input_releases)

    assert len(result) == 0


def test_parse_releases_invalid_semver():
    input_releases = [
        {"tag_name": "invalid", "zipball_url": "https://api.github.com/repos/owner/repo/zipball/invalid"},
    ]

    with pytest.raises(ValueError):
        parse_releases(input_releases)


# Sample data for tests
sample_releases: List[GithubRelease] = [
    {"version": semver.VersionInfo.parse("1.0.0"), "tag": "v1.0.0", "download_url": "url1"},
    {"version": semver.VersionInfo.parse("1.1.0"), "tag": "v1.1.0", "download_url": "url2"},
    {"version": semver.VersionInfo.parse("2.0.0"), "tag": "v2.0.0", "download_url": "url3"},
    {"version": None, "tag": "latest", "download_url": "url_latest"},
    {"version": None, "tag": "nightly", "download_url": "url_nightly"},
]


def test_select_version_latest():
    result = select_version(sample_releases, "latest")
    assert result is not None
    assert result["tag"] == "latest"
    assert result["download_url"] == "url_latest"


def test_select_version_specific():
    result = select_version(sample_releases, "1.1.0")
    assert result is not None
    assert result["version"] == semver.VersionInfo.parse("1.1.0")
    assert result["tag"] == "v1.1.0"


def test_select_version_with_v_prefix():
    result = select_version(sample_releases, "v2.0.0")
    assert result is not None
    assert result["version"] == semver.VersionInfo.parse("2.0.0")
    assert result["tag"] == "v2.0.0"


def test_select_version_nonexistent():
    result = select_version(sample_releases, "3.0.0")
    assert result is None


def test_select_version_invalid():
    result = select_version(sample_releases, "invalid_version")
    assert result is None


def test_select_version_case_insensitive_latest():
    result = select_version(sample_releases, "LATEST")
    assert result is not None
    assert result["tag"] == "latest"


def test_select_version_nightly():
    # Note: This test will fail with the current implementation
    # as it doesn't handle "nightly" specifically
    result = select_version(sample_releases, "nightly")
    assert result is None  # or assert result is not None if you want to handle nightly


def test_select_version_empty_list():
    result = select_version([], "1.0.0")
    assert result is None


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
