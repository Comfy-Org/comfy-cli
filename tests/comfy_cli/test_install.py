import pytest

from comfy_cli.command.install import validate_version


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


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
