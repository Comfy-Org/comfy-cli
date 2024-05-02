from comfy_cli.registry.types import NodeConfiguration


def extract_node_configuration() -> NodeConfiguration:
    # TODO(robinhuang)
    # Check pyproject.toml is in current directory
    # Check all fields are present
    # Check the version is valid
    return NodeConfiguration(
        publisher_id="",
        node_id="",
        display_name="",
        description="",
        version="",
        license="",
        dependencies=[],
        tags=[],
        repository="",
        documentation="",
        author="",
        issues="",
        icon="",
    )
