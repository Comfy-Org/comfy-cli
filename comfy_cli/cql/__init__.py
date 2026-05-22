"""object_info loader — normalize ComfyUI's ``/object_info`` into typed rows.

Public entry points:
    load_graph(input_path=..., host=..., port=...) -> dict
"""

from comfy_cli.cql.errors import CQLRuntimeError
from comfy_cli.cql.loader import load_graph

__all__ = ["CQLRuntimeError", "load_graph"]
