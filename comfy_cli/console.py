from rich.console import Console
import platform

is_windows = platform.system() == "Windows"

console = Console(
    emoji=is_windows,
)
