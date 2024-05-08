from rich.console import Console
import platform

is_windows = platform.system() == "Windows"

# Disable emoji on Windows because old Windows terminals don't support emoji.
console = Console(
    emoji=not is_windows,
)
