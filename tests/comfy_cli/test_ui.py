import io

from rich.console import Console

import comfy_cli.ui as ui_module
from comfy_cli.ui import display_error_message


def _capture(fn, *args, **kwargs):
    """Run fn against a Console that writes to an in-memory buffer, return the output string."""
    buf = io.StringIO()
    original = ui_module.console
    ui_module.console = Console(file=buf, force_terminal=False, markup=True)
    try:
        fn(*args, **kwargs)
    finally:
        ui_module.console = original
    return buf.getvalue()


class TestDisplayErrorMessageMarkup:
    """display_error_message must accept any string content without raising rich.errors.MarkupError
    or silently stripping bracketed substrings. Error messages can contain server-controlled text
    (e.g. JSON body echoed into a DownloadException) which may include arbitrary [ and ] chars."""

    def test_plain_message_rendered(self):
        out = _capture(display_error_message, "plain error")
        assert "plain error" in out

    def test_closing_tag_alone_does_not_crash(self):
        # Prior to the fix this raised rich.errors.MarkupError on console.print.
        out = _capture(display_error_message, "error with [/] in the middle")
        assert "[/]" in out

    def test_bracketed_substring_preserved(self):
        # Prior to the fix "[id]" was consumed as an unknown style and stripped.
        out = _capture(display_error_message, "URL /path/[id]/resource not found")
        assert "[id]" in out
        assert "/path/[id]/resource" in out

    def test_multiple_markup_like_tokens(self):
        out = _capture(display_error_message, "server said [redacted] at [host]:[port]")
        assert "[redacted]" in out
        assert "[host]" in out
        assert "[port]" in out

    def test_unbalanced_opening_bracket(self):
        out = _capture(display_error_message, "unbalanced [tag without close")
        assert "[tag without close" in out
