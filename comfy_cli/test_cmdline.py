from typer.testing import CliRunner

from .cmdline import app

runner = CliRunner()


def test_display_version():
    result = runner.invoke(
        app,
        ["-v"],
    )

    assert result.exit_code == 0
    assert "0.0.0" in result.stdout
