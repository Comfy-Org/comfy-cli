from typer.testing import CliRunner

from .cmdline import app

runner = CliRunner()


def test_install_here():
    result = runner.invoke(
        app,
        ["--here", "--skip-prompt", "install", "--cpu"],
    )
    print("Stdout:")
    print(result.stdout)
    assert result.exit_code == 0
    assert "ComfyUI is installed at" in result.stdout


def test_display_version():
    result = runner.invoke(
        app,
        ["-v"],
    )

    assert result.exit_code == 0
    assert "0.0.0" in result.stdout
