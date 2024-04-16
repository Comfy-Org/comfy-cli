import typer
from typing_extensions import Annotated
from comfy.command.models import models


from comfy.command import custom_nodes
from comfy.command import install as install_inner
from comfy.command import run as run_inner
from comfy import constants
from comfy.env_checker import EnvChecker


app = typer.Typer()

@app.command(help="Download and install ComfyUI")
def install(
    url: str = constants.COMFY_GITHUB_URL,
    workspace: Annotated[str, typer.Option(help="Path to the output directory.")] = None,
    ):
    install_inner.execute(url)

@app.command(help="Run workflow file")
def run(
    workflow_file: Annotated[str, typer.Option(help="Path to the workflow file.")],
    ):
    run_inner.execute(workflow_file)


@app.command(help="Print out current envirment variables.")
def env():
    env_checker = EnvChecker()
    env_checker.print()

app.add_typer(models.app, name="models", help="Manage models.")
app.add_typer(custom_nodes.app, name="custom_nodes", help="Manage custom nodes.")
