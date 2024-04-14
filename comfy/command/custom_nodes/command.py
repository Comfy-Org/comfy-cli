import typer

app = typer.Typer()

# TODO(dr.lt.data): Add support for custom nodes managements
@app.command()
def add(name: str):
  """Add a new custom node"""
  print(f"Adding a new custom node: {name}")


@app.command()
def remove(name: str):
  """Remove a custom node"""
  print(f"Removing a custom node: {name}")
