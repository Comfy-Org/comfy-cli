import typer
from typing_extensions import Annotated

app = typer.Typer()


@app.command()
def get(url: Annotated[str, typer.Argument(help="The url of the model")],
        path: Annotated[str, typer.Argument(help="The path to install the model.")]):
  """Download model"""
  print(f"Start downloading url: ${url} into ${path}")
  download_model(url, path)


@app.command()
def remove():
  """Remove a custom node"""
  # TODO


def download_model(url: str, path: str):
  import httpx
  import pathlib
  from tqdm import tqdm

  local_filename = url.split("/")[-1]
  local_filepath = pathlib.Path(path, local_filename)
  local_filepath.parent.mkdir(parents=True, exist_ok=True)

  print(f"downloading {url} ...")
  with httpx.stream("GET", url, follow_redirects=True) as stream:
    total = int(stream.headers["Content-Length"])
    with open(local_filepath, "wb") as f, tqdm(
      total=total, unit_scale=True, unit_divisor=1024, unit="B"
    ) as progress:
      num_bytes_downloaded = stream.num_bytes_downloaded
      for data in stream.iter_bytes():
        f.write(data)
        progress.update(
          stream.num_bytes_downloaded - num_bytes_downloaded
        )
        num_bytes_downloaded = stream.num_bytes_downloaded
