import typer
from typing_extensions import Annotated

from comfy_cli import tracking, logging

app = typer.Typer()


@app.command()
@tracking.track_command("model")
def get(url: Annotated[str, typer.Argument(help="The url of the model")],
        path: Annotated[str, typer.Argument(help="The path to install the model.")]):
  """Download model"""
  print(f"Start downloading url: ${url} into ${path}")
  download_model(url, path)


@app.command()
@tracking.track_command("model")
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

  logging.debug(f"downloading {url} ...")
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

  # def download_model(url: str, path: str):
  #   # Set up logging to file
  #   logging.basicConfig(level=logging.INFO, filename='download.log', filemode='w',
  #                       format='%(asctime)s - %(levelname)s - %(message)s')
  #
  #   local_filename = url.split("/")[-1]
  #   local_filepath = pathlib.Path(path, local_filename)
  #   local_filepath.parent.mkdir(parents=True, exist_ok=True)
  #
  #   # Log the URL being downloaded
  #   logging.info(f"Downloading {url} ...")
  #
  #   with httpx.stream("GET", url, follow_redirects=True) as stream:
  #     total = int(stream.headers["Content-Length"])
  #     with open(local_filepath, "wb") as f, tqdm(
  #       total=total, unit_scale=True, unit_divisor=1024, unit="B"
  #     ) as progress:
  #       num_bytes_downloaded = stream.num_bytes_downloaded
  #       for data in stream.iter_bytes():
  #         f.write(data)
  #         progress.update(
  #           stream.num_bytes_downloaded - num_bytes_downloaded
  #         )
  #         num_bytes_downloaded = stream.num_bytes_downloaded
  #
  #   # Log the completion of the download
  #   logging.info(f"Download completed. File saved to {local_filepath}")
