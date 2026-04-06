import json
import os
import pathlib
import subprocess
import zipfile

import httpx
import requests
from pathspec import PathSpec

from comfy_cli import constants, ui


class DownloadException(Exception):
    pass


def guess_status_code_reason(status_code: int, message: str) -> str:
    if status_code == 401:

        def parse_json(input_data):
            try:
                # Check if the input is a byte string
                if isinstance(input_data, bytes):
                    # Decode the byte string to a regular string
                    input_data = input_data.decode("utf-8")

                # Parse the string as JSON
                return json.loads(input_data)

            except json.JSONDecodeError as e:
                # Handle JSON decoding error
                print(f"JSON decoding error: {e}")

        msg_json = parse_json(message)
        if msg_json is not None:
            if "message" in msg_json:
                return f"Unauthorized download ({status_code}).\n{msg_json['message']}\nor you can set a CivitAI API token using `comfy model download --set-civitai-api-token` or via the `{constants.CIVITAI_API_TOKEN_ENV_KEY}` environment variable"
        return f"Unauthorized download ({status_code}), you might need to manually log into a browser to download this"
    elif status_code == 403:
        return f"Forbidden url ({status_code}), you might need to manually log into a browser to download this"
    elif status_code == 404:
        return "File not found on server (404)"
    return f"Unknown error occurred (status code: {status_code})"


def check_unauthorized(url: str, headers: dict | None = None) -> bool:
    """
    Perform a GET request to the given URL and check if the response status code is 401 (Unauthorized).

    Args:
        url (str): The URL to send the GET request to.
        headers (Optional[dict]): Optional headers to include in the request.

    Returns:
        bool: True if the response status code is 401, False otherwise.
    """
    try:
        response = requests.get(url, headers=headers, allow_redirects=True, stream=True)
        return response.status_code == 401
    except requests.RequestException:
        # If there's an error making the request, we can't determine if it's unauthorized
        return False


def _poll_aria2_download(download) -> None:
    """Poll an aria2 download until completion, showing progress."""
    import time

    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Downloading...", total=None)

        while True:
            download.update()

            if download.total_length > 0:
                progress.update(task, total=download.total_length, completed=download.completed_length)

            if download.is_complete:
                if download.total_length > 0:
                    progress.update(task, completed=download.total_length)
                break
            elif download.has_failed:
                raise DownloadException(
                    f"aria2 download failed: {download.error_message} (code: {download.error_code})"
                )
            elif download.is_removed:
                raise DownloadException("aria2 download was removed before completion")

            time.sleep(0.5)


def _download_file_aria2(url: str, local_filepath: pathlib.Path, headers: dict | None = None) -> None:
    """Download a file using aria2 RPC."""
    try:
        import aria2p
    except ImportError:
        raise DownloadException(
            "aria2p is required for aria2 downloads. Install it with: pip install aria2p\n"
            "You also need a running aria2c daemon. See: https://aria2.github.io/"
        )

    server = os.environ.get(constants.ARIA2_SERVER_ENV_KEY)
    if not server:
        raise DownloadException(
            f"aria2 downloader selected but {constants.ARIA2_SERVER_ENV_KEY} environment variable is not set.\n"
            f"Set it to your aria2 RPC server URL, e.g.: export {constants.ARIA2_SERVER_ENV_KEY}=http://localhost:6800"
        )

    secret = os.environ.get(constants.ARIA2_SECRET_ENV_KEY, "")

    from urllib.parse import urlparse

    parsed = urlparse(server)
    host = f"{parsed.scheme}://{parsed.hostname}"
    port = parsed.port or 6800

    try:
        api = aria2p.API(aria2p.Client(host=host, port=port, secret=secret))
    except Exception as e:
        raise DownloadException(f"Failed to connect to aria2 RPC server at {server}: {e}")

    options = {
        "dir": str(local_filepath.parent),
        "out": local_filepath.name,
    }

    if headers:
        options["header"] = [f"{k}: {v}" for k, v in headers.items()]

    try:
        download = api.add_uris([url], options=options)
    except Exception as e:
        raise DownloadException(f"Failed to add download to aria2: {e}")

    _poll_aria2_download(download)


def download_file(url: str, local_filepath: pathlib.Path, headers: dict | None = None, downloader: str = "httpx"):
    """Helper function to download a file."""
    local_filepath.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    if downloader == "aria2":
        return _download_file_aria2(url, local_filepath, headers)

    with httpx.stream("GET", url, follow_redirects=True, headers=headers) as response:
        if response.status_code == 200:
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length is not None else None
            if total is not None:
                description = f"Downloading {total // 1024 // 1024} MB"
            else:
                description = "Downloading..."
            try:
                with open(local_filepath, "wb") as f:
                    for data in ui.show_progress(
                        response.iter_bytes(),
                        total,
                        description=description,
                    ):
                        f.write(data)
            except KeyboardInterrupt:
                delete_eh = ui.prompt_confirm_action("Download interrupted, cleanup files?", True)
                if delete_eh:
                    local_filepath.unlink()
        else:
            status_reason = guess_status_code_reason(response.status_code, response.read())
            raise DownloadException(f"Failed to download file.\n{status_reason}")


def _load_comfyignore_spec(ignore_filename: str = ".comfyignore") -> PathSpec | None:
    if not os.path.exists(ignore_filename):
        return None
    try:
        with open(ignore_filename, encoding="utf-8") as ignore_file:
            patterns = [line.strip() for line in ignore_file if line.strip() and not line.lstrip().startswith("#")]
    except OSError:
        return None

    if not patterns:
        return None

    return PathSpec.from_lines("gitwildmatch", patterns)


def list_git_tracked_files(base_path: str | os.PathLike = ".") -> list[str]:
    try:
        result = subprocess.check_output(
            ["git", "-C", os.fspath(base_path), "ls-files"],
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    return [line for line in result.splitlines() if line.strip()]


def _normalize_path(path: str) -> str:
    rel_path = os.path.relpath(path, start=".")
    if rel_path == ".":
        return ""
    return rel_path.replace("\\", "/")


def _is_force_included(rel_path: str, include_prefixes: list[str]) -> bool:
    return any(rel_path == prefix or rel_path.startswith(prefix + "/") for prefix in include_prefixes if prefix)


def zip_files(zip_filename, includes=None):
    """Zip git-tracked files respecting optional .comfyignore patterns."""
    includes = includes or []
    include_prefixes: list[str] = [_normalize_path(os.path.normpath(include.lstrip("/"))) for include in includes]

    included_paths: set[str] = set()
    git_files: list[str] = []

    ignore_spec = _load_comfyignore_spec()

    def should_ignore(rel_path: str) -> bool:
        if not ignore_spec:
            return False
        if _is_force_included(rel_path, include_prefixes):
            return False
        return ignore_spec.match_file(rel_path)

    zip_target = os.fspath(zip_filename)
    zip_abs_path = os.path.abspath(zip_target)
    zip_basename = os.path.basename(zip_abs_path)

    git_files = list_git_tracked_files(".")
    if not git_files:
        print("Warning: Not in a git repository or git not installed. Zipping all files.")

    with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zipf:
        if git_files:
            for file_path in git_files:
                if file_path == zip_basename:
                    continue

                rel_path = _normalize_path(file_path)
                if should_ignore(rel_path):
                    continue

                actual_path = os.path.normpath(file_path)
                if os.path.abspath(actual_path) == zip_abs_path:
                    continue
                if os.path.exists(actual_path):
                    arcname = rel_path or os.path.basename(actual_path)
                    zipf.write(actual_path, arcname)
                    included_paths.add(rel_path)
                else:
                    print(f"File not found. Not including in zip: {file_path}")
        else:
            for root, dirs, files in os.walk("."):
                if ".git" in dirs:
                    dirs.remove(".git")
                dirs[:] = [d for d in dirs if not should_ignore(_normalize_path(os.path.join(root, d)))]
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = _normalize_path(file_path)
                    if (
                        os.path.abspath(file_path) == zip_abs_path
                        or rel_path in included_paths
                        or should_ignore(rel_path)
                    ):
                        continue
                    arcname = rel_path or file_path
                    zipf.write(file_path, arcname)
                    included_paths.add(rel_path)

        for include_dir in includes:
            include_dir = os.path.normpath(include_dir.lstrip("/"))
            rel_include = _normalize_path(include_dir)

            if os.path.isfile(include_dir):
                if not should_ignore(rel_include) and rel_include not in included_paths:
                    arcname = rel_include or include_dir
                    zipf.write(include_dir, arcname)
                    included_paths.add(rel_include)
                continue

            if not os.path.exists(include_dir):
                print(f"Warning: Included directory '{include_dir}' does not exist, creating empty directory")
                arcname = rel_include or include_dir
                if not arcname.endswith("/"):
                    arcname = arcname + "/"
                zipf.writestr(arcname, "")
                continue

            for root, dirs, files in os.walk(include_dir):
                dirs[:] = [d for d in dirs if not should_ignore(_normalize_path(os.path.join(root, d)))]
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = _normalize_path(file_path)
                    if (
                        os.path.abspath(file_path) == zip_abs_path
                        or rel_path in included_paths
                        or should_ignore(rel_path)
                    ):
                        continue
                    arcname = rel_path or file_path
                    zipf.write(file_path, arcname)
                    included_paths.add(rel_path)


def upload_file_to_signed_url(signed_url: str, file_path: str):
    with open(file_path, "rb") as f:
        headers = {"Content-Type": "application/zip"}
        response = requests.put(signed_url, data=f, headers=headers)

        if response.status_code == 200:
            print("Upload successful.")
        else:
            raise Exception(f"Upload failed with status code: {response.status_code}. Error: {response.text}")


def extract_package_as_zip(file_path: pathlib.Path, extract_path: pathlib.Path):
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted zip file to {extract_path}")
    except zipfile.BadZipFile:
        print("File is not a zip or is corrupted.")
