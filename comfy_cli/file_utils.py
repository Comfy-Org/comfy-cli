import json
import os
import pathlib
import subprocess
import zipfile
from typing import List, Optional, Union

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


def check_unauthorized(url: str, headers: Optional[dict] = None) -> bool:
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


def download_file(url: str, local_filepath: pathlib.Path, headers: Optional[dict] = None):
    """Helper function to download a file."""
    local_filepath.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    with httpx.stream("GET", url, follow_redirects=True, headers=headers) as response:
        if response.status_code == 200:
            total = int(response.headers["Content-Length"])
            try:
                with open(local_filepath, "wb") as f:
                    for data in ui.show_progress(
                        response.iter_bytes(),
                        total,
                        description=f"Downloading {total // 1024 // 1024} MB",
                    ):
                        f.write(data)
            except KeyboardInterrupt:
                delete_eh = ui.prompt_confirm_action("Download interrupted, cleanup files?", True)
                if delete_eh:
                    local_filepath.unlink()
        else:
            status_reason = guess_status_code_reason(response.status_code, response.read())
            raise DownloadException(f"Failed to download file.\n{status_reason}")


def _load_comfyignore_spec(ignore_filename: str = ".comfyignore") -> Optional[PathSpec]:
    if not os.path.exists(ignore_filename):
        return None
    try:
        with open(ignore_filename, "r", encoding="utf-8") as ignore_file:
            patterns = [
                line.strip()
                for line in ignore_file
                if line.strip() and not line.lstrip().startswith("#")
            ]
    except OSError:
        return None

    if not patterns:
        return None

    return PathSpec.from_lines("gitwildmatch", patterns)


def list_git_tracked_files(base_path: Union[str, os.PathLike] = ".") -> List[str]:
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


def _is_force_included(rel_path: str, include_prefixes: List[str]) -> bool:
    return any(
        rel_path == prefix or rel_path.startswith(prefix + "/")
        for prefix in include_prefixes
        if prefix
    )


def zip_files(zip_filename, includes=None):
    """Zip git-tracked files respecting optional .comfyignore patterns."""
    includes = includes or []
    include_prefixes: List[str] = [
        _normalize_path(os.path.normpath(include.lstrip("/")))
        for include in includes
    ]

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
                dirs[:] = [
                    d
                    for d in dirs
                    if not should_ignore(_normalize_path(os.path.join(root, d)))
                ]
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
                print(
                    f"Warning: Included directory '{include_dir}' does not exist, creating empty directory"
                )
                arcname = rel_include or include_dir
                if not arcname.endswith("/"):
                    arcname = arcname + "/"
                zipf.writestr(arcname, "")
                continue

            for root, dirs, files in os.walk(include_dir):
                dirs[:] = [
                    d
                    for d in dirs
                    if not should_ignore(_normalize_path(os.path.join(root, d)))
                ]
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
