import json
import os
import pathlib
import zipfile
from typing import Optional

import httpx
import requests

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
        response = requests.get(url, headers=headers, allow_redirects=True)
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


def zip_files(zip_filename, includes=None):
    """
    Zip all files in the current directory that are tracked by git,
    plus any additional directories specified in includes.
    """
    includes = includes or []
    included_paths = set()
    git_files = []

    try:
        import subprocess

        git_files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Not in a git repository or git not installed. Zipping all files.")

    # Zip only git-tracked files
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        if git_files:
            for file_path in git_files:
                if zip_filename in file_path:
                    continue
                if os.path.exists(file_path):
                    zipf.write(file_path)
                    included_paths.add(file_path)
                else:
                    print(f"File not found. Not including in zip: {file_path}")
        else:
            for root, dirs, files in os.walk("."):
                if ".git" in dirs:
                    dirs.remove(".git")
                for file in files:
                    file_path = os.path.join(root, file)
                    # Skip zipping the zip file itself
                    if zip_filename in file_path:
                        continue
                    relative_path = os.path.relpath(file_path, start=".")
                    zipf.write(file_path, relative_path)
                    included_paths.add(file_path)

        for include_dir in includes:
            include_dir = include_dir.lstrip("/")
            if not os.path.exists(include_dir):
                print(f"Warning: Included directory '{include_dir}' does not exist, creating empty directory")
                zipf.writestr(f"{include_dir}/", "")
                continue

            for root, dirs, files in os.walk(include_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if zip_filename in file_path or file_path in included_paths:
                        continue
                    relative_path = os.path.relpath(file_path, start=".")
                    zipf.write(file_path, relative_path)
                    included_paths.add(file_path)


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
