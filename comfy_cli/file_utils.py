import json
import os
import pathlib
import zipfile
from typing import Optional

import httpx
import requests
from pathspec import pathspec

from comfy_cli import ui


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
                json_object = json.loads(input_data)

                return json_object

            except json.JSONDecodeError as e:
                # Handle JSON decoding error
                print(f"JSON decoding error: {e}")

        msg_json = parse_json(message)
        if msg_json is not None:
            if "message" in msg_json:
                return f"Unauthorized download ({status_code}).\n{msg_json['message']}\nor you can set civitai api token using `comfy model download --set-civitai-api-token <token>`"
        return f"Unauthorized download ({status_code}), you might need to manually log into browser to download one"
    elif status_code == 403:
        return f"Forbidden url ({status_code}), you might need to manually log into browser to download one"
    elif status_code == 404:
        return "Sorry, your file is in another castle (404)"
    return f"Unknown error occurred (status code: {status_code})"


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


def zip_files(zip_filename):
    gitignore_path = ".gitignore"
    if not os.path.exists(gitignore_path):
        print(f"No .gitignore file found in {os.getcwd()}, proceeding without it.")
        gitignore = ""
    else:
        with open(gitignore_path, "r") as file:
            gitignore = file.read()

    spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore.splitlines())

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            if ".git" in dirs:
                dirs.remove(".git")
            for file in files:
                file_path = os.path.join(root, file)
                # Skip zipping the zip file itself
                if zip_filename in file_path:
                    continue
                relative_path = os.path.relpath(file_path, start=".")
                if not spec.match_file(relative_path):
                    zipf.write(file_path, relative_path)


def upload_file_to_signed_url(signed_url: str, file_path: str):
    try:
        with open(file_path, "rb") as f:
            headers = {"Content-Type": "application/gzip"}
            response = requests.put(signed_url, data=f, headers=headers)

            # Simple success check
            if response.status_code == 200:
                print("Upload successful.")
            else:
                # Print a generic error message with status code and response text
                print(f"Upload failed with status code: {response.status_code}. Error: {response.text}")

    except requests.exceptions.RequestException as e:
        # Print error related to the HTTP request
        print(f"An error occurred during the upload: {str(e)}")
    except FileNotFoundError:
        # Print file not found error
        print(f"Error: The file {file_path} does not exist.")


def extract_package_as_zip(file_path: pathlib.Path, extract_path: pathlib.Path):
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted zip file to {extract_path}")
    except zipfile.BadZipFile:
        print("File is not a zip or is corrupted.")
