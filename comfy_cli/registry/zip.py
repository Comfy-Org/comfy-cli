import os
import zipfile
import pathspec


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
                if not spec.match_file(file_path):
                    zipf.write(
                        file_path, os.path.relpath(file_path, os.path.join(root, ".."))
                    )


# TODO: check this code. this make slow down comfy-cli extremely
# zip_files("node.tar.gz")
