import os
import yaml
import concurrent.futures
from pathlib import Path
import os

from comfy_cli.env_checker import EnvChecker
from comfy_cli.utils import singleton
from comfy_cli import constants


@singleton
class MetadataManager:
    """
    Manages the metadata for ComfyUI when running comfy cli, including loading,
    validating, and saving metadata to a file.
    """
    def __init__(self):
        self.metadata_file = None
        self.env_checker = EnvChecker()
        self.metadata = {}

    
    def scan_dir(self):
        config_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(constants.SUPPORTED_PT_EXTENSIONS):
                    config_files.append(os.path.join(root, file))
        return config_files


    def scan_dir_concur(self):
        base_path = Path(".")
        model_files = []

        # Function to check if the file is config.json
        def check_file(path):
            if path.name.endswith(constants.SUPPORTED_PT_EXTENSIONS):
                return str(path)

        # Use ThreadPoolExecutor to manage concurrency
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(check_file, p) for p in base_path.rglob('*')]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    model_files.append(future.result())

        return model_files
        

    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        else:
            return {}

    def validate_metadata(self):
        if self.metadata and "models" not in self.metadata:
            self.metadata["models"] = {}

    def save_metadata(self):
        with open(self.metadata_file, "w", encoding="utf-8") as file:
            yaml.dump(self.metadata, file)

    def update_model_metadata(self, model_name, model_path):
        if "models" not in self.metadata:
            self.metadata["models"] = {}
        self.metadata["models"][model_name] = model_path
        self.save_metadata()

    def get_model_path(self, model_name):
        if "models" in self.metadata and model_name in self.metadata["models"]:
            return self.metadata["models"][model_name]
        else:
            return None


if __name__ == "__main__":
    manager = MetadataManager()
    import pdb

    pdb.set_trace()

    # model_name = "example_model"
    # model_path = "/path/to/example_model"
    # manager.update_model_metadata(model_name, model_path)

    # retrieved_path = manager.get_model_path(model_name)
    # print(f"Retrieved path for {model_name}: {retrieved_path}")
