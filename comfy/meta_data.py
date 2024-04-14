import os
import yaml


class MetadataManager:
    def __init__(self, metadata_file):
        self.metadata_file = metadata_file
        self.metadata = self.load_metadata()

    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as file:
                return yaml.safe_load(file)
        else:
            return {}

    def save_metadata(self):
        with open(self.metadata_file, "w") as file:
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
