import configparser
import os
import configparser
from comfy_cli.utils import singleton, get_os, is_running
from comfy_cli import constants
from rich import print


@singleton
class ConfigManager(object):
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.background = None
        self.load()

    @staticmethod
    def get_config_path():
        return constants.DEFAULT_CONFIG[get_os()]

    def get_config_file_path(self):
        return os.path.join(self.get_config_path(), "config.ini")

    def write_config(self):
        config_file_path = os.path.join(self.get_config_path(), "config.ini")
        dir_path = os.path.dirname(config_file_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        with open(config_file_path, "w") as configfile:
            self.config.write(configfile)

    def set(self, key, value):
        """
        Set a key-value pair in the config file.
        """
        self.config["DEFAULT"][key] = value
        self.write_config()  # Write changes to file immediately

    def get(self, key):
        """
        Get a value from the config file. Returns None if the key does not exist.
        """
        return self.config["DEFAULT"].get(
            key, None
        )  # Returns None if the key does not exist

    def load(self):
        config_file_path = self.get_config_file_path()
        if os.path.exists(config_file_path):
            self.config = configparser.ConfigParser()
            self.config.read(config_file_path)

        # TODO: We need a policy for clearing the tmp directory.
        tmp_path = os.path.join(self.get_config_path(), "tmp")
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        if "background" in self.config["DEFAULT"]:
            bg_info = self.config["DEFAULT"]["background"].strip("()").split(",")
            bg_info = [item.strip().strip("'") for item in bg_info]
            self.background = bg_info[0], int(bg_info[1]), int(bg_info[2])

            if not is_running(self.background[2]):
                self.remove_background()

    def fill_print_env(self, table):
        table.add_row("Config Path", self.get_config_file_path())
        if self.config.has_option("DEFAULT", "default_workspace"):
            table.add_row(
                "Default ComfyUI workspace", self.config["DEFAULT"]["default_workspace"]
            )
        else:
            table.add_row("Default ComfyUI workspace", "No default ComfyUI workspace")

        if self.config.has_option("DEFAULT", constants.CONFIG_KEY_RECENT_WORKSPACE):
            table.add_row(
                "Recent ComfyUI workspace",
                self.config["DEFAULT"][constants.CONFIG_KEY_RECENT_WORKSPACE],
            )
        else:
            table.add_row("Recent ComfyUI workspace", "No recent run")

        if self.config.has_option("DEFAULT", "background"):
            bg_info = self.background
            table.add_row(
                "Background ComfyUI",
                f"http://{bg_info[0]}:{bg_info[1]} (pid={bg_info[2]})",
            )
        else:
            table.add_row("Background ComfyUI", "[bold red]No[/bold red]")

    def remove_background(self):
        del self.config["DEFAULT"]["background"]
        self.write_config()
        self.background = None
