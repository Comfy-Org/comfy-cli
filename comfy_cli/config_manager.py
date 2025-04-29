import ast
import configparser
import os
from importlib.metadata import version
from typing import Optional, Tuple

from filelock import FileLock

from comfy_cli import constants, logging
from comfy_cli.utils import get_os, is_running, singleton


@singleton
class ConfigManager(object):
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.background: Optional[Tuple[str, int, int]] = None
        self.load()

    @staticmethod
    def get_config_path():
        return constants.DEFAULT_CONFIG[get_os()]

    def get_config_file_path(self):
        return os.path.join(self.get_config_path(), "config.ini")

    def write_config(self):
        config_file_path = os.path.join(self.get_config_path(), "config.ini")
        config_file_path_lock = os.path.join(self.get_config_path(), "config.ini.lock")
        dir_path = os.path.dirname(config_file_path)
        lock = FileLock(config_file_path_lock, timeout=10)
        with lock:
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            with open(config_file_path, "w") as configfile:
                self.config.write(configfile)

    def set(self, key, value, name: Optional[str] = None):
        """
        Set a key-value pair in the config file.
        """
        self.config[name or constants.CONFIG_DEFAULT_KEY][key] = value
        self.write_config()  # Write changes to file immediately

    def get(self, key, name: Optional[str] = None):
        """
        Get a value from the config file. Returns None if the key does not exist.
        """
        return self.config[name or constants.CONFIG_DEFAULT_KEY].get(
            key, None
        )  # Returns None if the key does not exist

    def load(self, name: Optional[str] = None):
        config_file_path = self.get_config_file_path()
        if os.path.exists(config_file_path):
            self.config = configparser.ConfigParser()
            config_file_path_lock = os.path.join(self.get_config_path(), "config.ini.lock")
            lock = FileLock(config_file_path_lock, timeout=10)
            with lock:
                self.config.read(config_file_path)

        # TODO: We need a policy for clearing the tmp directory.
        tmp_path = os.path.join(self.get_config_path(), "tmp")
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        section = name or constants.CONFIG_DEFAULT_KEY
        if self.config.has_option(section, constants.CONFIG_KEY_BACKGROUND):
            self.background = ast.literal_eval(self.config[section][constants.CONFIG_KEY_BACKGROUND])

            if not is_running(self.background[2]):
                self.remove_background(name=section)

    def fill_print_env(self, table, name: Optional[str] = None):
        section = name or constants.CONFIG_DEFAULT_KEY
        table.add_row("Config Path", self.get_config_file_path())

        launch_extras = ""
        if self.config.has_option(section, "default_workspace"):
            table.add_row(
                "Default ComfyUI workspace",
                self.config[section][constants.CONFIG_KEY_DEFAULT_WORKSPACE],
            )

            launch_extras = self.config[section].get(constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, "")
        else:
            table.add_row("Default ComfyUI workspace", "No default ComfyUI workspace")

        if launch_extras == "":
            launch_extras = "[bold red]None[/bold red]"

        table.add_row("Default ComfyUI launch extra options", launch_extras)

        if self.config.has_option(section, constants.CONFIG_KEY_RECENT_WORKSPACE):
            table.add_row(
                "Recent ComfyUI workspace",
                self.config[section][constants.CONFIG_KEY_RECENT_WORKSPACE],
            )
        else:
            table.add_row("Recent ComfyUI workspace", "No recent run")

        if self.config.has_option(section, "enable_tracking"):
            table.add_row(
                "Tracking Analytics",
                ("Enabled" if self.config[section]["enable_tracking"] == "True" else "Disabled"),
            )

        if self.config.has_option(section, constants.CONFIG_KEY_BACKGROUND):
            bg_info = self.background
            if bg_info:
                table.add_row(
                    "Background ComfyUI",
                    f"http://{bg_info[0]}:{bg_info[1]} (pid={bg_info[2]})",
                )
        else:
            table.add_row("Background ComfyUI", "[bold red]No[/bold red]")

    def remove_background(self, name: Optional[str] = None):
        section = name or constants.CONFIG_DEFAULT_KEY
        del self.config[section][constants.CONFIG_KEY_BACKGROUND]
        if name is not None:
            self.cleanup_session(name)
        self.write_config()
        self.background = None

    def cleanup_session(self, section: str):
        if section not in self.config or section == constants.CONFIG_DEFAULT_KEY:
            return
        overridden_options = [
            opt
            for opt in self.config.options(section)
            if section in self.config and opt in self.config[section] and opt not in self.config.defaults()
        ]
        if not overridden_options:
            self.config.remove_section(section)

    def get_cli_version(self):
        # Note: this approach should work for users installing the CLI via
        # PyPi and Homebrew (e.g., pip install comfy-cli)
        try:
            return version("comfy-cli")
        except Exception as e:
            logging.debug(f"Error occurred while retrieving CLI version using importlib.metadata: {e}")

        return "0.0.0"
