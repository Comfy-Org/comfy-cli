import configparser
import os
from importlib.metadata import version

from comfy_cli import constants, logging
from comfy_cli.utils import get_os, is_running, singleton


@singleton
class ConfigManager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.background: tuple[str, int, int] | None = None
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
        return self.config["DEFAULT"].get(key, None)  # Returns None if the key does not exist

    def get_bool(self, key) -> bool | None:
        """
        Get a boolean value from the config file using configparser's built-in
        getboolean, which accepts: true/false, yes/no, on/off, 1/0 (case-insensitive).

        Returns None if the key does not exist.
        """
        if not self.config.has_option("DEFAULT", key):
            return None
        return self.config.getboolean("DEFAULT", key)

    def get_or_override(self, env_key: str, config_key: str, set_value: str | None = None) -> str | None:
        """
        Resolves and conditionally stores a config value.

        The selected value and action is determined by the following priority:

        1. Use CLI-provided `--set-*` value (if not None), and save it to config via `set()`.
        2. Use process environment variable if exists (empty strings are allowed).
        3. Otherwise, use the current config value via `get()`.

        Returns None if the selected value is an empty string.
        """

        if set_value is not None:
            self.set(config_key, set_value)
            return set_value or None
        elif env_key in os.environ:
            return os.environ[env_key] or None
        else:
            return self.get(config_key) or None

    def load(self):
        config_file_path = self.get_config_file_path()
        if os.path.exists(config_file_path):
            self.config = configparser.ConfigParser()
            self.config.read(config_file_path)

        # TODO: We need a policy for clearing the tmp directory.
        tmp_path = os.path.join(self.get_config_path(), "tmp")
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        if constants.CONFIG_KEY_BACKGROUND in self.config["DEFAULT"]:
            bg_info = self.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND].strip("()").split(",")
            bg_info = [item.strip().strip("'") for item in bg_info]
            self.background = bg_info[0], int(bg_info[1]), int(bg_info[2])

            if not is_running(self.background[2]):
                self.remove_background()

    def get_env_data(self):
        """
        Get environment data as a list of tuples for display.

        Returns:
            List[Tuple[str, str]]: List of (key, value) tuples for environment data.
        """
        data = []
        data.append(("Config Path", self.get_config_file_path()))

        launch_extras = ""
        if self.config.has_option("DEFAULT", constants.CONFIG_KEY_DEFAULT_WORKSPACE):
            data.append(
                (
                    "Default ComfyUI workspace",
                    self.config["DEFAULT"][constants.CONFIG_KEY_DEFAULT_WORKSPACE],
                )
            )
            launch_extras = self.config["DEFAULT"].get(constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, "")
        else:
            data.append(("Default ComfyUI workspace", "No default ComfyUI workspace"))

        if launch_extras == "":
            launch_extras = "[bold red]None[/bold red]"

        data.append(("Default ComfyUI launch extra options", launch_extras))

        if self.config.has_option("DEFAULT", constants.CONFIG_KEY_RECENT_WORKSPACE):
            data.append(
                (
                    "Recent ComfyUI workspace",
                    self.config["DEFAULT"][constants.CONFIG_KEY_RECENT_WORKSPACE],
                )
            )
        else:
            data.append(("Recent ComfyUI workspace", "No recent run"))

        tracking = self.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING)
        if tracking is not None:
            data.append(
                (
                    "Tracking Analytics",
                    "Enabled" if tracking else "Disabled",
                )
            )

        if self.config.has_option("DEFAULT", constants.CONFIG_KEY_BACKGROUND):
            bg_info = self.background
            if bg_info:
                data.append(
                    (
                        "Background ComfyUI",
                        f"http://{bg_info[0]}:{bg_info[1]} (pid={bg_info[2]})",
                    )
                )
        else:
            data.append(("Background ComfyUI", "[bold red]No[/bold red]"))

        return data

    def remove_background(self):
        del self.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND]
        self.write_config()
        self.background = None

    def get_cli_version(self):
        # Note: this approach should work for users installing the CLI via
        # PyPi and Homebrew (e.g., pip install comfy-cli)
        try:
            return version("comfy-cli")
        except Exception as e:
            logging.debug(f"Error occurred while retrieving CLI version using importlib.metadata: {e}")

        return "0.0.0"
