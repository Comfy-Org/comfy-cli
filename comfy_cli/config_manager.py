import os
import configparser
from comfy_cli.utils import singleton, get_os, is_running
from comfy_cli import constants


@singleton
class ConfigManager(object):
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.background = None
        self.load()

    @staticmethod
    def get_config_path():
        return constants.DEFAULT_CONFIG[get_os()]

    def write_config(self):
        config_file_path = os.path.join(self.get_config_path(), 'config.ini')
        dir_path = os.path.dirname(config_file_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        with open(config_file_path, 'w') as configfile:
            self.config.write(configfile)

    def load(self):
        config_file_path = os.path.join(self.get_config_path(), 'config.ini')
        if os.path.exists(config_file_path):
            self.config = configparser.ConfigParser()
            self.config.read(config_file_path)

        # TODO: We need a policy for clearing the tmp directory.
        tmp_path = os.path.join(self.get_config_path(), 'tmp')
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        if 'background' in self.config['DEFAULT']:
            bg_info = self.config['DEFAULT']['background'].strip('()').split(',')
            bg_info = [item.strip().strip("'") for item in bg_info]
            self.background = bg_info[0], int(bg_info[1]), int(bg_info[2])

            if not is_running(self.background[2]):
                self.remove_background()

    def fill_print_env(self, table):
        if self.config.has_option('DEFAULT', 'default_workspace'):
            table.add_row("Default ComfyUI workspace", self.config['DEFAULT']['default_workspace'])
        else:
            table.add_row("Default ComfyUI workspace", "No default ComfyUI workspace")

        if self.config.has_option('DEFAULT', 'recent_path'):
            table.add_row("Recent ComfyUI", self.config['DEFAULT']['recent_path'])
        else:
            table.add_row("Recent ComfyUI", "No recent run")

        if self.config.has_option('DEFAULT', 'background'):
            table.add_row("Background ComfyUI", self.config['DEFAULT']['background'])
        else:
            table.add_row("Background ComfyUI", "N/A")

    def remove_background(self):
        del self.config['DEFAULT']['background']
        self.write_config()
        self.background = None
