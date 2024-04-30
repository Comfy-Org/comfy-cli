import configparser
import os

from comfy_cli import constants
from comfy_cli.utils import singleton, get_os


@singleton
class ConfigManager(object):
  def __init__(self):
    self.config = configparser.ConfigParser()
    self.load()

  @staticmethod
  def get_config_path():
    return constants.DEFAULT_CONFIG[get_os()]

  def get_config_file_path(self):
    return os.path.join(self.get_config_path(), 'config.ini')

  def write_config(self):
    config_file_path = os.path.join(self.get_config_path(), 'config.ini')
    dir_path = os.path.dirname(config_file_path)
    if not os.path.exists(dir_path):
      os.mkdir(dir_path)

    with open(config_file_path, 'w') as configfile:
      self.config.write(configfile)

  def set(self, key, value):
    """
    Set a key-value pair in the config file.
    """
    self.config['DEFAULT'][key] = value
    self.write_config()  # Write changes to file immediately

  def get(self, key):
    """
    Get a value from the config file. Returns None if the key does not exist.
    """
    return self.config['DEFAULT'].get(key, None)  # Returns None if the key does not exist

  def load(self):
    config_file_path = self.get_config_file_path()
    if os.path.exists(config_file_path):
      self.config = configparser.ConfigParser()
      self.config.read(config_file_path)

    # TODO: We need a policy for clearing the tmp directory.
    tmp_path = os.path.join(self.get_config_path(), 'tmp')
    if not os.path.exists(tmp_path):
      os.makedirs(tmp_path)

  def fill_print_env(self, table):
    table.add_row("Config Path", self.get_config_file_path())
    if self.config.has_option('DEFAULT', 'default_workspace'):
      table.add_row("Default ComfyUI workspace", self.config['DEFAULT']['default_workspace'])
    else:
      table.add_row("Default ComfyUI workspace", "No default ComfyUI workspace")

    if self.config.has_option('DEFAULT', 'recent_path'):
      table.add_row("Recent ComfyUI", self.config['DEFAULT']['recent_path'])
    else:
      table.add_row("Recent ComfyUI", "No recent run")
