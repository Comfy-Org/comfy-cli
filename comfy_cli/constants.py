import os
from enum import Enum


class OS(str, Enum):
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"


class PROC(str, Enum):
    X86_64 = "x86_64"
    ARM = "arm"


COMFY_GITHUB_URL = "https://github.com/comfyanonymous/ComfyUI"

MANAGER_REQUIREMENTS_FILE = "manager_requirements.txt"

DEFAULT_COMFY_MODEL_PATH = "models"
DEFAULT_COMFY_WORKSPACE = {
    OS.WINDOWS: os.path.join(os.path.expanduser("~"), "Documents", "comfy", "ComfyUI"),
    OS.MACOS: os.path.join(os.path.expanduser("~"), "Documents", "comfy", "ComfyUI"),
    OS.LINUX: os.path.join(os.path.expanduser("~"), "comfy", "ComfyUI"),
}

DEFAULT_CONFIG = {
    OS.WINDOWS: os.path.join(os.path.expanduser("~"), "AppData", "Local", "comfy-cli"),
    OS.MACOS: os.path.join(os.path.expanduser("~"), "Library", "Application Support", "comfy-cli"),
    OS.LINUX: os.path.join(os.path.expanduser("~"), ".config", "comfy-cli"),
}

CONTEXT_KEY_WORKSPACE = "workspace"
CONTEXT_KEY_RECENT = "recent"
CONTEXT_KEY_HERE = "here"

CONFIG_KEY_DEFAULT_WORKSPACE = "default_workspace"
CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS = "default_launch_extras"
CONFIG_KEY_RECENT_WORKSPACE = "recent_workspace"
CONFIG_KEY_ENABLE_TRACKING = "enable_tracking"
CONFIG_KEY_USER_ID = "user_id"
CONFIG_KEY_INSTALL_EVENT_TRIGGERED = "install_event_triggered"
CONFIG_KEY_BACKGROUND = "background"
CONFIG_KEY_MANAGER_GUI_ENABLED = "manager_gui_enabled"  # Legacy, kept for backward compatibility
CONFIG_KEY_MANAGER_GUI_MODE = "manager_gui_mode"  # Valid: "disable", "enable-gui", "disable-gui", "enable-legacy-gui"
CONFIG_KEY_UV_COMPILE_DEFAULT = "uv_compile_default"

CIVITAI_API_TOKEN_KEY = "civitai_api_token"
CIVITAI_API_TOKEN_ENV_KEY = "CIVITAI_API_TOKEN"
HF_API_TOKEN_KEY = "hf_api_token"
HF_API_TOKEN_ENV_KEY = "HF_API_TOKEN"

ARIA2_SERVER_ENV_KEY = "COMFYUI_MANAGER_ARIA2_SERVER"
ARIA2_SECRET_ENV_KEY = "COMFYUI_MANAGER_ARIA2_SECRET"
CONFIG_KEY_DEFAULT_DOWNLOADER = "default_downloader"

DEFAULT_TRACKING_VALUE = True

COMFY_LOCK_YAML_FILE = "comfy.lock.yaml"

# TODO: figure out a better way to check if this is a comfy repo
COMFY_ORIGIN_URL_CHOICES = {
    "git@github.com:Comfy-Org/ComfyUI.git",
    "git@github.com:comfyanonymous/ComfyUI.git",
    "git@github.com:drip-art/comfy.git",
    "git@github.com:ltdrdata/ComfyUI.git",
    "https://github.com/Comfy-Org/ComfyUI.git",
    "https://github.com/comfyanonymous/ComfyUI.git",
    "https://github.com/drip-art/ComfyUI.git",
    "https://github.com/ltdrdata/ComfyUI.git",
    "https://github.com/Comfy-Org/ComfyUI",
    "https://github.com/comfyanonymous/ComfyUI",
    "https://github.com/drip-art/ComfyUI",
    "https://github.com/ltdrdata/ComfyUI",
}


class CUDAVersion(str, Enum):
    v13_0 = "13.0"
    v12_9 = "12.9"
    v12_8 = "12.8"
    v12_6 = "12.6"
    v12_4 = "12.4"
    v12_1 = "12.1"
    v11_8 = "11.8"


class ROCmVersion(str, Enum):
    v7_1 = "7.1"
    v7_0 = "7.0"
    v6_3 = "6.3"
    v6_2 = "6.2"
    v6_1 = "6.1"


class GPU_OPTION(str, Enum):
    CPU = None
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL_ARC = "intel_arc"
    MAC_M_SERIES = "mac_m_series"
    MAC_INTEL = "mac_intel"


# Referencing supported pt extension from ComfyUI
# https://github.com/comfyanonymous/ComfyUI/blob/a88b0ebc2d2f933c94e42aa689c42e836eedaf3c/folder_paths.py#L5
SUPPORTED_PT_EXTENSIONS = (".ckpt", ".pt", ".bin", ".pth", ".safetensors")

NODE_ZIP_FILENAME = "node.zip"

# The default minor version series to download from python-build-standalone.
# The exact patch version is resolved dynamically from the release metadata.
DEFAULT_STANDALONE_PYTHON_MINOR_VERSION = "3.12"
