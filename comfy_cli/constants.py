from enum import Enum
import os


class OS(Enum):
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"


COMFY_GITHUB_URL = "https://github.com/comfyanonymous/ComfyUI"
COMFY_MANAGER_GITHUB_URL = "https://github.com/ltdrdata/ComfyUI-Manager"

DEFAULT_COMFY_MODEL_PATH = "models/checkpoints"
DEFAULT_COMFY_WORKSPACE = {
    OS.WINDOWS: os.path.join(os.path.expanduser("~"), "Documents", "comfy", "ComfyUI"),
    OS.MACOS: os.path.join(os.path.expanduser("~"), "Documents", "comfy", "ComfyUI"),
    OS.LINUX: os.path.join(os.path.expanduser("~"), "comfy", "ComfyUI"),
}

DEFAULT_CONFIG = {
    OS.WINDOWS: os.path.join(os.path.expanduser("~"), "AppData", "Local", "comfy-cli"),
    OS.MACOS: os.path.join(
        os.path.expanduser("~"), "Library", "Application Support", "comfy-cli"
    ),
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

COMFY_LOCK_YAML_FILE = "comfy.lock.yaml"

# TODO: figure out a better way to check if this is a comfy repo
COMFY_ORIGIN_URL_CHOICES = [
    "git@github.com:comfyanonymous/ComfyUI.git",
    "git@github.com:drip-art/comfy.git",
    "https://github.com/comfyanonymous/ComfyUI.git",
    "https://github.com/drip-art/ComfyUI.git",
    "https://github.com/comfyanonymous/ComfyUI",
    "https://github.com/drip-art/ComfyUI",
]


class GPU_OPTION(Enum):
    NVIDIA = "Nvidia"
    AMD = "Amd"
    INTEL_ARC = "Intel Arc"
    M_SERIES = "Mac M Series"
    MAC_INTEL = "Mac Intel"


# Referencing supported pt extension from ComfyUI
# https://github.com/comfyanonymous/ComfyUI/blob/a88b0ebc2d2f933c94e42aa689c42e836eedaf3c/folder_paths.py#L5
SUPPORTED_PT_EXTENSIONS = (".ckpt", ".pt", ".bin", ".pth", ".safetensors")

COMFY_REGISTRY_URL_ROOT = "http://localhost:8080"
