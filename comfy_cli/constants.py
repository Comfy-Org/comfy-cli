
COMFY_GITHUB_URL = 'https://github.com/comfyanonymous/ComfyUI'
COMFY_MANAGER_GITHUB_URL = 'https://github.com/ltdrdata/ComfyUI-Manager'
COMFY_WORKSPACE = '~/comfy'

# TODO: figure out a better way to check if this is a comfy repo
COMFY_ORIGIN_URL_CHOICES = [
    "git@github.com:comfyanonymous/ComfyUI.git",
    "git@github.com:drip-art/comfy.git",
    "https://github.com/comfyanonymous/ComfyUI.git",
    "https://github.com/drip-art/ComfyUI.git",
    "https://github.com/comfyanonymous/ComfyUI",
    "https://github.com/drip-art/ComfyUI",
]

# Referencing supported pt extension from ComfyUI
# https://github.com/comfyanonymous/ComfyUI/blob/a88b0ebc2d2f933c94e42aa689c42e836eedaf3c/folder_paths.py#L5
SUPPORTED_PT_EXTENSIONS = set(['.ckpt', '.pt', '.bin', '.pth', '.safetensors'])