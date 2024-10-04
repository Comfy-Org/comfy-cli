from comfy_cli.command.models.models import check_civitai_url, check_huggingface_url


def test_valid_model_url():
    url = "https://civitai.com/models/43331"
    assert check_civitai_url(url) == (True, False, 43331, None)


def test_valid_model_url_with_version():
    url = "https://civitai.com/models/43331/majicmix-realistic"
    assert check_civitai_url(url) == (True, False, 43331, None)


def test_valid_model_url_with_query():
    url = "https://civitai.com/models/43331?version=12345"
    assert check_civitai_url(url) == (True, False, 43331, 12345)


def test_valid_api_url():
    url = "https://civitai.com/api/download/models/67890"
    assert check_civitai_url(url) == (False, True, None, 67890)


def test_invalid_url():
    url = "https://example.com/models/43331"
    assert check_civitai_url(url) == (False, False, None, None)


def test_malformed_url():
    url = "https://civitai.com/models/"
    assert check_civitai_url(url) == (False, False, None, None)


def test_malformed_query_url():
    url = "https://civitai.com/models/43331?version="
    assert check_civitai_url(url) == (False, False, None, None)


def test_valid_huggingface_url():
    url = "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/sd-v1-4.ckpt"
    assert check_huggingface_url(url) == (True, "CompVis/stable-diffusion-v1-4", "sd-v1-4.ckpt", None, "main")


def test_valid_huggingface_url_sd_audio():
    url = "https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/model.safetensors"
    assert check_huggingface_url(url) == (True, "stabilityai/stable-audio-open-1.0", "model.safetensors", None, "main")


def test_valid_huggingface_url_with_folder():
    url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    assert check_huggingface_url(url) == (
        True,
        "runwayml/stable-diffusion-v1-5",
        "v1-5-pruned-emaonly.ckpt",
        None,
        "main",
    )


def test_valid_huggingface_url_with_subfolder():
    url = "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt"
    assert check_huggingface_url(url) == (
        True,
        "stabilityai/stable-diffusion-2-1",
        "v2-1_768-ema-pruned.ckpt",
        None,
        "main",
    )


def test_valid_huggingface_url_with_encoded_filename():
    url = "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/sd-v1-4%20(1).ckpt"
    assert check_huggingface_url(url) == (True, "CompVis/stable-diffusion-v1-4", "sd-v1-4 (1).ckpt", None, "main")


def test_invalid_huggingface_url():
    url = "https://example.com/CompVis/stable-diffusion-v1-4/resolve/main/sd-v1-4.ckpt"
    assert check_huggingface_url(url) == (False, None, None, None, None)


def test_invalid_huggingface_url_structure():
    url = "https://huggingface.co/CompVis/stable-diffusion-v1-4/main/sd-v1-4.ckpt"
    assert check_huggingface_url(url) == (False, None, None, None, None)


def test_huggingface_url_with_com_domain():
    url = "https://huggingface.com/CompVis/stable-diffusion-v1-4/resolve/main/sd-v1-4.ckpt"
    assert check_huggingface_url(url) == (True, "CompVis/stable-diffusion-v1-4", "sd-v1-4.ckpt", None, "main")


def test_huggingface_url_with_folder_structure():
    url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    assert check_huggingface_url(url) == (
        True,
        "stabilityai/stable-diffusion-xl-base-1.0",
        "sd_xl_base_1.0.safetensors",
        None,
        "main",
    )
