from comfy_cli.command.models.models import check_civitai_url


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
