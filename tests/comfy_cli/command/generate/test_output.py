"""Tests for download templating."""

import httpx

from comfy_cli.command.generate import output


def test_resolve_template_directory_shorthand(tmp_path):
    p = output._resolve_template(f"{tmp_path}/", "abc123", 0, "png")
    assert p == tmp_path / "abc123_0.png"


def test_resolve_template_placeholders(tmp_path):
    tmpl = str(tmp_path / "out_{request_id}_{index}.{ext}")
    p = output._resolve_template(tmpl, "abc", 2, "jpg")
    assert p == tmp_path / "out_abc_2.jpg"


def test_ext_from_response_known_mime():
    r = httpx.Response(200, headers={"content-type": "image/jpeg"})
    assert output._ext_from_response(r) == "jpg"


def test_ext_from_url_strips_query():
    assert output._ext_from_url("https://x/result.webp?sig=abc") == "webp"
