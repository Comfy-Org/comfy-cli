"""`generate` names a partner MODEL that is not an alias, instead of guessing aliases.

`comfy generate <name>` used to fail with
`Unknown model: 'gpt-image-1'` and `Unknown model: 'seedream'` ("Did you mean:
seedance, ideogram?"). Neither is a typo:

* gpt-image-1 is an OpenAI image model. The `dalle` alias
  (openai/images/generations) takes it as `--model`, but that schema types
  `model` as a free string, so nothing linked the name to the alias.
* seedream is ByteDance's IMAGE model family, served at
  byteplus/api/v3/images/generations (an enum of seedream-* ids in the spec).
  `comfy generate` has no alias for that route. The suggested `seedance` is
  the VIDEO model, which is the wrong substitute.
"""

from __future__ import annotations

import pytest

from comfy_cli.command.generate import spec


@pytest.fixture(autouse=True)
def _bundled_spec(monkeypatch, tmp_path):
    # Read the vendored spec, never a developer's ~/.comfy cache.
    monkeypatch.setattr(spec, "_USER_CACHE", tmp_path / "absent.yml")
    spec.load_raw_spec.cache_clear()
    spec._registry.cache_clear()
    yield
    spec.load_raw_spec.cache_clear()
    spec._registry.cache_clear()


@pytest.mark.parametrize("name", ["gpt-image-1", "gpt-image-1.5", "dall-e-3"])
def test_openai_model_name_points_at_the_dalle_alias(name):
    msg = spec._unknown_endpoint_message(name)
    assert f"comfy generate dalle --model {name}" in msg, msg


def test_seedream_names_its_route_and_does_not_suggest_seedance():
    msg = spec._unknown_endpoint_message("seedream")
    assert "byteplus/api/v3/images/generations" in msg, msg
    assert "seedream-4-5-251128" in msg, msg
    assert "no `comfy generate` alias" in msg, msg
    assert "seedance" not in msg, f"seedance is the video model, not a substitute: {msg}"


def test_plain_typo_keeps_did_you_mean():
    msg = spec._unknown_endpoint_message("flux-pr")
    assert "Did you mean:" in msg and "flux-pro" in msg, msg
