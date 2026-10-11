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


@pytest.mark.parametrize("name", ["gpt-image-1", "gpt-image-1.5", "GPT-Image-2"])
def test_gpt_image_name_points_at_the_partner_node_first(name):
    """A caller building a workflow runs generate with --emit-workflow, and
    `dalle` cannot emit. So the workflow route (the
    partner node) comes first, and `generate dalle --model` is labelled as the
    direct, non-workflow route."""
    msg = spec._unknown_endpoint_message(name)
    assert "OpenAIGPTImageNodeV2" in msg, msg
    assert msg.index("OpenAIGPTImageNodeV2") < msg.index("comfy generate dalle"), msg
    assert f"--model {name.lower()}" in msg, msg
    assert "--emit-workflow" in msg, f"must say the direct route cannot emit a workflow: {msg}"


def test_dall_e_name_names_only_the_direct_route():
    msg = spec._unknown_endpoint_message("dall-e-3")
    assert "comfy generate dalle --model dall-e-3" in msg, msg


def test_seedream_names_its_route_and_partner_node_search():
    msg = spec._unknown_endpoint_message("seedream")
    assert "byteplus/api/v3/images/generations" in msg, msg
    assert "seedream-4-5-251128" in msg, msg
    assert "no `comfy generate` alias" in msg, msg
    assert "comfy nodes search seedream" in msg, msg


@pytest.mark.parametrize(
    ("name", "alias"),
    [("flux", "flux-pro"), ("stable", "stability-sd3"), ("minimax", "minimax/video_generation"), ("seed", "seedance")],
)
def test_alias_typos_keep_did_you_mean_even_when_a_model_hint_applies(name, alias):
    """A short name that prefixes some partner's model ids ("flux" → recraft's
    flux1dev) is still most likely an alias typo; the hint is appended, never
    replaces the alias suggestions."""
    msg = spec._unknown_endpoint_message(name)
    assert "Did you mean:" in msg, msg
    did_you_mean = msg.split("Did you mean:", 1)[1].split("\n", 1)[0]
    assert alias in did_you_mean, msg


def test_model_hint_for_an_alias_with_emit_names_the_emit_route():
    """seedance can emit a workflow, so its --model hint may say so."""
    msg = spec._unknown_endpoint_message("seedance-1-5-pro")
    assert "comfy generate seedance --model" in msg, msg


def test_plain_typo_keeps_did_you_mean():
    msg = spec._unknown_endpoint_message("flux-pr")
    assert "Did you mean:" in msg and "flux-pro" in msg, msg
