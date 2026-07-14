"""COMFY_OBJECT_INFO_FILE is honored by the single object_info loader, so EVERY
CQL consumer routed through it — workflow edits, `nodes show`/`find`, `validate`,
fragments — resolves the node schema from a baked/pre-warmed file offline: no
network fetch, no cloud credential. A host sets one env var instead of threading
`--input` through each command."""

from __future__ import annotations

import comfy_cli.cql.engine as engine
import comfy_cli.cql.loader as loader


def test_resilient_load_honors_object_info_file_env(monkeypatch, tmp_path):
    dump = tmp_path / "object_info.json"
    dump.write_text('{"KSampler": {}}')
    seen: dict[str, str] = {}

    def fake_load(p):
        seen["path"] = p
        return {"ok": True}

    monkeypatch.setattr(engine, "_load_from_file", fake_load)
    # The network path must NOT run when the env dump is set.
    monkeypatch.setattr(
        engine, "_load_from_target",
        lambda **_: (_ for _ in ()).throw(AssertionError("network fetch should not run with COMFY_OBJECT_INFO_FILE set")),
    )
    monkeypatch.setenv("COMFY_OBJECT_INFO_FILE", str(dump))

    out = loader.resilient_load_object_info(mode="cloud")

    assert out == {"ok": True}
    assert seen["path"] == str(dump), "no --input => COMFY_OBJECT_INFO_FILE is read"


def test_explicit_input_wins_over_env(monkeypatch, tmp_path):
    env_dump = tmp_path / "env.json"
    env_dump.write_text("{}")
    explicit = tmp_path / "explicit.json"
    explicit.write_text("{}")
    seen: dict[str, str] = {}
    monkeypatch.setattr(engine, "_load_from_file", lambda p: seen.setdefault("path", p) or {})
    monkeypatch.setenv("COMFY_OBJECT_INFO_FILE", str(env_dump))

    loader.resilient_load_object_info(mode="cloud", input_path=str(explicit))

    assert seen["path"] == str(explicit), "explicit --input overrides the env default"


def test_no_env_falls_through_to_network(monkeypatch):
    # Neither --input nor the env var: the loader proceeds to the cache/network
    # path (asserted by _load_from_file NOT being called with an env path).
    called: dict[str, bool] = {}
    monkeypatch.setattr(engine, "_load_from_file", lambda p: called.setdefault("file", True) or {})
    monkeypatch.setattr(engine, "_load_from_target", lambda **_: {"from": "network"})
    monkeypatch.delenv("COMFY_OBJECT_INFO_FILE", raising=False)
    # A fresh cache miss forces the network path.
    monkeypatch.setattr(loader, "read_fresh_object_info_cache", lambda *a, **k: None)
    monkeypatch.setattr(loader, "write_object_info_cache", lambda *a, **k: None)

    out = loader.resilient_load_object_info(mode="cloud")

    assert out == {"from": "network"}
    assert "file" not in called, "no env => the offline file path is not taken"
