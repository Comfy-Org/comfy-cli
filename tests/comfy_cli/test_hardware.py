"""Tests for :mod:`comfy_cli.hardware`.

Each branch (macOS/apple, nvidia-smi, ctypes fallback, total failure) is mocked
at the module boundary so the tests run identically on any CI runner. The
overriding contract under test: ``detect_hardware`` never raises, always returns
RAM, and degrades every failed probe to ``None``.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import jsonschema

from comfy_cli import _safe_exec, hardware
from comfy_cli.env_checker import EnvChecker, format_hardware_summary

_EnvCheckerCls = EnvChecker.__closure__[0].cell_contents

SCHEMA_PATH = Path(hardware.__file__).parent / "schemas" / "env.json"


def _env_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


class TestDetectHardwareMacOS:
    def test_apple_silicon_unified_block(self):
        with (
            patch.object(hardware.platform, "system", return_value="Darwin"),
            patch.object(hardware.platform, "machine", return_value="arm64"),
            patch.object(hardware.platform, "release", return_value="25.4.0"),
            patch.object(hardware, "_run", return_value="Apple M4 Max"),
            patch.object(hardware, "_detect_ram_bytes", return_value=68719476736),
        ):
            hw = hardware.detect_hardware()

        assert hw["os"] == "Darwin"
        assert hw["os_version"] == "25.4.0"
        assert hw["arch"] == "arm64"
        assert hw["cpu"] == "Apple M4 Max"
        assert hw["ram_bytes"] == 68719476736
        assert hw["gpu"] == {
            "vendor": "apple",
            "model": "Apple M4 Max",
            "vram_bytes": None,
            "unified_memory": True,
        }

    def test_intel_mac_has_no_apple_gpu(self):
        """Intel Macs are not unified-memory; they fall through to the NVIDIA/AMD
        probes (which return nothing here) → gpu null, not an apple block."""
        with (
            patch.object(hardware.platform, "system", return_value="Darwin"),
            patch.object(hardware.platform, "machine", return_value="x86_64"),
            patch.object(hardware, "_run", return_value=None),
            patch.object(hardware.cuda_detect, "_load_libcuda", side_effect=OSError),
            patch.object(hardware, "_detect_ram_bytes", return_value=17179869184),
        ):
            hw = hardware.detect_hardware()

        assert hw["gpu"] is None
        assert hw["ram_bytes"] == 17179869184

    def test_apple_silicon_under_rosetta_unified_block(self):
        """An x86_64 Python under Rosetta 2 reports machine == 'x86_64', but
        sysctl.proc_translated == '1' reveals the underlying Apple Silicon, so
        the unified-memory block is still reported (not gpu null)."""

        def fake_run(cmd):
            if cmd == ["sysctl", "-n", "sysctl.proc_translated"]:
                return "1"
            if cmd == ["sysctl", "-n", "machdep.cpu.brand_string"]:
                return "Apple M4 Max"
            return None

        with (
            patch.object(hardware.platform, "system", return_value="Darwin"),
            patch.object(hardware.platform, "machine", return_value="x86_64"),
            patch.object(hardware.platform, "release", return_value="25.4.0"),
            patch.object(hardware, "_run", side_effect=fake_run),
            patch.object(hardware, "_detect_ram_bytes", return_value=68719476736),
        ):
            hw = hardware.detect_hardware()

        assert hw["arch"] == "x86_64"
        assert hw["cpu"] == "Apple M4 Max"
        assert hw["gpu"] == {
            "vendor": "apple",
            "model": "Apple M4 Max",
            "vram_bytes": None,
            "unified_memory": True,
        }


class TestDetectHardwareNvidiaSmi:
    def test_nvidia_smi_happy_path(self):
        def fake_run(cmd):
            if cmd[0] == "nvidia-smi":
                return "NVIDIA GeForce RTX 4090, 24564"
            return None

        with (
            patch.object(hardware.platform, "system", return_value="Linux"),
            patch.object(hardware.platform, "machine", return_value="x86_64"),
            patch.object(hardware, "_run", side_effect=fake_run),
            patch.object(hardware, "_detect_ram_bytes", return_value=137438953472),
        ):
            hw = hardware.detect_hardware()

        assert hw["gpu"] == {
            "vendor": "nvidia",
            "model": "NVIDIA GeForce RTX 4090",
            "vram_bytes": 24564 * 1024 * 1024,
            "unified_memory": False,
        }

    def test_nvidia_smi_multi_gpu_takes_first(self):
        with patch.object(
            hardware,
            "_run",
            return_value="NVIDIA GeForce RTX 4090, 24564\nNVIDIA GeForce RTX 3090, 24576",
        ):
            gpu = hardware._detect_gpu_nvidia_smi()
        assert gpu["model"] == "NVIDIA GeForce RTX 4090"
        assert gpu["vram_bytes"] == 24564 * 1024 * 1024


class _FakeLibCuda:
    """Minimal stand-in for a loaded ``libcuda`` CDLL for the ctypes path."""

    def __init__(self, name: str, total_bytes: int):
        self._name = name
        self._total = total_bytes

    def cuInit(self, flags):
        return 0

    def cuDeviceGet(self, dev_ptr, ordinal):
        dev_ptr._obj.value = 0
        return 0

    def cuDeviceGetName(self, buf, length, dev):
        buf.value = self._name.encode("utf-8")
        return 0

    def cuDeviceTotalMem_v2(self, size_ptr, dev):
        size_ptr._obj.value = self._total
        return 0


class TestDetectHardwareCtypesFallback:
    def test_nvidia_smi_absent_libcuda_used(self):
        fake = _FakeLibCuda("NVIDIA A100-SXM4-80GB", 85899345920)

        def fake_run(cmd):
            # nvidia-smi absent → None so we exercise the ctypes fallback.
            return None

        with (
            patch.object(hardware.platform, "system", return_value="Linux"),
            patch.object(hardware.platform, "machine", return_value="x86_64"),
            patch.object(hardware, "_run", side_effect=fake_run),
            patch.object(hardware.cuda_detect, "_load_libcuda", return_value=fake),
            patch.object(hardware, "_detect_ram_bytes", return_value=270582939648),
        ):
            hw = hardware.detect_hardware()

        assert hw["gpu"] == {
            "vendor": "nvidia",
            "model": "NVIDIA A100-SXM4-80GB",
            "vram_bytes": 85899345920,
            "unified_memory": False,
        }

    def test_byref_obj_contract(self):
        """Guard the ctypes mock trick: byref exposes the wrapped object so the
        fake driver can write results back through the pointer."""
        val = ctypes.c_int(7)
        assert ctypes.byref(val)._obj is val


class TestDetectHardwareNeverRaises:
    def test_all_probes_fail_yields_nulls_but_ram(self):
        with (
            patch.object(hardware.platform, "system", return_value="Windows"),
            patch.object(hardware.platform, "machine", return_value="AMD64"),
            patch.object(hardware.platform, "processor", return_value=""),
            patch.object(hardware, "_run", side_effect=TimeoutError("boom")),
            patch.object(hardware.cuda_detect, "_load_libcuda", side_effect=OSError),
        ):
            # Must not raise even though every subprocess/ctypes probe blows up.
            hw = hardware.detect_hardware()

        assert hw["os"] == "Windows"
        assert hw["cpu"] is None
        assert hw["gpu"] is None
        # RAM comes from psutil (a hard dep) and is unaffected by probe failures.
        assert isinstance(hw["ram_bytes"], int)
        assert hw["ram_bytes"] > 0

    def test_platform_calls_raising_do_not_escape(self):
        with (
            patch.object(hardware.platform, "system", side_effect=RuntimeError),
            patch.object(hardware.platform, "machine", side_effect=RuntimeError),
            patch.object(hardware.platform, "release", side_effect=RuntimeError),
            # Block the real GPU probes too: without these, a dev box with an
            # actual GPU detects it and the `gpu is None` assert fails.
            patch.object(hardware, "_run", side_effect=TimeoutError("boom")),
            patch.object(hardware.cuda_detect, "_load_libcuda", side_effect=OSError),
        ):
            hw = hardware.detect_hardware()
        assert hw["os"] is None
        assert hw["arch"] is None
        assert hw["gpu"] is None


class TestDetectRamBytesImportFailure:
    """The lazy ``import psutil`` in ``_detect_ram_bytes`` must live inside its
    own ``try`` block, or a failed import escapes and breaks
    ``detect_hardware``'s "never raises" contract."""

    def test_psutil_import_failure_falls_back_to_none(self):
        # Setting the sys.modules entry to None makes `import psutil` raise
        # ImportError, whether or not psutil is actually installed.
        with patch.dict(sys.modules, {"psutil": None}):
            assert hardware._detect_ram_bytes() is None

    def test_detect_hardware_survives_psutil_import_failure(self):
        with (
            patch.dict(sys.modules, {"psutil": None}),
            patch.object(hardware, "_run", side_effect=TimeoutError("boom")),
            patch.object(hardware.cuda_detect, "_load_libcuda", side_effect=OSError),
        ):
            hw = hardware.detect_hardware()

        assert hw["ram_bytes"] is None


class TestDetectGpuAmd:
    def test_amd_happy_path(self):
        payload = json.dumps(
            {"card0": {"Card SKU": "gfx90a", "GPU Name": "AMD Instinct MI210", "VRAM Total Memory (B)": "68702699520"}}
        )
        with patch.object(hardware, "_run", return_value=payload):
            gpu = hardware._detect_gpu_amd()
        assert gpu == {
            "vendor": "amd",
            "model": "AMD Instinct MI210",
            "vram_bytes": 68702699520,
            "unified_memory": False,
        }

    def test_amd_total_key_beats_used_key(self):
        """The 'VRAM Total Used Memory' key also contains 'vram'+'total'; the probe
        must report the total-capacity key, not usage."""
        payload = json.dumps(
            {
                "card0": {
                    "VRAM Total Used Memory (B)": "1073741824",
                    "VRAM Total Memory (B)": "68702699520",
                }
            }
        )
        with patch.object(hardware, "_run", return_value=payload):
            gpu = hardware._detect_gpu_amd()
        assert gpu["vram_bytes"] == 68702699520

    def test_amd_skips_non_card_metadata_block(self):
        """A leading non-card metadata dict must not be mistaken for the GPU."""
        payload = json.dumps(
            {
                "system": {"Driver version": "6.0.0"},
                "card0": {"GPU Name": "AMD Radeon RX 7900 XTX", "VRAM Total Memory (B)": "25757220864"},
            }
        )
        with patch.object(hardware, "_run", return_value=payload):
            gpu = hardware._detect_gpu_amd()
        assert gpu["model"] == "AMD Radeon RX 7900 XTX"
        assert gpu["vram_bytes"] == 25757220864

    def test_amd_empty_first_card_falls_through_to_next(self):
        """A card entry lacking the queried fields must not mask a later card
        that has them (heterogeneous multi-GPU payloads)."""
        payload = json.dumps(
            {
                "card0": {"Something Unrelated": "x"},
                "card1": {"GPU Name": "AMD Instinct MI210", "VRAM Total Memory (B)": "68702699520"},
            }
        )
        with patch.object(hardware, "_run", return_value=payload):
            gpu = hardware._detect_gpu_amd()
        assert gpu["model"] == "AMD Instinct MI210"
        assert gpu["vram_bytes"] == 68702699520

    def test_amd_all_none_reports_no_gpu(self):
        """An error/metadata-only payload with nothing parseable must yield None,
        not a phantom {vendor: amd, model: None, vram_bytes: None} block."""
        payload = json.dumps({"card0": {"Something Unrelated": "x"}})
        with patch.object(hardware, "_run", return_value=payload):
            assert hardware._detect_gpu_amd() is None


class TestFormatHardwareSummary:
    def test_markup_in_model_is_escaped_not_crashing(self):
        """CPU/GPU strings are untrusted; markup-like content (e.g. a close tag
        '[/]') must be escaped so rendering through Rich can't raise MarkupError."""
        import pytest
        from rich.errors import MarkupError
        from rich.text import Text

        raw_model = "GPU [/] X"
        # Sanity: the raw model really would crash Rich markup parsing.
        with pytest.raises(MarkupError):
            Text.from_markup(raw_model)

        hw = {
            "cpu": "Fake CPU",
            "ram_bytes": 17179869184,
            "gpu": {"vendor": "nvidia", "model": raw_model, "vram_bytes": 8 * 1024**3, "unified_memory": False},
        }
        summary = format_hardware_summary(hw)
        assert "\\[/]" in summary
        # Rendering with markup enabled (as cmdline.py does) must not raise.
        Text.from_markup(summary)

    def test_unified_memory_summary(self):
        hw = {
            "cpu": "Apple M4 Max",
            "ram_bytes": 68719476736,
            "gpu": {"vendor": "apple", "model": "Apple M4 Max", "vram_bytes": None, "unified_memory": True},
        }
        summary = format_hardware_summary(hw)
        assert "Apple M4 Max" in summary
        assert "64 GB RAM" in summary
        assert "unified" in summary

    def test_discrete_vram_summary(self):
        hw = {
            "cpu": "AMD Ryzen 9",
            "ram_bytes": 137438953472,
            "gpu": {"vendor": "nvidia", "model": "RTX 4090", "vram_bytes": 24 * 1024**3, "unified_memory": False},
        }
        summary = format_hardware_summary(hw)
        assert "RTX 4090" in summary
        assert "24 GB VRAM" in summary

    def test_no_gpu_and_missing_fields(self):
        summary = format_hardware_summary({"cpu": None, "ram_bytes": None, "gpu": None})
        assert "no GPU" in summary
        assert "unknown CPU" in summary


class TestFillDataHardware:
    @patch("comfy_cli.env_checker.check_comfy_server_running", return_value=False)
    @patch("comfy_cli.env_checker.ConfigManager")
    def test_fill_data_includes_hardware_and_validates(self, mock_cm, _mock_server):
        mock_cm.return_value.get_data.return_value = {"path": "/tmp/config"}

        checker = _EnvCheckerCls.__new__(_EnvCheckerCls)
        checker.virtualenv_path = None
        checker.conda_env = None

        data = checker.fill_data()
        assert "hardware" in data
        hw = data["hardware"]
        assert set(hw) >= {"os", "os_version", "arch", "cpu", "ram_bytes", "gpu"}

        # Complete the payload the way cmdline.py does (workspace is added there)
        # and validate the whole thing against env.json.
        data["workspace"] = {"path": None, "type": None}
        jsonschema.Draft202012Validator(_env_schema()).validate(data)

    def test_fully_nulled_hardware_still_validates(self):
        """A failed-probe payload (all nulls, gpu null) must stay schema-valid —
        this is why `hardware` is optional and every leaf is nullable."""
        payload = {
            "python": {"version": "3.12.1", "executable": "/usr/bin/python"},
            "config": {},
            "workspace": {"path": None, "type": None},
            "server": {"running": False},
            "hardware": {
                "os": None,
                "os_version": None,
                "arch": None,
                "cpu": None,
                "ram_bytes": None,
                "gpu": None,
            },
        }
        jsonschema.Draft202012Validator(_env_schema()).validate(payload)

    def test_payload_without_hardware_still_validates(self):
        """Older consumers / pre-hardware payloads must remain valid — `hardware`
        is not in `required`."""
        payload = {
            "python": {"version": "3.12.1", "executable": "/usr/bin/python"},
            "config": {},
            "workspace": {"path": None, "type": None},
            "server": {"running": False},
        }
        jsonschema.Draft202012Validator(_env_schema()).validate(payload)


class TestRunResolvesBinaryPath:
    """``_run`` must resolve the binary to an absolute path (never invoke by bare
    name) so Windows ``CreateProcess`` can't pick up a CWD-planted executable."""

    def test_run_invokes_resolved_absolute_path(self):
        captured = {}

        def fake_check_output(cmd, **kwargs):
            captured["cmd"] = cmd
            return "ok\n"

        with (
            patch.object(_safe_exec.shutil, "which", return_value="/usr/bin/sysctl"),
            patch.object(hardware.subprocess, "check_output", side_effect=fake_check_output),
        ):
            result = hardware._run(["sysctl", "-n", "machdep.cpu.brand_string"])

        assert result == "ok"
        # First element is the resolved absolute path, not the bare name; the
        # remaining arguments are preserved verbatim.
        assert captured["cmd"] == ["/usr/bin/sysctl", "-n", "machdep.cpu.brand_string"]

    def test_run_skips_when_binary_absent(self):
        """A binary missing from PATH resolves to None → probe is skipped and the
        subprocess is never spawned."""
        with (
            patch.object(_safe_exec.shutil, "which", return_value=None),
            patch.object(hardware.subprocess, "check_output") as mock_run,
        ):
            assert hardware._run(["nvidia-smi", "--query-gpu=name"]) is None
        mock_run.assert_not_called()

    def test_run_never_raises_on_resolve_failure(self):
        """Even a broken PATH lookup degrades to None, honoring the never-raise
        contract."""
        with patch.object(_safe_exec.shutil, "which", side_effect=RuntimeError("boom")):
            assert hardware._run(["nvidia-smi"]) is None

    def test_run_empty_cmd_returns_none(self):
        """An empty command degrades to None instead of raising IndexError,
        honoring the never-raise contract."""
        with patch.object(hardware.subprocess, "check_output") as mock_run:
            assert hardware._run([]) is None
        mock_run.assert_not_called()

    def test_run_never_spawns_a_relative_path(self):
        """A relative ``which`` match is anchored in the CWD, so ``_run`` skips
        the probe rather than letting ``subprocess`` re-resolve it there."""
        with (
            patch.object(_safe_exec.shutil, "which", return_value=os.path.join("subdir", "nvidia-smi")),
            patch.object(hardware.subprocess, "check_output") as mock_run,
        ):
            assert hardware._run(["nvidia-smi", "--query-gpu=name"]) is None
        mock_run.assert_not_called()
