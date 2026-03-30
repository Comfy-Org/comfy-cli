import subprocess
from unittest.mock import MagicMock, patch

import pytest

from comfy_cli.cuda_detect import (
    DEFAULT_CUDA_TAG,
    PYTORCH_CUDA_WHEELS,
    _detect_via_ctypes,
    _detect_via_nvidia_smi,
    _load_libcuda,
    detect_cuda_driver_version,
    resolve_cuda_wheel,
)


class TestDetectViaCtypes:
    def test_happy_path(self):
        lib = MagicMock()
        lib.cuInit.return_value = 0

        def fake_get(ptr):
            ptr._obj.value = 13000
            return 0

        lib.cuDriverGetVersion.side_effect = fake_get

        with patch("comfy_cli.cuda_detect._load_libcuda", return_value=lib):
            assert _detect_via_ctypes() == 13000

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (12060, (12, 6)),
            (11080, (11, 8)),
            (13010, (13, 1)),
            (13000, (13, 0)),
            (12080, (12, 8)),
        ],
    )
    def test_version_decoding(self, raw, expected):
        lib = MagicMock()
        lib.cuInit.return_value = 0

        def fake_get(ptr):
            ptr._obj.value = raw
            return 0

        lib.cuDriverGetVersion.side_effect = fake_get

        with patch("comfy_cli.cuda_detect._load_libcuda", return_value=lib):
            result = _detect_via_ctypes()
        assert result == raw
        major = result // 1000
        minor = (result % 1000) // 10
        assert (major, minor) == expected

    def test_library_not_found(self):
        with patch("comfy_cli.cuda_detect._load_libcuda", side_effect=OSError("not found")):
            assert _detect_via_ctypes() is None

    def test_cuinit_fails(self):
        lib = MagicMock()
        lib.cuInit.return_value = 100

        with patch("comfy_cli.cuda_detect._load_libcuda", return_value=lib):
            assert _detect_via_ctypes() is None


class TestDetectViaNvidiaSmi:
    def test_happy_path(self):
        output = (
            "Mon Mar 30 12:00:00 2026\n"
            "+-------------------------+\n"
            "| NVIDIA-SMI 560.35.03    Driver Version: 560.35.03    CUDA Version: 12.6  |\n"
        )
        with patch("comfy_cli.cuda_detect.subprocess.check_output", return_value=output):
            assert _detect_via_nvidia_smi() == (12, 6)

    def test_cuda_13(self):
        output = "| NVIDIA-SMI 570.00    Driver Version: 570.00    CUDA Version: 13.0  |\n"
        with patch("comfy_cli.cuda_detect.subprocess.check_output", return_value=output):
            assert _detect_via_nvidia_smi() == (13, 0)

    def test_not_found(self):
        with patch("comfy_cli.cuda_detect.subprocess.check_output", side_effect=FileNotFoundError):
            assert _detect_via_nvidia_smi() is None

    def test_parse_failure(self):
        with patch("comfy_cli.cuda_detect.subprocess.check_output", return_value="some random output"):
            assert _detect_via_nvidia_smi() is None

    def test_timeout(self):
        with patch(
            "comfy_cli.cuda_detect.subprocess.check_output",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 10),
        ):
            assert _detect_via_nvidia_smi() is None


class TestDetectCudaDriverVersion:
    def test_ctypes_success_skips_smi(self):
        lib = MagicMock()
        lib.cuInit.return_value = 0

        def fake_get(ptr):
            ptr._obj.value = 12060
            return 0

        lib.cuDriverGetVersion.side_effect = fake_get

        with (
            patch("comfy_cli.cuda_detect._load_libcuda", return_value=lib),
            patch("comfy_cli.cuda_detect._detect_via_nvidia_smi") as mock_smi,
        ):
            result = detect_cuda_driver_version()
            assert result == (12, 6)
            mock_smi.assert_not_called()

    def test_ctypes_fails_falls_back_to_smi(self):
        with (
            patch("comfy_cli.cuda_detect._load_libcuda", side_effect=OSError),
            patch("comfy_cli.cuda_detect._detect_via_nvidia_smi", return_value=(13, 0)),
        ):
            assert detect_cuda_driver_version() == (13, 0)

    def test_both_fail(self):
        with (
            patch("comfy_cli.cuda_detect._load_libcuda", side_effect=OSError),
            patch("comfy_cli.cuda_detect._detect_via_nvidia_smi", return_value=None),
        ):
            assert detect_cuda_driver_version() is None

    def test_cuda_visible_devices_restored(self):
        import os

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}),
            patch("comfy_cli.cuda_detect._load_libcuda", side_effect=OSError),
            patch("comfy_cli.cuda_detect._detect_via_nvidia_smi", return_value=None),
        ):
            detect_cuda_driver_version()
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_cuda_visible_devices_empty_string(self):
        import os

        lib = MagicMock()
        env_during_call = {}

        def capturing_cuInit(val):
            env_during_call["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")
            return 0

        lib.cuInit.side_effect = capturing_cuInit

        def fake_get(ptr):
            ptr._obj.value = 13000
            return 0

        lib.cuDriverGetVersion.side_effect = fake_get

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
            patch("comfy_cli.cuda_detect._load_libcuda", return_value=lib),
        ):
            result = detect_cuda_driver_version()
            assert result == (13, 0)
            assert env_during_call["CUDA_VISIBLE_DEVICES"] == "UNSET"
            assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


class TestResolveCudaWheel:
    @pytest.mark.parametrize(
        "driver_version,expected",
        [
            ((13, 0), "cu130"),
            ((12, 9), "cu129"),
            ((12, 8), "cu128"),
            ((12, 7), "cu126"),
            ((12, 6), "cu126"),
            ((12, 5), "cu124"),
            ((12, 4), "cu124"),
            ((12, 1), "cu121"),
            ((12, 0), "cu118"),
            ((11, 8), "cu118"),
        ],
    )
    def test_mapping(self, driver_version, expected):
        assert resolve_cuda_wheel(driver_version) == expected

    def test_driver_too_old(self):
        assert resolve_cuda_wheel((11, 7)) is None
        assert resolve_cuda_wheel((10, 0)) is None

    def test_very_new_driver(self):
        assert resolve_cuda_wheel((14, 0)) == "cu130"
        assert resolve_cuda_wheel((15, 5)) == "cu130"

    def test_exact_match_preferred(self):
        assert resolve_cuda_wheel((13, 0)) == "cu130"
        assert resolve_cuda_wheel((12, 6)) == "cu126"


class TestLoadLibcuda:
    def test_linux_paths_tried_in_order(self):
        calls = []

        def tracking_cdll(path):
            calls.append(path)
            raise OSError("not found")

        with (
            patch("comfy_cli.cuda_detect.platform.system", return_value="Linux"),
            patch("comfy_cli.cuda_detect.ctypes.CDLL", side_effect=tracking_cdll),
            pytest.raises(OSError),
        ):
            _load_libcuda()

        assert calls == [
            "libcuda.so.1",
            "/usr/lib/wsl/lib/libcuda.so.1",
            "/usr/lib64/nvidia/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        ]

    def test_windows_path(self):
        calls = []

        def tracking_cdll(path):
            calls.append(path)
            raise OSError("not found")

        with (
            patch("comfy_cli.cuda_detect.platform.system", return_value="Windows"),
            patch("comfy_cli.cuda_detect.ctypes.CDLL", side_effect=tracking_cdll),
            pytest.raises(OSError),
        ):
            _load_libcuda()

        assert calls == ["nvcuda.dll"]

    def test_first_success_wins(self):
        mock_lib = MagicMock()

        def first_success(path):
            if path == "libcuda.so.1":
                return mock_lib
            raise OSError("not found")

        with (
            patch("comfy_cli.cuda_detect.platform.system", return_value="Linux"),
            patch("comfy_cli.cuda_detect.ctypes.CDLL", side_effect=first_success),
        ):
            result = _load_libcuda()
            assert result is mock_lib


class TestConstants:
    def test_wheels_in_descending_order(self):
        def parse_tag(tag):
            digits = tag[2:]
            return int(digits[0:2]), int(digits[2:])

        versions = [parse_tag(t) for t in PYTORCH_CUDA_WHEELS]
        assert versions == sorted(versions, reverse=True)

    def test_default_tag_is_in_wheel_list(self):
        assert DEFAULT_CUDA_TAG in PYTORCH_CUDA_WHEELS
