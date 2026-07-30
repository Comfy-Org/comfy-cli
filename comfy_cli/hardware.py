"""Cross-platform hardware probing for ``comfy env --json``.

A single entry point, :func:`detect_hardware`, returns a JSON-serializable dict
describing the machine's generation-relevant hardware (OS, CPU, RAM, GPU). Agent
surfaces use this to route weak machines away from local diffusion.

Contract: **never raise, never block long.** Every probe is wrapped so a failure
yields ``None`` rather than an exception, and every subprocess is bounded by a
``timeout=5``. The RAM figure is the only value expected to always populate
(``psutil`` is a hard dependency); everything else degrades to ``None``.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import shutil
import subprocess

import psutil

from comfy_cli import cuda_detect

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 5


def _is_planted_in_cwd(path: str) -> bool:
    """Return ``True`` only if ``path`` resolves to a file sitting *directly* in
    ``os.getcwd()`` — the signature of a planted probe binary.

    ``shutil.which`` searches the current directory first on Windows (and on any
    platform whose ``$PATH`` contains ``.`` or an empty entry), so an attacker who
    controls the directory the user runs ``comfy env`` from can drop a malicious
    ``nvidia-smi.exe`` there. Such a plant always lands in the CWD *itself*, so we
    reject only a resolved binary whose parent directory **is** the CWD. A
    legitimate system binary in a *subdirectory* — e.g. ``System32`` even when the
    CWD is ``C:\\Windows``, or a drive root — is left untouched, honouring the
    "a legitimate system binary is never rejected" guarantee. Paths are compared
    with :func:`os.path.normcase` so Windows' case-insensitivity can't fail the
    guard open. Ambiguity — a path on a different drive, or an unresolvable one —
    is treated as *not* planted so a legitimate binary is never rejected.
    """
    try:
        cwd = os.path.normcase(os.path.realpath(os.getcwd()))
        # ``os.path.dirname`` of a bare/relative ``which`` result is "", which
        # ``realpath`` correctly resolves against the CWD.
        parent = os.path.normcase(os.path.realpath(os.path.dirname(path)))
        return parent == cwd
    except (OSError, ValueError):
        # Different drives (Windows) or an unresolvable path → not planted.
        return False


def _resolve_binary(name: str) -> str | None:
    """Resolve a probe binary to a trusted absolute path, or ``None`` to skip it.

    :func:`shutil.which` performs a PATH lookup and returns ``None`` when the
    binary is absent (so the probe simply degrades to ``None``). Passing the
    resolved absolute path to :func:`subprocess.check_output` — rather than the
    bare name — prevents Windows ``CreateProcess`` from searching the current
    working directory, so running ``comfy env`` from an attacker-controlled
    directory cannot execute a planted ``nvidia-smi.exe``.

    ``shutil.which`` may itself resolve against the current directory (always on
    Windows; on any platform when ``$PATH`` holds ``.`` or an empty entry), so as
    defense-in-depth two CWD-anchored results are additionally rejected on every
    platform:

    * a **relative** result. ``which`` returns ``os.path.join(entry, name)``, so a
      relative path means the matching ``$PATH`` entry was itself relative (``.``,
      an empty entry, ``subdir``, or Windows' implicitly prepended ``os.curdir``)
      and the binary therefore lives under the attacker-controlled CWD. Handing
      that string to :func:`subprocess.check_output` would re-resolve it against
      the CWD — exactly the hijack this function exists to prevent — so the probe
      is skipped instead. A binary found through a normal absolute ``$PATH`` entry
      always comes back absolute and is unaffected.
    * an absolute result sitting directly **in** the CWD (see
      :func:`_is_planted_in_cwd`), which covers the CWD appearing in ``$PATH`` as
      an absolute entry.

    A legitimate system binary (e.g. ``nvidia-smi.exe`` under ``System32``) is
    unaffected by either check.
    """
    try:
        path = shutil.which(name)
        if path is None:
            return None
        if not os.path.isabs(path):
            logger.debug("skipping hardware probe %r: relative PATH match anchored in CWD (%s)", name, path)
            return None
        if _is_planted_in_cwd(path):
            logger.debug("skipping hardware probe %r: resolved into CWD (%s)", name, path)
            return None
        return path
    except Exception:
        logger.debug("resolving hardware probe binary %r failed", name, exc_info=True)
        return None


def _run(cmd: list[str]) -> str | None:
    """Run ``cmd`` and return stripped stdout, or ``None`` on any failure.

    ``cmd[0]`` is resolved to a trusted absolute path via :func:`_resolve_binary`
    before execution (skipping the probe when absent or CWD-planted), and the run
    is bounded by ``timeout=5`` so a hung binary can never block the probe. An
    empty ``cmd`` degrades to ``None`` rather than raising, honouring the
    module's never-raise contract.
    """
    if not cmd:
        return None
    resolved = _resolve_binary(cmd[0])
    if resolved is None:
        return None
    try:
        output = subprocess.check_output(
            [resolved, *cmd[1:]],
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        logger.debug("hardware probe command failed: %s", cmd, exc_info=True)
        return None
    output = output.strip()
    return output or None


def _detect_cpu(system: str) -> str | None:
    """Best-effort human-readable CPU/chip name, or ``None`` if unknown."""
    try:
        if system == "Darwin":
            return _run(["sysctl", "-n", "machdep.cpu.brand_string"])

        if system == "Linux":
            try:
                with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if line.startswith("model name"):
                            _, _, value = line.partition(":")
                            value = value.strip()
                            if value:
                                return value
            except OSError:
                logger.debug("reading /proc/cpuinfo failed", exc_info=True)

        # Windows / Linux fallback: platform.processor() is often populated on
        # Windows and empty on Linux (hence the /proc/cpuinfo pass above first).
        return platform.processor() or None
    except Exception:
        logger.debug("CPU detection failed", exc_info=True)
        return None


def _detect_gpu_nvidia_smi() -> dict | None:
    """Query ``nvidia-smi`` for the first GPU's name and VRAM.

    Returns a gpu dict or ``None`` if the binary is absent / produced no usable
    output. ``cuda_detect`` already parses nvidia-smi in production.
    """
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None

    # First line is the first GPU: "NVIDIA GeForce RTX 4090, 24564"
    first = output.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        return None

    model = parts[0] or None
    vram_bytes = None
    try:
        mib = int(parts[1])
        vram_bytes = mib * 1024 * 1024
    except (ValueError, TypeError):
        logger.debug("could not parse nvidia-smi VRAM: %r", parts[1])

    if not model and vram_bytes is None:
        return None

    return {
        "vendor": "nvidia",
        "model": model,
        "vram_bytes": vram_bytes,
        "unified_memory": False,
    }


def _detect_gpu_nvidia_ctypes() -> dict | None:
    """ctypes fallback for NVIDIA when ``nvidia-smi`` is unavailable.

    Uses the CUDA driver API via ``cuda_detect._load_libcuda``: ``cuInit`` +
    ``cuDeviceGet`` + ``cuDeviceGetName`` + ``cuDeviceTotalMem_v2`` — model and
    VRAM without the binary. Returns a gpu dict or ``None``.
    """
    try:
        libcuda = cuda_detect._load_libcuda()
    except OSError:
        logger.debug("libcuda not loadable for GPU probe")
        return None

    # Mirror cuda_detect.detect_cuda_driver_version: an exported
    # CUDA_VISIBLE_DEVICES="" or "-1" (common on shared/CI hosts) hides all
    # devices from cuDeviceGet, so a physically-present GPU would report as
    # gpu:null. Drop it for the duration of the probe and restore it after.
    saved_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if saved_visible is not None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        if libcuda.cuInit(0) != 0:
            return None

        device = ctypes.c_int()
        if libcuda.cuDeviceGet(ctypes.byref(device), 0) != 0:
            return None

        name_buf = ctypes.create_string_buffer(256)
        model = None
        if libcuda.cuDeviceGetName(name_buf, len(name_buf), device) == 0:
            decoded = name_buf.value.decode("utf-8", errors="replace").strip()
            model = decoded or None

        vram_bytes = None
        total = ctypes.c_size_t()
        total_mem = getattr(libcuda, "cuDeviceTotalMem_v2", None) or getattr(libcuda, "cuDeviceTotalMem", None)
        if total_mem is not None and total_mem(ctypes.byref(total), device) == 0:
            vram_bytes = int(total.value)

        if not model and vram_bytes is None:
            return None

        return {
            "vendor": "nvidia",
            "model": model,
            "vram_bytes": vram_bytes,
            "unified_memory": False,
        }
    except Exception:
        logger.debug("ctypes NVIDIA GPU probe failed", exc_info=True)
        return None
    finally:
        if saved_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_visible


def _detect_gpu_amd() -> dict | None:
    """Best-effort AMD GPU probe on Linux via ``rocm-smi``.

    Minimal by design — nulls are acceptable. Returns a gpu dict (vendor "amd")
    or ``None`` if ``rocm-smi`` is absent / unparseable.
    """
    import json

    output = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if not output:
        return None

    try:
        payload = json.loads(output)
    except (ValueError, TypeError):
        logger.debug("could not parse rocm-smi JSON", exc_info=True)
        return None

    if not isinstance(payload, dict) or not payload:
        return None

    # rocm-smi keys cards as "card0", "card1", ...; take the first card entry
    # that yields any data (a card missing the queried fields shouldn't mask a
    # later card that has them). Gate on the "card" prefix so a non-card
    # metadata block (e.g. a "system" dict) that iterates first isn't mistaken
    # for the GPU.
    model = None
    vram_bytes = None
    for key, card in payload.items():
        if not str(key).lower().startswith("card") or not isinstance(card, dict):
            continue
        card_model = None
        card_vram_bytes = None
        for field, value in card.items():
            lowered = field.lower()
            if card_model is None and "name" in lowered:
                card_model = str(value).strip() or None
            # Match the total-capacity key ("VRAM Total Memory (B)"), excluding
            # the usage key ("VRAM Total Used Memory (B)") which also contains
            # both "vram" and "total" and would otherwise understate capacity.
            if card_vram_bytes is None and "vram" in lowered and "total" in lowered and "used" not in lowered:
                try:
                    card_vram_bytes = int(value)
                except (ValueError, TypeError):
                    pass
        if card_model is not None or card_vram_bytes is not None:
            model = card_model
            vram_bytes = card_vram_bytes
            break

    # Report no GPU rather than a phantom all-None AMD block (which would spoof
    # GPU presence for routing decisions), matching the NVIDIA probes.
    if not model and vram_bytes is None:
        return None

    return {
        "vendor": "amd",
        "model": model,
        "vram_bytes": vram_bytes,
        "unified_memory": False,
    }


def _detect_gpu(system: str, machine: str, cpu: str | None) -> dict | None:
    """Resolve the GPU block, or ``None`` if no GPU is detected.

    Apple Silicon is unified-memory (the RAM figure IS the GPU budget). Every
    other path tries NVIDIA (nvidia-smi then ctypes), then AMD.
    """
    try:
        if system == "Darwin" and machine == "arm64":
            return {
                "vendor": "apple",
                "model": cpu,
                "vram_bytes": None,
                "unified_memory": True,
            }

        gpu = _detect_gpu_nvidia_smi()
        if gpu is not None:
            return gpu

        gpu = _detect_gpu_nvidia_ctypes()
        if gpu is not None:
            return gpu

        if system == "Linux":
            gpu = _detect_gpu_amd()
            if gpu is not None:
                return gpu

        return None
    except Exception:
        logger.debug("GPU detection failed", exc_info=True)
        return None


def _detect_ram_bytes() -> int | None:
    try:
        return int(psutil.virtual_memory().total)
    except Exception:
        logger.debug("RAM detection failed", exc_info=True)
        return None


def detect_hardware() -> dict:
    """Detect generation-relevant hardware. Never raises.

    Returns a dict shaped like ``schemas/env.json``'s ``hardware`` block:
    ``os``/``os_version``/``arch``/``cpu``/``ram_bytes`` plus a ``gpu`` sub-dict
    (or ``None``). Any probe that fails contributes ``None`` rather than raising.
    """
    try:
        system = platform.system()
    except Exception:
        system = ""
    try:
        os_version = platform.release()
    except Exception:
        os_version = None
    try:
        machine = platform.machine()
    except Exception:
        machine = ""

    cpu = _detect_cpu(system)

    return {
        "os": system or None,
        "os_version": os_version or None,
        "arch": machine or None,
        "cpu": cpu,
        "ram_bytes": _detect_ram_bytes(),
        "gpu": _detect_gpu(system, machine, cpu),
    }
