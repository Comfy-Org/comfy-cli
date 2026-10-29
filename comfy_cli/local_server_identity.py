"""Bind inventory metadata to a direct endpoint and, when provable, a local process.

No server-supplied path establishes filesystem provenance. Process inspection is
read-only, and inability to inspect an owner leaves metadata useful but unverified.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from comfy_cli.command.stop_port import Listener, _is_python_ish, _proc_net_connections
from comfy_cli.http import NoRedirectHandler, build_http_only_opener, read_capped

_MAX_BYTES = 64 * 1024 * 1024
_PROCESS_ERRORS = (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError, NotImplementedError)


@dataclass(frozen=True)
class ServerDocument:
    data: Any
    filesystem_verified: bool = False


def _ip(value: str):
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address):
        return address.ipv4_mapped or address
    return address


def _is_local_address(address: str) -> bool:
    candidate = _ip(address)
    if candidate.is_loopback:
        return True
    try:
        return any(
            candidate == _ip(item.address)
            for addresses in psutil.net_if_addrs().values()
            for item in addresses
            if item.family in (socket.AF_INET, socket.AF_INET6)
        )
    except (psutil.Error, OSError, ValueError):
        return False


def _listener(address: str, port: int) -> Listener | None:
    """Find exactly one owner of this local address/port, never a remote namesake."""
    if not _is_local_address(address):
        return None
    candidate = _ip(address)
    owners = {}
    try:
        processes = list(psutil.process_iter(["pid"]))
    except _PROCESS_ERRORS:
        return None
    for candidate_process in processes:
        try:
            # process_iter caches Process instances (including create_time).
            # A fresh instance per scan must observe PID reuse across the GET.
            proc = psutil.Process(candidate_process.pid)
            for conn in _proc_net_connections(proc):
                bound = conn.laddr
                if conn.status != psutil.CONN_LISTEN or not bound or bound.port != port:
                    continue
                ip = _ip(bound.ip)
                if ip != candidate and not (ip.is_unspecified and ip.version == candidate.version):
                    continue
                owners[proc.pid] = Listener(
                    proc.pid,
                    proc.cmdline(),
                    proc.cwd(),
                    proc.name(),
                    bound.ip,
                    proc.create_time(),
                )
        except _PROCESS_ERRORS:
            # A hidden potential owner is not positive evidence. Other matching
            # visible owners are still considered; no system-wide privileged API.
            continue
    return next(iter(owners.values())) if len(owners) == 1 else None


def _same_filesystem_namespace(pid: int) -> bool:
    if sys.platform.startswith("linux"):
        try:
            return Path(f"/proc/{pid}/ns/mnt").samefile("/proc/self/ns/mnt") and Path(f"/proc/{pid}/root").samefile("/")
        except OSError:
            return False
    return sys.platform in ("darwin", "win32")


def _python_script_argv(cmdline: list[str]) -> list[str] | None:
    """The actual script entrypoint, not a later main.py passed as data."""
    if not cmdline or not _is_python_ish(cmdline[0]):
        return None
    index = 1
    while index < len(cmdline):
        token = cmdline[index]
        if token == "--":
            index += 1
            break
        if token in ("-u", "-B", "-E", "-I", "-s", "-S", "-O", "-OO", "-q", "-v", "-b", "-bb", "-d", "-R"):
            index += 1
        elif token in ("-X", "-W"):
            index += 2
        elif token.startswith(("-X", "-W")) and len(token) > 2:
            index += 1
        else:
            break
    if index >= len(cmdline) or os.path.basename(cmdline[index]) != "main.py":
        return None
    return cmdline[index:]


def _verified(stats: Any, before: Listener | None, after: Listener | None) -> bool:
    if before is None or before.create_time is None or before != after:
        return False
    if not _same_filesystem_namespace(before.pid):
        return False
    system = stats.get("system") if isinstance(stats, dict) else None
    if (
        not isinstance(system, dict)
        or not isinstance(system.get("comfyui_version"), str)
        or not system["comfyui_version"]
    ):
        return False
    argv = system.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        return False
    process_argv = _python_script_argv(before.cmdline)
    if process_argv is None or argv[1:] != process_argv[1:]:
        return False
    script = process_argv[0]
    if not os.path.isabs(script):
        if not before.cwd or not os.path.isabs(before.cwd):
            return False
        script = os.path.join(before.cwd, script)
    reported = argv[0]
    if not os.path.isabs(reported):
        if not before.cwd or not os.path.isabs(before.cwd):
            return False
        reported = os.path.join(before.cwd, reported)
    try:
        return Path(script).is_file() and Path(script).samefile(reported)
    except OSError:
        return False


def fetch_json(url: str, *, verify_filesystem: bool = False) -> ServerDocument:
    """Read one direct HTTP endpoint, pinning DNS once and preserving its Host header.

    Proxies and redirects cannot supply local filesystem proof. Address fallback
    still supports hostnames resolving to an unavailable IPv6 then a live IPv4
    endpoint. Metadata remains available when local process evidence is missing.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("inventory requires an uncredentialed http server address")
    port = parsed.port or 8188
    addresses = []
    for family, _, _, _, sockaddr in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM):
        address = sockaddr[0]
        if family == socket.AF_INET6 and sockaddr[3] and "%" not in address:
            address += f"%{sockaddr[3]}"
        if address not in addresses:
            addresses.append(address)
    opener = build_http_only_opener(urllib.request.ProxyHandler({}), NoRedirectHandler())
    last_error = None
    for address in addresses:
        # A wildcard bind address is commonly supplied as --server. Connect to
        # its own family's loopback, never treat an unspecified address as proof.
        ip = _ip(address)
        if ip.is_unspecified:
            address = "127.0.0.1" if ip.version == 4 else "::1"
        numeric_host = f"[{urllib.parse.quote(address, safe=':')}]" if ":" in address else address
        bound_url = urllib.parse.urlunsplit(("http", f"{numeric_host}:{port}", parsed.path, parsed.query, ""))
        before = _listener(address, port) if verify_filesystem else None
        request = urllib.request.Request(bound_url, headers={"Host": parsed.netloc})
        try:
            response = opener.open(request, timeout=10)
        except urllib.error.HTTPError:
            raise  # An HTTP reply is not a failed connection to try elsewhere.
        except OSError as error:
            last_error = error
            continue
        # Once this endpoint has answered, a body failure must not select a
        # different server from another DNS address.
        with response:
            data = json.loads(read_capped(response, url, max_bytes=_MAX_BYTES))
        after = _listener(address, port) if before is not None else None
        return ServerDocument(data, _verified(data, before, after))
    if last_error is not None:
        raise last_error
    raise OSError("server hostname resolved to no HTTP addresses")
