"""Output handling: --download templating, URL printing, binary response writes.

Templating tokens: ``{request_id}``, ``{index}``, ``{ext}``. A trailing ``/``
on the template means "use a default filename in this directory."
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path

import httpx
from rich import print as rprint

from comfy_cli.command.generate import client

_EXT_FROM_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/svg+xml": "svg",
}


def _ext_from_url(url: str) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix.lstrip(".").lower()
    return suffix or "png"


def _ext_from_response(resp: httpx.Response) -> str:
    ct = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if ct in _EXT_FROM_MIME:
        return _EXT_FROM_MIME[ct]
    guess = mimetypes.guess_extension(ct) or ""
    return guess.lstrip(".") or "bin"


def _resolve_template(template: str, request_id: str, index: int, ext: str) -> Path:
    if template.endswith(("/", "\\")) or Path(template).is_dir():
        # Directory shorthand.
        path = Path(template) / f"{request_id}_{index}.{ext}"
    else:
        path = Path(template.format(request_id=request_id, index=index, ext=ext))
    return path.expanduser()


def save_urls(urls: list[str], template: str, request_id: str) -> list[Path]:
    """Download each URL and save under the resolved template path. Returns saved paths."""
    saved: list[Path] = []
    for i, url in enumerate(urls):
        ext = _ext_from_url(url)
        dest = _resolve_template(template, request_id, i, ext)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = client.download_bytes(url)
        dest.write_bytes(data)
        saved.append(dest)
    return saved


def save_binary_response(resp: httpx.Response, template: str, request_id: str) -> Path:
    """Save a single binary response body (e.g. Stability returns image/* bytes)."""
    ext = _ext_from_response(resp)
    dest = _resolve_template(template, request_id, 0, ext)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


def print_urls(urls: list[str], request_id: str | None = None) -> None:
    if not urls:
        rprint("[yellow]No image URLs found in response. Pass --json to inspect.[/yellow]")
        return
    if request_id:
        rprint(f"[bold green]Request:[/bold green] {request_id}")
    rprint("[bold green]Outputs:[/bold green]")
    for url in urls:
        rprint(f"  {url}")


def print_json(body: dict | list | str) -> None:
    if isinstance(body, str):
        print(body)
        return
    print(json.dumps(body, indent=2, default=str))


def print_saved(paths: list[Path]) -> None:
    if not paths:
        return
    rprint("[bold green]Saved:[/bold green]")
    for p in paths:
        rprint(f"  {p}")
