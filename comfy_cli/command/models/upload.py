"""``comfy model upload`` — put one local model file into the Comfy Cloud model library.

The upload itself is ``assets_library.run_upload``, shared with ``comfy assets
library upload``. This command adds what an agent needs to act on a person's
approval: ``--workspace`` and ``--expected-sha256`` pin the approval to one
workspace and one set of bytes, and the upload starts only with ``--yes`` or
an answered prompt. A dry run reports what the person is asked to approve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from comfy_cli import tracking

app = typer.Typer()


@app.command(
    "upload",
    help="Upload one named .safetensors model file (up to 2 GB) to your Comfy Cloud model library, so a "
    "workflow run on Comfy Cloud can load it. Uploads only the file named; succeeds only once Comfy Cloud's "
    "hash of the stored file matches this file's. Run with --dry-run first to see its sha256 and whether "
    "Comfy Cloud already has it.",
)
@tracking.track_command("model")
def upload_command(
    ctx: typer.Context,
    file: Annotated[Path, typer.Argument(help="The .safetensors file to upload.", show_default=False)],
    folder: Annotated[
        str,
        typer.Option(
            "--type",
            help="Model folder to file it under (e.g. loras, checkpoints, vae); "
            "`comfy model list-folders --where cloud` lists them.",
        ),
    ],
    workspace_id: Annotated[
        str | None,
        typer.Option(
            "--workspace",
            show_default=False,
            help="The Comfy Cloud workspace id the upload was approved for (`comfy cloud status`). "
            "If this sign-in uploads into another, nothing is uploaded.",
        ),
    ] = None,
    expected_sha256: Annotated[
        str | None,
        typer.Option(
            "--expected-sha256",
            show_default=False,
            help="The sha256 a --dry-run reported. If the file's bytes no longer have it, nothing is uploaded.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Check the file, the folder, the workspace and the library, and report the file's sha256 and "
            "whether Comfy Cloud already has it; upload nothing.",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Upload without asking.")] = False,
    where: Annotated[str | None, typer.Option("--where", show_default=False)] = None,
):
    from comfy_cli.command.assets_library import run_upload

    run_upload(
        file,
        folder,
        where=where,
        dry_run=dry_run,
        command="model upload",
        workspace=workspace_id,
        expected_sha256=expected_sha256,
        yes=yes,
        ctx=ctx,
    )
