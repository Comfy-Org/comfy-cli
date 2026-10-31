"""CLI subprocess tests must inherit their own test store, never a user's store."""

import os
import subprocess
import sys

from comfy_cli.auth.store import secrets_path


def test_cli_subprocess_inherits_the_per_test_credential_destination():
    expected = str(secrets_path())
    child = subprocess.run(
        [sys.executable, "-c", "from comfy_cli.auth.store import secrets_path; print(secrets_path())"],
        env=dict(os.environ),
        text=True,
        capture_output=True,
        check=True,
    )
    assert child.stdout.strip() == expected
