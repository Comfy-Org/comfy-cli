import shutil
from pathlib import Path

import pytest

from comfy_cli import ui
from comfy_cli.constants import GPU_OPTION
from comfy_cli.uv import DependencyCompiler

hereDir = Path(__file__).parent.resolve()
reqsDir = hereDir / "mock_requirements"

# set up a temp dir to write files to
testsDir = hereDir.parent.resolve()
temp = testsDir / "temp" / "test_uv"
shutil.rmtree(temp, ignore_errors=True)
temp.mkdir(exist_ok=True, parents=True)


@pytest.fixture
def mock_prompt_select(monkeypatch):
    mockChoices = ["==1.13.0", "==2.0.0"]

    def _mock_prompt_select(*args, **kwargs):
        return mockChoices.pop(0)

    monkeypatch.setattr(ui, "prompt_select", _mock_prompt_select)


def test_compile(mock_prompt_select):
    depComp = DependencyCompiler(
        cwd=temp,
        gpu=GPU_OPTION.AMD,
        outDir=temp,
        reqFilesCore=[reqsDir / "core_reqs.txt"],
        reqFilesExt=[reqsDir / "x_reqs.txt", reqsDir / "y_reqs.txt"],
    )

    depComp.make_override()
    depComp.compile_core_plus_ext()

    with open(reqsDir / "requirements.compiled", "r") as known, open(temp / "requirements.compiled", "r") as test:
        # compare all non-commented lines in generated file vs reference file
        knownLines, testLines = [
            [line for line in known.readlines() if not line.strip().startswith("#")],
            [line for line in test.readlines() if not line.strip().startswith("#")],
        ]
        assert knownLines == testLines
