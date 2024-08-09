from itertools import cycle
from pathlib import Path
import pytest
import shutil

from comfy_cli.uv import DependencyCompiler
from comfy_cli import ui

hereDir = Path(__file__).parent.resolve()
testsDir = hereDir.parent.resolve()
temp = testsDir / "temp" / "test_uv"
shutil.rmtree(temp, ignore_errors=True)
temp.mkdir(exist_ok=True, parents=True)

@pytest.fixture
def mock_prompt_select(monkeypatch):
    mockChoices = [">=1.13.0", ">=2.0.0"]
    def _mock_prompt_select(*args, **kwargs):
        return mockChoices.pop(0)

    monkeypatch.setattr(ui, "prompt_select", _mock_prompt_select)

def test_compile(mock_prompt_select):
    depComp = DependencyCompiler(
        cwd=temp,
        reqFilesCore=[hereDir/"mock_requirements/core_reqs.txt"],
        reqFilesExt=[hereDir/"mock_requirements/x_reqs.txt", hereDir/"mock_requirements/y_reqs.txt"],
    )

    depComp.makeOverride()
    depComp.compileCorePlusExt()

if __name__ == "__main__":
    test_compile()
