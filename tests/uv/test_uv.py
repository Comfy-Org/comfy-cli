from pathlib import Path
import pytest
import shutil

from comfy_cli.uv import DependencyCompiler
from comfy_cli import ui

hereDir = Path(__file__).parent.resolve()
reqsDir = hereDir/"mock_requirements"

# set up a temp dir to write files to
testsDir = hereDir.parent.resolve()
temp = testsDir/"temp"/"test_uv"
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
        reqFilesCore=[reqsDir/"core_reqs.txt"],
        reqFilesExt=[reqsDir/"x_reqs.txt", reqsDir/"y_reqs.txt"],
    )

    DependencyCompiler.Install_Build_Deps()
    depComp.make_override()
    depComp.compile_core_plus_ext()

    with open(reqsDir/"requirements.compiled", "r") as known, open(temp/"requirements.compiled", "r") as test:
        # compare all non-commented lines in generated file vs reference file
        knownLines, testLines = [
            [line for line in known.readlines() if line.strip()[0]!="#"],
            [line for line in test.readlines() if line.strip()[0]!="#"],
        ]
        assert knownLines == testLines
