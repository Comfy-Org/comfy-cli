from pathlib import Path

from comfy_cli.uv import DependencyCompiler

testsDir = Path(__file__).parent.resolve()
temp = testsDir / "temp"
temp.mkdir(exist_ok=True)
here = Path(__file__).resolve()

depComp = DependencyCompiler(
    cwd=temp,
    reqFilesCore=[here / "mock_requirements/core_reqs.txt"],
    reqFilesExt=[here / "mock_requirements/x_reqs.txt", here / "mock_requirements/y_reqs.txt"],
)
