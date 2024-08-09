from pathlib import Path
import shutil

from comfy_cli.uv import DependencyCompiler

hereDir = Path(__file__).parent.resolve()
testsDir = hereDir.parent.resolve()
temp = testsDir / "temp" / "test_uv"
shutil.rmtree(temp, ignore_errors=True)
temp.mkdir(exist_ok=True, parents=True)

def test_compile():
    depComp = DependencyCompiler(
        cwd=temp,
        reqFilesCore=[hereDir/"mock_requirements/core_reqs.txt"],
        reqFilesExt=[hereDir/"mock_requirements/x_reqs.txt", hereDir/"mock_requirements/y_reqs.txt"],
    )

    depComp.makeOverride()
    depComp.compileCorePlusExt()

if __name__ == "__main__":
    test_compile()
