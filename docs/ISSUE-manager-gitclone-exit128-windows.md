# 이슈: `progress=GitProgress()`가 Windows에서 파이프 데드락 → git clone exit 128

**컴포넌트**: `comfyui_manager/common/git_helper.py` — `gitclone()`
**근본 원인**: **확인됨** — tqdm progress 콜백이 Windows에서 git 서브프로세스 파이프를 데드락시킴
**심각도**: Medium — Windows에서 URL 기반 노드 설치 실패. 이름 기반 설치는 정상
**영향 범위**: Manager 4.1b6–b7
**증거**: comfy-cli PR #363 CI — 진단 커밋 `8fd463b8`

## 근본 원인 (CI 진단으로 확인)

`git.Repo.clone_from(url, repo_path, recursive=True, progress=GitProgress())`는 tqdm 기반 progress 콜백으로 git의 stderr에서 진행률을 읽습니다. Windows에서 상위 프로세스(`execute_cm_cli`)가 `subprocess.PIPE`로 stdout을 파이프하면, 파이프 버퍼링이 **데드락**을 일으킵니다: git은 stderr 쓰기에서 블로킹, tqdm은 읽기에서 블로킹, 프로세스가 멈추고 exit 128로 종료.

**진단 증거** (커밋 `8fd463b8`):

첫 번째 실패 후 `progress=None`으로 재시도를 추가했습니다. 재시도도 실패했지만, **다른 에러**가 나왔습니다:

```
fatal: destination path '...\nodepack-test1-do-not-install' already exists and is not an empty directory.
```

이것이 증명하는 것:
1. **첫 번째 clone은 시작됨** (디렉토리 생성) → **파이프 데드락으로 중단** → exit 128
2. `shutil.rmtree(repo_path, ignore_errors=True)`가 **Windows 파일 잠금으로 삭제 실패** (무음)
3. **재시도 실패** — 1단계의 잔여 디렉토리가 남아있기 때문

## 수정 방안

`git_helper.py`의 `gitclone()`에서 Windows(또는 stdout이 TTY가 아닌 경우) `progress=None` 사용:

```python
def gitclone(custom_nodes_path, url, target_hash=None, repo_path=None):
    repo_name = os.path.splitext(os.path.basename(url))[0]
    if repo_path is None:
        repo_path = os.path.join(custom_nodes_path, repo_name)

    use_progress = sys.stdout.isatty()
    progress = GitProgress() if use_progress else None
    repo = git.Repo.clone_from(url, repo_path, recursive=True, progress=progress)
    ...
```

`f87d2513`에서 동일한 접근(tqdm isatty 게이트)을 시도했지만, 다른 코드 경로까지 변경해서 reinstall regression이 발생했습니다. **수정은 `progress=` 파라미터에만 한정**해야 합니다.

## Windows에서만 발생하는 이유

`manager_core.py:1335`에 Windows 전용 코드 경로가 있습니다:

```python
if not instant_execution and platform.system() == 'Windows':
    res = manager_funcs.run_script([sys.executable, context.git_script_path, "--clone", ...])
else:
    repo = git.Repo.clone_from(clone_url, repo_path, recursive=True, progress=GitProgress())
```

- **Windows**: clone이 `run_script()` → `subprocess.run()` → `git_helper.py` 서브프로세스 → `git.Repo.clone_from(progress=GitProgress())`를 거칩니다. 추가 서브프로세스 레이어가 파이프 체인에 하나 더 추가되어 데드락 가능성이 높아짐.
- **Linux/macOS**: clone이 프로세스 내에서 직접 호출됩니다. 서브프로세스 파이프 체인 없음, 데드락 없음.

## 버전 이력

| Manager | 에러 | 원인 | 실패 |
|---------|------|------|------|
| 4.1b5 | `ModuleNotFoundError` | `__init__.py` import | 3/18 |
| 4.1b6 | exit 128, stderr 없음 | 파이프 데드락 (tqdm) | 2/18 |
| 4.1b7 | exit 128, stderr 없음 | 파이프 데드락 (tqdm) | 2/18 |
| `8fd463b8` (진단) | `fatal: destination path already exists` | 확인: tqdm 데드락 + rmtree 실패 | 2/18 |

## 관련

- comfy-cli PR #363: feat: add ComfyUI-Manager v4 support
- Manager `fix/git-clone-windows-pipe` 브랜치: 진단 커밋들
- `f87d2513`: tqdm isatty 게이트 (reinstall regression → 되돌림)
- `58debee9`: stderr 출력 개선 (regression 없음, GitPython이 stderr 삼킴)
- `8fd463b8`: progress 없이 재시도 (근본 원인 확인)
