# Windows 재현 가이드: git clone 파이프 데드락

실제 Windows 머신에서 `git_helper.py`의 tqdm progress 콜백이 파이프 데드락을 일으키는지 재현하는 가이드.

## 사전 준비

- Python 3.10+ 설치됨
- git 설치됨
- pip, uv 사용 가능

## 재현 단계

### 1. comfy-cli 설치 (PR 브랜치)

```powershell
git clone -b feat/support-manager-v4 https://github.com/Comfy-Org/comfy-cli.git
cd comfy-cli
pip install -e .
```

### 2. ComfyUI workspace 생성

```powershell
mkdir C:\test-comfy
cd C:\test-comfy

comfy --skip-prompt --workspace C:\test-comfy\ws install --cpu
comfy --skip-prompt set-default C:\test-comfy\ws
```

### 3. Manager 버전 확인

```powershell
C:\test-comfy\ws\.venv\Scripts\python.exe -c "import importlib.metadata; print(importlib.metadata.version('comfyui-manager'))"
```

`comfy install`이 `manager_requirements.txt`에서 설치한 버전이 나옴. 어떤 버전이든 상관없음 — 버그는 Manager의 `git_helper.py`에서 `progress=GitProgress()`를 사용할 때 발생하며, 이 코드는 4.1b5 이후 모든 버전에 동일하게 존재.

### 4. 버그 재현 — URL 기반 노드 설치

```powershell
comfy --workspace C:\test-comfy\ws node install https://github.com/ltdrdata/nodepack-test1-do-not-install
```

**예상 결과 (버그):**
```
Download: git clone 'https://github.com/ltdrdata/nodepack-test1-do-not-install'
Install(git-clone) error[2]: ...
Cmd('git') failed due to: exit code(128)
```

**정상 결과 (버그 없음):**
```
Download: git clone 'https://github.com/ltdrdata/nodepack-test1-do-not-install'
Installation was successful.
```

### 5. 대조군 — 이름 기반 설치 (정상 동작해야 함)

```powershell
comfy --workspace C:\test-comfy\ws node install comfyui-impact-pack
```

이것은 `git_helper.py`를 거치지 않으므로 항상 성공해야 함.

### 6. 직접 git_helper.py 실행 (파이프 없이)

```powershell
# 잔여 디렉토리 정리
rmdir /s /q C:\test-comfy\ws\custom_nodes\nodepack-test1-do-not-install 2>nul

# git_helper.py 직접 실행 (파이프 없음 → 성공해야 함)
set COMFYUI_PATH=C:\test-comfy\ws
C:\test-comfy\ws\.venv\Scripts\python.exe C:\test-comfy\ws\.venv\Lib\site-packages\comfyui_manager\common\git_helper.py --clone C:\test-comfy\ws\custom_nodes https://github.com/ltdrdata/nodepack-test1-do-not-install C:\test-comfy\ws\custom_nodes\nodepack-test1-do-not-install
```

이것이 성공하면 → 파이프 체인이 원인임을 확인.

### 7. 파이프 시뮬레이션 (데드락 재현)

```powershell
# 잔여 디렉토리 정리
rmdir /s /q C:\test-comfy\ws\custom_nodes\nodepack-test1-do-not-install 2>nul

# stdout을 파이프로 연결 (CI와 동일 조건)
set COMFYUI_PATH=C:\test-comfy\ws
C:\test-comfy\ws\.venv\Scripts\python.exe C:\test-comfy\ws\custom_nodes\nodepack-test1-do-not-install C:\test-comfy\ws\.venv\Lib\site-packages\comfyui_manager\common\git_helper.py --clone C:\test-comfy\ws\custom_nodes https://github.com/ltdrdata/nodepack-test1-do-not-install C:\test-comfy\ws\custom_nodes\nodepack-test1-do-not-install | more
```

`| more`로 stdout을 파이프하면 데드락이 재현될 수 있음.

### 8. progress=None으로 패치 테스트

git_helper.py를 수정하여 progress 없이 테스트:

```powershell
# git_helper.py 위치
C:\test-comfy\ws\.venv\Lib\site-packages\comfyui_manager\common\git_helper.py
```

`gitclone()` 함수에서:
```python
# 변경 전
repo = git.Repo.clone_from(url, repo_path, recursive=True, progress=GitProgress())

# 변경 후
repo = git.Repo.clone_from(url, repo_path, recursive=True)
```

수정 후 4단계를 다시 실행하여 성공하는지 확인.

## 정리

```powershell
comfy --workspace C:\test-comfy\ws stop
rmdir /s /q C:\test-comfy
```

## 예상 결과 요약

| 테스트 | 예상 |
|--------|------|
| 4. URL 설치 (`comfy node install {URL}`) | **FAIL** — exit 128 |
| 5. 이름 설치 (`comfy node install {name}`) | PASS |
| 6. git_helper.py 직접 실행 | PASS |
| 7. git_helper.py 파이프 시뮬레이션 (`\| more`) | **FAIL** — 데드락 |
| 8. progress=None 패치 후 URL 설치 | PASS |

4, 7이 실패하고 6, 8이 성공하면 근본 원인이 tqdm progress 파이프 데드락임이 확정됩니다.
