# E2E Testing with Manager Override

ComfyUI-Manager 패치를 PyPI 등록이나 ComfyUI Core PR 없이 바로 Windows CI에서 검증하는 방법.

## 사용법

PR 코멘트에 `/test-manager` 명령어를 입력합니다.

### git branch에서 직접 테스트 (PyPI 불필요)

```
/test-manager @fix-branch
```

`Comfy-Org/ComfyUI-Manager`의 `fix-branch` 브랜치를 clone하여 workspace venv에 설치 후 E2E 테스트를 실행합니다.

### PyPI pre-release 버전 테스트

```
/test-manager 4.1b7
```

### 옵션

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| spec | O | - | `@branch` 또는 PyPI 버전 (예: `4.1b7`) |
| os | X | `windows-latest` | `ubuntu-latest`, `macos-latest`, `all` |
| test_filter | X | 전체 E2E | pytest `-k` 필터 |

### 전체 예시

```bash
# Windows만 (기본)
/test-manager @fix-windows

# 전체 OS
/test-manager @fix-windows all

# 특정 테스트만
/test-manager 4.1b7 windows-latest test_e2e_uv_compile
```

## 동작 방식

```
PR 코멘트 "/test-manager @branch"
  ↓
1. parse job: 코멘트 파싱 + 입력 검증 + PR SHA 추출
  ↓
2. test job (per OS):
   a. PR 코드 checkout (comfy-cli 변경사항 포함)
   b. comfy install → workspace venv에 기본 Manager 설치
   c. MANAGER_OVERRIDE 환경변수 → fixture에서 감지
   d. @branch: git clone → uv pip install (workspace venv에 덮어쓰기)
      version: pip install comfyui-manager==version --pre
   e. E2E 테스트 실행
  ↓
3. PR commit status에 결과 표시
```

## 제약 사항

- **첫 사용**: 이 workflow(`test-manager-override.yml`)가 main 브랜치에 머지된 후에만 `/test-manager` 코멘트가 트리거됩니다. `issue_comment` 이벤트는 default branch의 workflow만 실행합니다.
- **git branch 설치**: `uv pip install`을 사용합니다. Manager repo의 flat-layout이 `pip install`과 호환되지 않기 때문입니다. CI 환경에 `uv`가 필요합니다 (`comfy install`이 자동으로 설치).
- **repo 고정**: `Comfy-Org/ComfyUI-Manager` repo만 지원. 다른 fork는 지원하지 않습니다.

## 관련 파일

| 파일 | 역할 |
|------|------|
| `.github/workflows/test-manager-override.yml` | PR 코멘트 트리거 workflow |
| `tests/e2e/test_e2e_uv_compile.py` | `MANAGER_OVERRIDE` 환경변수 처리 (workspace fixture) |
| `.github/workflows/build-and-test.yml` | 기존 E2E (변경 없음) |
