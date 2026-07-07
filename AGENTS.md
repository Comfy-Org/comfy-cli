# Agent Instructions For comfy-cli

## Critical Follow-Up: ComfyUI GGUF On ARM/aarch64

This workspace is used with a ComfyUI install at `/home/kkk/Apps/ComfyUI` on
ARM/aarch64 hardware with an NVIDIA GB10 GPU. If a ComfyUI workflow using
`ComfyUI_Simple_Qwen3-VL-gguf`, `llama_cpp`, or a Qwen/Qwen3.5 GGUF fails with:

```text
ValueError: Failed to load model from file: ...gguf
llama_prepare_model_devices: invalid value for main_gpu: 0 (available devices: 0)
```

do not assume the GGUF is missing or corrupt. First verify whether
`llama-cpp-python` was installed CPU-only. The observed failure came from a
valid GGUF and a CPU-only `llama_cpp` build that lacked `libggml-cuda.so`, while
the workflow requested GPU offload with `n_gpu_layers: -1` and `main_gpu: 0`.

Useful checks:

```bash
/home/kkk/Apps/ComfyUI/.venv/bin/python -c "import llama_cpp; print(llama_cpp.__version__)"
find /home/kkk/Apps/ComfyUI/.venv/lib/python*/site-packages/llama_cpp/lib \
  -name 'libggml-cuda*' -o -name '*cuda*'
```

Torch CUDA support is not the fix for GGUF inference. `llama-cpp-python` wraps
`llama.cpp`, so the CUDA backend must be enabled in the `llama-cpp-python`
source build. On ARM/aarch64, prebuilt wheels may be CPU-only or unavailable,
especially for Python 3.13.

Rebuild from the user's normal shell, in the ComfyUI venv:

```bash
cd /home/kkk/Apps/ComfyUI
source .venv/bin/activate

CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
python -m pip install --upgrade --force-reinstall --no-cache-dir \
  --no-binary llama-cpp-python llama-cpp-python
```

After rebuilding, confirm that `libggml-cuda.so` exists before retrying GPU
workflow settings such as `n_gpu_layers: -1` and `main_gpu: 0`.

Temporary CPU-only workaround:

```json
{
  "n_gpu_layers": 0,
  "main_gpu": -1
}
```
