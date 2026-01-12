#!/bin/bash

# Fail fast and surface the real error (especially important for compiled extensions like TransformerEngine)
set -euo pipefail

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}
TE_REF=${TE_REF:-v2.6}
RUN_TE_SMOKE_TEST=${RUN_TE_SMOKE_TEST:-1}

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    pip install "sglang[all]==0.5.2" --no-cache-dir && pip install torch-memory-saver --no-cache-dir
fi
pip install --no-cache-dir "vllm==0.11.0"

echo "2. install basic packages"
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler pytest py-spy pre-commit ruff tensorboard 

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.8.1 (cxx11abi=False)
# wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
pip install --no-cache-dir flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

pip install --no-cache-dir flashinfer-python==0.3.1


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    pip install "onnxscript==0.3.1"
    # Build tools required by TE (CMake is mandatory; Ninja is recommended)
    # NOTE: with `set -e`, don't use a "probe that exits non-zero" unless it's inside an if/! guard.
    if ! command -v cmake >/dev/null 2>&1; then
        echo "Missing build tool: cmake. Installing via pip..."
        pip install --no-cache-dir "cmake>=3.26"
    fi
    if ! command -v ninja >/dev/null 2>&1; then
        echo "Missing build tool: ninja. Installing via pip..."
        pip install --no-cache-dir ninja
    fi
    # IMPORTANT:
    # - Without --no-build-isolation, pip may build TE against a *different* torch in an isolated env,
    #   leading to runtime errors like "undefined symbol: c10::SymInt::maybe_as_int_slow_path".
    # - We force TE to compile against the *currently installed* torch in this environment.
    python3 -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)"
    NVTE_FRAMEWORK=pytorch pip3 install -v --no-build-isolation --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@${TE_REF}
#     if [ "${RUN_TE_SMOKE_TEST}" -eq 1 ]; then
#         echo "4.1 TransformerEngine smoke test (import + Linear.cuda())"
#         python3 - <<'PY'
# import torch
# import transformer_engine
# from transformer_engine.pytorch import Linear
# print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
# layer = Linear(16, 32).cuda()
# print("TE OK: Linear.cuda() worked")
# PY
#     fi
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
fi


echo "5. May need to fix opencv"
# IMPORTANT:
# - opencv-python>=4.12 requires numpy>=2, but megatron-core requires numpy<2.0.0.
# - To avoid breaking Megatron, pin opencv-python to <4.12 and keep numpy<2.
pip install --no-cache-dir "numpy<2.0.0" "opencv-python<4.12"
pip install --no-cache-dir opencv-fixer && python -c "from opencv_fixer import AutoFix; AutoFix()"
pip check


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    pip install nvidia-cudnn-cu12==9.10.2.21
fi

echo "Successfully installed all packages"