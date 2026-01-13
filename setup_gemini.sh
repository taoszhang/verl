cd /mnt/sh/mmvision/home/taoszhang/project/Vis-Reason/verl

export http_proxy=http://9.131.113.25:11113
export https_proxy=http://9.131.113.25:11113
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

pip install --upgrade pip
cd apex 
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..
# Avoid PEP517 build isolation downloading build deps from index (which may be flaky/misconfigured).
pip install --no-deps --no-build-isolation -e .
# 修复qwen3_vl找不到的问题
pip install --upgrade transformers
pip uninstall mbridge -y
pip install git+https://github.com/ISEEKYAN/mbridge.git@qwen3vl_cp
# vllm引擎依赖numpy<=2.2.x
pip install numpy==2.2.0
# for debug
pip install "debugpy==1.8.0"
pip install ipdb