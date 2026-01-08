cd /mnt/sh/mmvision/home/taoszhang/project/Vis-Reason/verl
# unset http_proxy
# unset https_proxy

yum install -y cuda-toolkit-12-8
rm -rf /usr/local/cuda
ln -s /usr/local/cuda-12.8 /usr/local/cuda
ls -l /usr/local/cuda
source ~/.bashrc
nvcc --version

yum install -y libcudnn9-devel-cuda-12.x86_64 --allowerasing
ls -l /usr/include/cudnn.h
ln -sf /usr/include/cudnn*.h /usr/local/cuda/include/
ln -sf /usr/lib64/libcudnn* /usr/local/cuda/lib64/
ls -l /usr/local/cuda/include/cudnn.h

# export http_proxy=http://9.131.113.25:11113
# export https_proxy=http://9.131.113.25:11113

conda create -n verl python==3.12 -y
conda activate verl

pip install --upgrade pip
# Ensure build backend deps exist in current env.
# Otherwise `pip install -e .` may fail while trying to fetch build deps from index (PEP517 build isolation).
pip install --upgrade "setuptools>=61.0" wheel
bash scripts/install_vllm_sglang_mcore_test.sh
pip install word2number
pip install codetiming
pip install faiss-gpu-cu12
# git clone https://github.com/NVIDIA/apex.git
cd apex 
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..
# Avoid PEP517 build isolation downloading build deps from index (which may be flaky/misconfigured).
pip install --no-deps --no-build-isolation -e .
pip install --upgrade transformers
pip uninstall mbridge -y
pip install git+https://github.com/ISEEKYAN/mbridge.git@qwen3vl_cp # for correct mbridge