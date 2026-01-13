

#ln -s /mnt/csp /mnt/sh
export http_proxy=http://9.131.113.25:11113
export https_proxy=http://9.131.113.25:11113

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_IB_TIMEOUT=24
export NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME=bond1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400
export TORCH_NCCL_ENABLE_MONITORING=0

export NCCL_TIMEOUT_MINS=30
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:8192
export DISABLE_VERSION_CHECK=1

export WANDB_API_KEY=2ba0887d400849bdd96a2ad62fc5acf55947fe79


pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
pip config set global.trusted-host mirrors.tencent.com

nvidia-smi

ray stop --force

PROJECT_DIR=/mnt/sh/mmvision/home/taoszhang/project/Vis-Reason/verl

cd ${PROJECT_DIR}

if [ "$RANK" = "0" ]; then
  echo "MASTER:${MASTER_ADDR}:6379"
  ray start --head --port=6379 --dashboard-port=8080 --node-ip-address=${MASTER_ADDR} --dashboard-host=0.0.0.0
else
  ray start --address="${MASTER_ADDR}:6379"
fi

