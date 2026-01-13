PROJECT_DIR=/mnt/sh/mmvision/home/taoszhang/project/Vis-Reason/verl
cd ${PROJECT_DIR}
nvidia-smi
which python3

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
pip list
WORLD_SIZE=$WORLD_SIZE   # 你可以 export WORLD_SIZE=4
RANK=${RANK}                  # 每个节点的唯一编号，0~(WORLD_SIZE-1)
SYNC_DIR=${PROJECT_DIR}/outputs/test/barrier

mkdir -p "$SYNC_DIR"

########################################
# 每个节点上，都启动一个localhost:8000的检索服务
########################################
echo "[RANK $RANK] Start retrieval server..."

file_path=/mnt/sh/mmvision/home/taoszhang/project/MMhops/Infoseek_multi_hop/search_engine/multihop3_100k_refine/croups
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/multi_hop_wiki_contents.jsonl
retriever_name=e5
retriever_path=/mnt/sh/mmvision/home/taoszhang/models/e5-base-v2

nohup python3 /mnt/sh/mmvision/home/taoszhang/project/Vis-Reason/verl/rerank_search/local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu \
    --port 8000 >> retrieval_server_${PORT}.log 2>&1 < /dev/null &
echo "Retrieval server started, waiting for 2 minutes to ensure it is ready..."
sleep 120

echo "[RANK $RANK] Step 1: Run bash1.sh"
bash rerank_search/multi_node_ray.sh

# 创建标志文件
touch "$SYNC_DIR/done_${RANK}"
echo "[RANK $RANK] Finished bash1.sh, waiting for others..."

# 等待所有节点完成
while true; do
    COUNT=$(ls "$SYNC_DIR"/done_* 2>/dev/null | wc -l)
    if [ "$COUNT" -ge "$WORLD_SIZE" ]; then
        echo "[RANK $RANK] All nodes finished pod initialization, proceeding to training..."
        break
    fi
    sleep 1
done

# 在主节点上删除临时同步文件夹
if [ "$RANK" -eq 0 ]; then
    sleep 2
    rm -r ${SYNC_DIR}
fi

bash rerank_search/multi_node_master_1.sh
