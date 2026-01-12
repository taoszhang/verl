file_path=/mnt/sh/mmvision/home/taoszhang/project/MMhops/Infoseek_multi_hop/search_engine/multihop3_100k_refine/croups
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/multi_hop_wiki_contents.jsonl
retriever_name=e5
retriever_path=/mnt/sh/mmvision/home/taoszhang/models/e5-base-v2

python3 /mnt/sh/mmvision/home/taoszhang/project/Vis-Reason/verl/examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu \
    --port 8000

echo "Retrieval server started, waiting for 5 minutes to ensure it is ready..."
