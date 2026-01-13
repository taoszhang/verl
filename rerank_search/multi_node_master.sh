#!/bin/bash

export http_proxy=http://9.131.113.25:11113
export https_proxy=http://9.131.113.25:11113

# 生成时间戳
time_stamp=$(date +%Y%m%d%H%M)
exp_name="test"
JOB_ID="${exp_name}_${time_stamp}"

# 主节点提交任务
if [ $RANK = "0" ]; then
  export RAY_ADDRESS=${MASTER_ADDR}:6379

  # 尝试使用Ray提交作业
  echo "尝试使用Ray提交作业，ID: $JOB_ID"
  bash /mnt/sh/mmvision/home/taoszhang/project/Vis-Reason/verl/rerank_search/multinode_qwen3_vl-8b_think_infoseek_search_multiturn.sh
  sleep 60

else
  # 其它节点等待主节点完成后退出任务
  echo "等待主节点完成作业"
  sleep infinity
fi