# run on 8xA800
# make sure your current working directory is the root of the project

set -euo pipefail
set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/rerank_search/config"

train_files=/mnt/sh/mmvision/home/taoszhang/project/Vis-Reason/data/playground/infoseek_verl/train_data_r5_correct_text_only.json
test_files=/mnt/sh/mmvision/home/taoszhang/project/Vis-Reason/data/playground/infoseek_verl/test_data_2000_text_only.json

TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"
CHAT_TEMPLATE_PATH="$PROJECT_DIR/rerank_search/config/qwen_vl_tool_chat_template.jinja2"
[[ -s "$CHAT_TEMPLATE_PATH" ]] || { echo "Missing/empty chat template: $CHAT_TEMPLATE_PATH" >&2; exit 1; }
HF_MODEL_PATH=${HF_MODEL_PATH:-"/mnt/sh/mmvision/share/pretrained_models/Qwen3-VL-8B-Thinking"}
Experiment_name="qwen3-vl-8b-thinking_async-sgl-infoseek-text-search-multiturn-3-grpo"
# wandb api key
export WANDB_API_KEY=2ba0887d400849bdd96a2ad62fc5acf55947fe79
# proxy
export http_proxy=http://9.131.113.25:11113
export https_proxy=http://9.131.113.25:11113

# debug mode
# export RAY_DEBUG=1
# export HYDRA_FULL_ERROR=1

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.dataloader_num_workers=8 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    data.train_files="$train_files" \
    data.val_files="$test_files"  \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=15000 \
    data.max_response_length=1024 \
    data.prompt_key=prompt \
    data.image_key=images \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.custom_cls.path=$PROJECT_DIR/rerank_search/infoseek/infoseek_thinking_dataset.py \
    data.custom_cls.name=InfoseekRLHFDataset \
    custom_reward_function.path=$PROJECT_DIR/rerank_search/infoseek/infoseek_thinking_rm.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2000 \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.custom_chat_template_path="$CHAT_TEMPLATE_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl' \
    trainer.experiment_name=$Experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=20 \
    trainer.max_actor_ckpt_to_keep=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_epochs=2 $@ 
    
# > ./logs/${Experiment_name}.log 2>&1 < /dev/null

