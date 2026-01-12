# run on 8xH20
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/rerank_search/config"

train_files=/mnt/sh/mmvision/home/taoszhang/project/Vis-Reason/data/playground/infoseek_verl/train_data_r5_correct_text_only.json
test_files=/mnt/sh/mmvision/home/taoszhang/project/Vis-Reason/data/playground/infoseek_verl/test_data_2000_text_only.json

TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"
HF_MODEL_PATH=${HF_MODEL_PATH:-"/mnt/sh/mmvision/share/pretrained_models/Qwen3-VL-8B-Instruct"}
Experiment_name="qwen3-vl-8b-instruct_function_rm-search-async-sgl-multi-w-searchtool-verify-n16"
# wandb api key
export WANDB_API_KEY=2ba0887d400849bdd96a2ad62fc5acf55947fe79
# proxy
export http_proxy=http://9.131.113.25:11113
export https_proxy=http://9.131.113.25:11113

# debug mode
export RAY_DEBUG=1

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    data.train_files="$train_files" \
    data.val_files="$test_files"  \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=3000 \
    data.prompt_key=prompt \
    data.image_key=images \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.custom_cls.path=$PROJECT_DIR/rerank_search/infoseek_dataset.py \
    data.custom_cls.name=InfoseekRLHFDataset \
    custom_reward_function.path=$PROJECT_DIR/rerank_search/reward_model/infoseek_rerank_search.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
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
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name='verl' \
    trainer.experiment_name=$Experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_epochs=1 $@ 
    
# > ./logs/${Experiment_name}.log 2>&1 < /dev/null

