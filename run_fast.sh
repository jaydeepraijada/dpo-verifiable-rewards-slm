#!/bin/bash
# Fast falsification run: reduced scale + larger batches for HF generate throughput.
# Qwen2.5-0.5B on GSM8K. Expected wall-clock: ~2-2.5 hours total on RTX 4090.
#
# Scale reduced from run.sh (7000 -> 1200 train problems) because rollout
# generation uses HF .generate() not vLLM; 1200 problems still yields ~600
# pairs/round, plenty for the squeeze signal. gen_batch_size bumped 4 -> 32
# to saturate the GPU (was only at ~33% util / 3.8GB on a 24GB card).
set -e

echo "========================================"
echo " Step 1: DPO-VP (3 rounds, verifiable pairs) — FAST"
echo "========================================"
python -m experiments.dpo_vp \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir results/dpo_vp \
    --num_rounds 3 \
    --n_rollouts 8 \
    --max_new_tokens 400 \
    --gen_batch_size 32 \
    --train_batch_size 4 \
    --grad_accum 4 \
    --lr 5e-6 \
    --beta 0.1 \
    --train_size 1200 \
    --eval_size 300 \
    --probe_size 300 \
    --probe_every 25

echo ""
echo "========================================"
echo " Step 2: GRPO baseline (600 steps) — FAST"
echo "========================================"
python -m experiments.grpo_baseline \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir results/grpo \
    --num_steps 600 \
    --group_size 8 \
    --max_new_tokens 400 \
    --train_batch_size 2 \
    --grad_accum 4 \
    --lr 5e-6 \
    --eval_every 150 \
    --train_size 1200 \
    --eval_size 300

echo ""
echo "========================================"
echo " Done. Results in results/"
echo "  DPO-VP: results/dpo_vp/results.json"
echo "  GRPO:   results/grpo/results.json"
echo "========================================"
