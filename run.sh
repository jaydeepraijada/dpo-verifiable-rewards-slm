#!/bin/bash
# Falsification experiment: Qwen2.5-0.5B on GSM8K
# Run on a Linux GPU pod (RTX 4090 or A100 40GB).
# Expected total wall-clock: ~4-5 hours on RTX 4090.
set -e

pip install -q -r requirements.txt

echo "========================================"
echo " Step 1: DPO-VP (3 rounds, verifiable pairs)"
echo "========================================"
python -m experiments.dpo_vp \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir results/dpo_vp \
    --num_rounds 3 \
    --n_rollouts 8 \
    --max_new_tokens 512 \
    --gen_batch_size 4 \
    --train_batch_size 2 \
    --grad_accum 8 \
    --lr 5e-6 \
    --beta 0.1 \
    --train_size 7000 \
    --eval_size 500 \
    --probe_size 500 \
    --probe_every 50

echo ""
echo "========================================"
echo " Step 2: GRPO baseline (1000 steps)"
echo "========================================"
python -m experiments.grpo_baseline \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir results/grpo \
    --num_steps 1000 \
    --group_size 8 \
    --max_new_tokens 512 \
    --train_batch_size 1 \
    --grad_accum 8 \
    --lr 5e-6 \
    --eval_every 250 \
    --train_size 7000 \
    --eval_size 500

echo ""
echo "========================================"
echo " Done. Results in results/"
echo "  DPO-VP: results/dpo_vp/results.json"
echo "  GRPO:   results/grpo/results.json"
echo "========================================"
