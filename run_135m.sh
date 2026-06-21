#!/bin/bash
# 135M follow-up: same protocol as run_fast.sh's DPO-VP step, swapped to a much
# smaller model, to test the squeezing hypothesis more aggressively after the
# 0.5B run showed no squeezing (chosen flat, rejected falling, gap growing).
#
# Qwen2.5 has no 135M size, so this uses HuggingFaceTB/SmolLM2-135M-Instruct.
# Same hyperparams/scale as the 0.5B run for an apples-to-apples comparison.
#
# DPO-VP only (no GRPO baseline) — this is the cheap way to test the hypothesis;
# add a matched GRPO run later only if this result is interesting enough to follow up.
#
# Built-in viability gate: the script evaluates base pass@1 and builds the squeeze
# probe set (~3-4 min) BEFORE the expensive 3-round loop starts. Watch the first
# few minutes of output — if probe pair_rate is very low or you see the
# "CRITICAL: probe set is nearly empty" warning, Ctrl+C now rather than letting
# the full ~35-40 min run finish on an uninformative setup (model too weak for
# GSM8K to construct correct/incorrect pairs).
set -e

echo "========================================"
echo " DPO-VP (3 rounds) — SmolLM2-135M-Instruct"
echo "========================================"
python -m experiments.dpo_vp \
    --model_name HuggingFaceTB/SmolLM2-135M-Instruct \
    --output_dir results/dpo_vp_135m \
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
echo " Done. Results in results/dpo_vp_135m/results.json"
echo "========================================"
