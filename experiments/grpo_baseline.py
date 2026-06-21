"""
GRPO baseline: online RL with verifiable rewards on GSM8K.

Equivalent compute budget to the DPO-VP experiment (matched by wall-clock steps).
Run from project root:
    python -m experiments.grpo_baseline [--args]
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import (
    answers_match,
    extract_gt_answer,
    extract_model_answer,
    load_gsm8k,
    make_prompt,
)
from src.rollout import construct_pairs, evaluate_pass_at_1, generate_rollouts, score_rollouts
from src.squeeze_probe import PolicyProbe


def reward_fn(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    """
    Verifiable reward for GRPO. TRL passes dataset columns as kwargs,
    so ground_truth arrives from the dataset's ground_truth column.
    """
    return [
        1.0 if answers_match(extract_model_answer(c), extract_gt_answer(gt)) else 0.0
        for c, gt in zip(completions, ground_truth)
    ]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", default="results/grpo")
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--train_size", type=int, default=7000)
    p.add_argument("--eval_size", type=int, default=500)
    p.add_argument("--probe_size", type=int, default=300)
    p.add_argument("--probe_every", type=int, default=50)
    p.add_argument("--gen_batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading GSM8K...")
    train_data, test_data = load_gsm8k()
    train_q = list(train_data["question"])[: args.train_size]
    train_s = list(train_data["answer"])[: args.train_size]
    eval_q = list(test_data["question"])[: args.eval_size]
    eval_s = list(test_data["answer"])[: args.eval_size]

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Frozen snapshot for KL-from-SFT tracking, matched to dpo_vp.py's original_model.
    original_model = copy.deepcopy(model)
    for p in original_model.parameters():
        p.requires_grad_(False)
    original_model.eval()

    # ── Baseline eval ─────────────────────────────────────────────────────
    print("\n--- Evaluating base model ---")
    base_acc, base_len = evaluate_pass_at_1(
        model, tokenizer, eval_q, eval_s, max_new_tokens=args.max_new_tokens
    )
    print(f"  Base model pass@1: {base_acc:.4f}  (avg completion len: {base_len:.1f} tokens)")
    results = [{"step": 0, "pass_at_1": base_acc, "avg_eval_len": base_len}]

    # ── Build fixed probe set for entropy/KL tracking, matched to dpo_vp.py ───
    probe_cache_path = output_dir / "probe_items.json"
    if probe_cache_path.exists():
        print(f"\nLoading cached policy probe set from {probe_cache_path}...")
        with open(probe_cache_path) as f:
            probe_items = json.load(f)
    else:
        print(f"\nBuilding policy probe set from {args.probe_size} problems...")
        probe_rollouts = generate_rollouts(
            model, tokenizer, train_q[: args.probe_size],
            n=4, max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch_size,
        )
        probe_scores = score_rollouts(probe_rollouts, train_s[: args.probe_size])
        probe_pairs, _ = construct_pairs(
            train_q[: args.probe_size], probe_rollouts, probe_scores, tokenizer
        )
        # Flatten chosen+rejected into individual (prompt, completion) items —
        # GRPO has no preference structure, just a fixed set to probe entropy/KL on.
        probe_items = []
        for pair in probe_pairs:
            probe_items.append({"prompt": pair["prompt"], "completion": pair["chosen"]})
            probe_items.append({"prompt": pair["prompt"], "completion": pair["rejected"]})
        probe_items = probe_items[:80]
        with open(probe_cache_path, "w") as f:
            json.dump(probe_items, f)
    print(f"  Policy probe set: {len(probe_items)} items")

    # ── Dataset for GRPO ──────────────────────────────────────────────────
    # GRPOTrainer expects a "prompt" column; extra columns are passed to reward_fn
    train_dataset = Dataset.from_dict(
        {
            "prompt": [make_prompt(q, tokenizer) for q in train_q],
            "ground_truth": train_s,
        }
    )

    # ── Eval callback ─────────────────────────────────────────────────────
    from transformers import TrainerCallback

    cli = args  # capture before class definition — 'args' is shadowed inside callback

    class EvalCallback(TrainerCallback):
        def __init__(self):
            self.log: list[dict] = []

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % cli.eval_every != 0 or state.global_step == 0:
                return
            acc, avg_len = evaluate_pass_at_1(
                model, tokenizer, eval_q, eval_s,
                max_new_tokens=cli.max_new_tokens,
            )
            self.log.append({"step": state.global_step, "pass_at_1": acc, "avg_eval_len": avg_len})
            print(f"  [eval] step={state.global_step} pass@1={acc:.4f} avg_len={avg_len:.1f}")
            results.append({"step": state.global_step, "pass_at_1": acc, "avg_eval_len": avg_len})
            with open(output_dir / "results.json", "w") as f:
                json.dump({"accuracy": results, "policy_probe": policy_probe.history}, f, indent=2)

    eval_cb = EvalCallback()
    policy_probe = PolicyProbe(
        probe_items, tokenizer, log_every=args.probe_every, original_model=original_model
    )

    # ── GRPO training ─────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.num_steps,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.group_size,
        max_completion_length=args.max_new_tokens,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
        callbacks=[eval_cb, policy_probe],
    )

    print(f"\n--- GRPO training for {args.num_steps} steps ---")
    trainer.train()

    # ── Final eval ────────────────────────────────────────────────────────
    final_acc, final_len = evaluate_pass_at_1(
        model, tokenizer, eval_q, eval_s, max_new_tokens=args.max_new_tokens
    )
    print(f"\n  Final pass@1 (step {args.num_steps}): {final_acc:.4f}  (avg len: {final_len:.1f})")
    results.append({"step": args.num_steps, "pass_at_1": final_acc, "avg_eval_len": final_len})

    with open(output_dir / "results.json", "w") as f:
        json.dump({"accuracy": results, "policy_probe": policy_probe.history}, f, indent=2)
    print(f"Saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
