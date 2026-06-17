"""
DPO-VP falsification experiment: 3 rounds of iterative DPO with verifiable pairs.

Hypothesis: DPO squeezing effect onsets within 1-2 rounds for sub-1B models,
earlier than reported for 7B+ models in the DPO-VP paper (arXiv:2503.12854).

Run from project root:
    python -m experiments.dpo_vp [--args]
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
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import load_gsm8k
from src.rollout import construct_pairs, evaluate_pass_at_1, generate_rollouts, score_rollouts
from src.squeeze_probe import SqueezeProbe


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", default="results/dpo_vp")
    p.add_argument("--num_rounds", type=int, default=3)
    p.add_argument("--n_rollouts", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gen_batch_size", type=int, default=4)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--probe_every", type=int, default=50)
    p.add_argument("--train_size", type=int, default=7000)
    p.add_argument("--eval_size", type=int, default=500)
    p.add_argument("--probe_size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def dpo_round(
    model,
    tokenizer,
    ref_model,
    questions: list[str],
    solutions: list[str],
    probe_pairs: list[dict],
    round_idx: int,
    output_dir: Path,
    args,
) -> tuple[float, list[dict]]:
    """
    One round of DPO-VP. Returns (pair_rate, squeeze_history).
    Modifies model in-place.
    """
    print(f"\n--- Round {round_idx}: generating {args.n_rollouts} rollouts per problem ---")
    rollouts = generate_rollouts(
        model, tokenizer, questions,
        n=args.n_rollouts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch_size,
    )

    scores = score_rollouts(rollouts, solutions)
    pairs, pair_rate = construct_pairs(questions, rollouts, scores, tokenizer)

    print(f"  pair_rate={pair_rate:.3f}  ({len(pairs)} pairs from {len(questions)} problems)")
    if len(pairs) < 50:
        print("  WARNING: very few pairs — base model may be too weak. "
              "Consider using a subset of easier problems.")

    hf_dataset = Dataset.from_list(pairs)
    probe = SqueezeProbe(probe_pairs, tokenizer, log_every=args.probe_every)

    dpo_config = DPOConfig(
        output_dir=str(output_dir / f"round_{round_idx}"),
        num_train_epochs=1,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        loss_type="sigmoid",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_length=args.max_new_tokens + 384,
        max_prompt_length=384,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        callbacks=[probe],
    )
    trainer.train()

    return pair_rate, probe.history


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

    # ── Baseline eval ─────────────────────────────────────────────────────
    print("\n--- Evaluating base model ---")
    base_acc = evaluate_pass_at_1(
        model, tokenizer, eval_q, eval_s, max_new_tokens=args.max_new_tokens
    )
    print(f"  Base model pass@1: {base_acc:.4f}")

    results = [{"round": 0, "pass_at_1": base_acc, "pair_rate": None}]
    squeeze_history: dict[str, list[dict]] = {}

    # ── Build fixed probe set from base model rollouts ────────────────────
    print(f"\nBuilding squeeze probe set from {args.probe_size} problems...")
    probe_rollouts = generate_rollouts(
        model, tokenizer, train_q[: args.probe_size],
        n=4, max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch_size,
    )
    probe_scores = score_rollouts(probe_rollouts, train_s[: args.probe_size])
    probe_pairs, probe_rate = construct_pairs(
        train_q[: args.probe_size], probe_rollouts, probe_scores, tokenizer
    )
    probe_pairs = probe_pairs[:80]
    print(f"  Probe set: {len(probe_pairs)} pairs (probe pair_rate={probe_rate:.3f})")

    if len(probe_pairs) < 10:
        print("  CRITICAL: probe set is nearly empty. The base model likely can't "
              "solve enough GSM8K problems. Experiment will run but squeeze signal "
              "may not be reliable.")

    # ── Iterative DPO rounds ──────────────────────────────────────────────
    for round_idx in range(1, args.num_rounds + 1):
        # Snapshot current model as the reference for this round
        ref_model = copy.deepcopy(model)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()

        pair_rate, sq_hist = dpo_round(
            model, tokenizer, ref_model,
            train_q, train_s, probe_pairs,
            round_idx, output_dir, args,
        )

        # Free ref model memory before evaluation
        del ref_model
        torch.cuda.empty_cache()

        acc = evaluate_pass_at_1(
            model, tokenizer, eval_q, eval_s, max_new_tokens=args.max_new_tokens
        )
        print(f"  Round {round_idx} pass@1: {acc:.4f}")

        results.append({"round": round_idx, "pass_at_1": acc, "pair_rate": pair_rate})
        squeeze_history[f"round_{round_idx}"] = sq_hist

        # Save after each round so a crash doesn't lose data
        _save(output_dir, results, squeeze_history)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Results ===")
    for r in results:
        pr = f"  pair_rate={r['pair_rate']:.3f}" if r["pair_rate"] is not None else ""
        print(f"  Round {r['round']}: pass@1={r['pass_at_1']:.4f}{pr}")

    _save(output_dir, results, squeeze_history)
    print(f"\nSaved to {output_dir}/results.json")


def _save(output_dir: Path, results: list, squeeze: dict):
    with open(output_dir / "results.json", "w") as f:
        json.dump({"accuracy": results, "squeeze": squeeze}, f, indent=2)


if __name__ == "__main__":
    main()
