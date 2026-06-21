import random
import torch
from tqdm import tqdm
from src.data import make_prompt, extract_gt_answer, extract_model_answer, answers_match


def generate_rollouts(
    model,
    tokenizer,
    questions: list[str],
    n: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    batch_size: int = 4,
) -> list[list[str]]:
    """
    For each question, generate n completions.
    Returns list[list[str]] shape (len(questions), n).
    batch_size controls how many questions are processed together (each expanded x n).
    """
    model.eval()
    all_rollouts: list[list[str]] = []

    pbar = tqdm(
        range(0, len(questions), batch_size),
        desc=f"rollouts (n={n})",
        total=(len(questions) + batch_size - 1) // batch_size,
    )
    for i in pbar:
        batch_q = questions[i : i + batch_size]
        # Expand: [q0, q0, ..., q1, q1, ...] — n copies each
        expanded = [q for q in batch_q for _ in range(n)]
        prompts = [make_prompt(q, tokenizer) for q in expanded]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=384,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)

        for j in range(len(batch_q)):
            all_rollouts.append(decoded[j * n : (j + 1) * n])

    return all_rollouts


def score_rollouts(
    rollouts: list[list[str]], ground_truths: list[str]
) -> list[list[float]]:
    return [
        [
            1.0 if answers_match(extract_model_answer(r), extract_gt_answer(gt)) else 0.0
            for r in rollout_set
        ]
        for rollout_set, gt in zip(rollouts, ground_truths)
    ]


def construct_pairs(
    questions: list[str],
    rollouts: list[list[str]],
    scores: list[list[float]],
    tokenizer,
) -> tuple[list[dict], float]:
    """
    Build DPO preference pairs from rollouts scored by verifiable reward.
    Returns (pairs, pair_rate) where pair_rate = fraction of problems that
    had at least one correct AND one incorrect completion.
    """
    pairs = []

    for question, rollout_set, score_set in zip(questions, rollouts, scores):
        correct = [r for r, s in zip(rollout_set, score_set) if s > 0.5]
        incorrect = [r for r, s in zip(rollout_set, score_set) if s < 0.5]

        if not correct or not incorrect:
            continue

        pairs.append(
            {
                "prompt": make_prompt(question, tokenizer),
                "chosen": random.choice(correct),
                "rejected": random.choice(incorrect),
            }
        )

    pair_rate = len(pairs) / max(len(questions), 1)
    return pairs, pair_rate


def evaluate_pass_at_1(
    model,
    tokenizer,
    questions: list[str],
    solutions: list[str],
    batch_size: int = 32,
    max_new_tokens: int = 512,
) -> tuple[float, float]:
    """Greedy-decode pass@1 on the provided split. Returns (accuracy, avg_completion_tokens)."""
    model.eval()
    correct = 0
    seen = 0
    total_len = 0

    pbar = tqdm(
        range(0, len(questions), batch_size),
        desc="eval",
        total=(len(questions) + batch_size - 1) // batch_size,
    )
    for i in pbar:
        batch_q = questions[i : i + batch_size]
        batch_s = solutions[i : i + batch_size]
        prompts = [make_prompt(q, tokenizer) for q in batch_q]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=384,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        completions = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)

        for completion, solution in zip(completions, batch_s):
            seen += 1
            total_len += len(tokenizer(completion, add_special_tokens=False)["input_ids"])
            if answers_match(extract_model_answer(completion), extract_gt_answer(solution)):
                correct += 1
        pbar.set_postfix(pass_at_1=f"{correct / seen:.3f}", avg_len=f"{total_len / seen:.1f}")

    n = max(len(questions), 1)
    return correct / n, total_len / n
