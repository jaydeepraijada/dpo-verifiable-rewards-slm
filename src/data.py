import re
from datasets import load_dataset

SYSTEM_PROMPT = (
    "Solve the problem step by step. "
    "At the end write 'The answer is: X' where X is the final number."
)


def load_gsm8k():
    ds = load_dataset("openai/gsm8k", "main")
    return ds["train"], ds["test"]


def make_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_gt_answer(solution: str) -> str | None:
    m = re.search(r"####\s*([\-\d,\.]+)", solution)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def extract_model_answer(text: str) -> str | None:
    # Prefer explicit "The answer is: X" pattern
    m = re.search(r"[Tt]he answer is[:\s]+([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fall back to last standalone number
    nums = re.findall(r"(?<!\d)-?\d+(?:,\d{3})*(?:\.\d+)?(?!\d)", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def answers_match(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()
