import torch
import numpy as np
from transformers import TrainerCallback


class SqueezeProbe(TrainerCallback):
    """
    Tracks mean per-token log-prob of chosen and rejected completions
    on a fixed probe set throughout DPO training.

    The squeezing effect manifests as both chosen_logprob AND rejected_logprob
    declining together — the model is "forgetting" how to produce either type
    of response. A healthy DPO run has rejected_logprob falling while
    chosen_logprob stays stable or rises.
    """

    def __init__(self, probe_pairs: list[dict], tokenizer, log_every: int = 50):
        self.pairs = probe_pairs[:80]  # cap for speed
        self.tokenizer = tokenizer
        self.log_every = log_every
        self.history: list[dict] = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every != 0 or model is None:
            return

        chosen_lps, rejected_lps = [], []
        model.eval()

        with torch.no_grad():
            for item in self.pairs:
                c_lp = self._mean_completion_logprob(model, item["prompt"], item["chosen"])
                r_lp = self._mean_completion_logprob(model, item["prompt"], item["rejected"])
                if c_lp is not None and r_lp is not None:
                    chosen_lps.append(c_lp)
                    rejected_lps.append(r_lp)

        model.train()

        if not chosen_lps:
            return

        record = {
            "step": state.global_step,
            "chosen_logprob": float(np.mean(chosen_lps)),
            "rejected_logprob": float(np.mean(rejected_lps)),
            "gap": float(np.mean(chosen_lps) - np.mean(rejected_lps)),
        }
        self.history.append(record)
        print(
            f"  [squeeze] step={record['step']:4d} | "
            f"chosen={record['chosen_logprob']:+.3f} | "
            f"rejected={record['rejected_logprob']:+.3f} | "
            f"gap={record['gap']:+.3f}"
        )

    def _mean_completion_logprob(
        self, model, prompt: str, completion: str
    ) -> float | None:
        full = prompt + completion
        inputs = self.tokenizer(
            full, return_tensors="pt", truncation=True, max_length=768
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=384
        )["input_ids"]
        prompt_len = prompt_ids.shape[1]

        completion_len = inputs["input_ids"].shape[1] - prompt_len
        if completion_len <= 0:
            return None

        with torch.no_grad():
            logits = model(**inputs).logits  # (1, seq_len, vocab)

        log_probs = torch.log_softmax(logits[0], dim=-1)  # (seq_len, vocab)

        # Shift: logits at position t predict token at t+1
        # Completion tokens start at prompt_len in the full sequence
        completion_ids = inputs["input_ids"][0, prompt_len:]  # (completion_len,)
        lp_slice = log_probs[prompt_len - 1 : prompt_len - 1 + len(completion_ids)]
        token_lps = lp_slice.gather(1, completion_ids.unsqueeze(1)).squeeze(1)

        return float(token_lps.mean().cpu())
