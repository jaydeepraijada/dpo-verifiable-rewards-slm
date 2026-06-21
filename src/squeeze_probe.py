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

    If `original_model` is provided (a frozen deepcopy of the round-0 model,
    distinct from the per-round DPO ref_model), also tracks:
      - entropy: mean per-token entropy of the current policy's distribution
        over completion positions — collapse often shows up as entropy crashing.
      - kl_from_sft: mean per-token KL(current policy || original SFT model)
        over the same positions — measures drift from the starting point,
        independent of the round-to-round ref_model used in the DPO loss itself.
    """

    def __init__(
        self,
        probe_pairs: list[dict],
        tokenizer,
        log_every: int = 50,
        original_model=None,
    ):
        self.pairs = probe_pairs[:80]  # cap for speed
        self.tokenizer = tokenizer
        self.log_every = log_every
        self.original_model = original_model
        self.history: list[dict] = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every != 0 or model is None:
            return

        chosen_lps, rejected_lps, entropies, kls = [], [], [], []
        model.eval()

        with torch.no_grad():
            for item in self.pairs:
                c_stats = self._completion_stats(model, item["prompt"], item["chosen"])
                r_stats = self._completion_stats(model, item["prompt"], item["rejected"])
                if c_stats is None or r_stats is None:
                    continue
                chosen_lps.append(c_stats["logprob"])
                rejected_lps.append(r_stats["logprob"])
                entropies.append(c_stats["entropy"])
                entropies.append(r_stats["entropy"])
                if c_stats["kl"] is not None:
                    kls.append(c_stats["kl"])
                    kls.append(r_stats["kl"])

        model.train()

        if not chosen_lps:
            return

        record = {
            "step": state.global_step,
            "chosen_logprob": float(np.mean(chosen_lps)),
            "rejected_logprob": float(np.mean(rejected_lps)),
            "gap": float(np.mean(chosen_lps) - np.mean(rejected_lps)),
            "entropy": float(np.mean(entropies)),
            "kl_from_sft": float(np.mean(kls)) if kls else None,
        }
        self.history.append(record)
        kl_str = f" | kl_sft={record['kl_from_sft']:+.4f}" if record["kl_from_sft"] is not None else ""
        print(
            f"  [squeeze] step={record['step']:4d} | "
            f"chosen={record['chosen_logprob']:+.3f} | "
            f"rejected={record['rejected_logprob']:+.3f} | "
            f"gap={record['gap']:+.3f} | "
            f"entropy={record['entropy']:.3f}"
            f"{kl_str}"
        )

    def _completion_stats(
        self, model, prompt: str, completion: str
    ) -> dict | None:
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
            logits = model(**inputs).logits[0]  # (seq_len, vocab)

        log_probs = torch.log_softmax(logits, dim=-1)  # (seq_len, vocab)
        probs = log_probs.exp()

        # Shift: logits at position t predict token at t+1
        # Completion tokens start at prompt_len in the full sequence
        completion_ids = inputs["input_ids"][0, prompt_len:]  # (completion_len,)
        lp_slice = log_probs[prompt_len - 1 : prompt_len - 1 + len(completion_ids)]
        p_slice = probs[prompt_len - 1 : prompt_len - 1 + len(completion_ids)]
        token_lps = lp_slice.gather(1, completion_ids.unsqueeze(1)).squeeze(1)

        logprob = float(token_lps.mean().cpu())
        entropy = float((-(p_slice * lp_slice).sum(dim=-1)).mean().cpu())

        kl = None
        if self.original_model is not None:
            with torch.no_grad():
                ref_logits = self.original_model(**inputs).logits[0]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            ref_lp_slice = ref_log_probs[prompt_len - 1 : prompt_len - 1 + len(completion_ids)]
            # KL(current || original_sft) per token, summed over vocab
            kl_per_token = (p_slice * (lp_slice - ref_lp_slice)).sum(dim=-1)
            kl = float(kl_per_token.mean().cpu())

        return {"logprob": logprob, "entropy": entropy, "kl": kl}
