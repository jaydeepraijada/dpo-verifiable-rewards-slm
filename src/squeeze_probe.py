import numpy as np
import torch
from transformers import TrainerCallback

from src.probe_utils import completion_stats


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
                c_stats = completion_stats(
                    model, self.tokenizer, item["prompt"], item["chosen"], self.original_model
                )
                r_stats = completion_stats(
                    model, self.tokenizer, item["prompt"], item["rejected"], self.original_model
                )
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


class PolicyProbe(TrainerCallback):
    """
    GRPO analogue of SqueezeProbe. GRPO has no chosen/rejected preference
    structure, so there's no "gap" to track — but entropy and KL-from-original-SFT
    on a fixed probe set are tracked the same way, so DPO-VP and GRPO can be
    compared on matched diagnostics rather than just final accuracy.
    """

    def __init__(
        self,
        probe_items: list[dict],
        tokenizer,
        log_every: int = 50,
        original_model=None,
    ):
        self.items = probe_items[:80]  # cap for speed
        self.tokenizer = tokenizer
        self.log_every = log_every
        self.original_model = original_model
        self.history: list[dict] = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every != 0 or model is None:
            return

        logprobs, entropies, kls = [], [], []
        model.eval()

        with torch.no_grad():
            for item in self.items:
                stats = completion_stats(
                    model, self.tokenizer, item["prompt"], item["completion"], self.original_model
                )
                if stats is None:
                    continue
                logprobs.append(stats["logprob"])
                entropies.append(stats["entropy"])
                if stats["kl"] is not None:
                    kls.append(stats["kl"])

        model.train()

        if not logprobs:
            return

        record = {
            "step": state.global_step,
            "logprob": float(np.mean(logprobs)),
            "entropy": float(np.mean(entropies)),
            "kl_from_sft": float(np.mean(kls)) if kls else None,
        }
        self.history.append(record)
        kl_str = f" | kl_sft={record['kl_from_sft']:+.4f}" if record["kl_from_sft"] is not None else ""
        print(
            f"  [policy] step={record['step']:4d} | "
            f"logprob={record['logprob']:+.3f} | "
            f"entropy={record['entropy']:.3f}"
            f"{kl_str}"
        )
