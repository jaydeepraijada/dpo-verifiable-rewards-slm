import torch


def completion_stats(
    model, tokenizer, prompt: str, completion: str, original_model=None
) -> dict | None:
    """
    Per-token logprob, entropy, and (if original_model given) KL(model || original_model)
    of `completion` under `model`, conditioned on `prompt`. Shared by SqueezeProbe (DPO)
    and PolicyProbe (GRPO) so both methods get matched diagnostics.
    """
    full = prompt + completion
    inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)["input_ids"]
    prompt_len = prompt_ids.shape[1]

    completion_len = inputs["input_ids"].shape[1] - prompt_len
    if completion_len <= 0:
        return None

    with torch.no_grad():
        logits = model(**inputs).logits[0]  # (seq_len, vocab)

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # Shift: logits at position t predict token at t+1
    completion_ids = inputs["input_ids"][0, prompt_len:]
    lp_slice = log_probs[prompt_len - 1 : prompt_len - 1 + len(completion_ids)]
    p_slice = probs[prompt_len - 1 : prompt_len - 1 + len(completion_ids)]
    token_lps = lp_slice.gather(1, completion_ids.unsqueeze(1)).squeeze(1)

    logprob = float(token_lps.mean().cpu())
    entropy = float((-(p_slice * lp_slice).sum(dim=-1)).mean().cpu())

    kl = None
    if original_model is not None:
        with torch.no_grad():
            ref_logits = original_model(**inputs).logits[0]
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        ref_lp_slice = ref_log_probs[prompt_len - 1 : prompt_len - 1 + len(completion_ids)]
        kl_per_token = (p_slice * (lp_slice - ref_lp_slice)).sum(dim=-1)
        kl = float(kl_per_token.mean().cpu())

    return {"logprob": logprob, "entropy": entropy, "kl": kl}
