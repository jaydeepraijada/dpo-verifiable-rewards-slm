# Experiment Observations: DPO-VP Falsification

**Model:** Qwen2.5-0.5B-Instruct  
**Dataset:** GSM8K  
**Pod:** RunPod, RTX 4090 24GB, CUDA 12.4  
**Date:** 2026-06-17 (initial setup) — 2026-06-21 (actual run, via `run_fast.sh`)

Note: the Base Model / Probe numbers below from 2026-06-17 used the original
full-scale config (7000 train / 500 eval) on a slow CPU-fallback run that was
killed before Round 1 finished. The run actually used for results (`run_fast.sh`,
2026-06-21) uses 1200 train / 300 eval / 300 probe problems — those numbers are
logged separately below and are the ones that matter for the squeeze analysis.

---

## Environment Notes

- Pod image had torch 2.4.1+cu124 pre-installed
- `pip install --upgrade transformers trl` pulled torch 2.12.0, transformers 5.11.0, trl 1.5.1
- torchvision 0.19.1 (built for torch 2.4.1) broke `AutoProcessor` import via transformers → had to `pip uninstall torchvision torchaudio`
- Final working stack: torch 2.12.0, transformers 5.11.0, trl 1.5.1, datasets 5.0.0

**Second pod (new restart):**
- `pip install -r requirements.txt` pulled torch 2.12.0+cu130 by default (PyPI default index), but this pod's driver (550.127.05) only supports up to CUDA 12.4 → `torch.cuda.is_available()` silently returned `False` and the run fell back to CPU (looked "stuck" — eval batch of 32 on CPU never finished)
- Diagnosed via `nvidia-smi` showing 0% GPU util / no process while `top` showed 100% CPU, then confirmed with `python -c "import torch; print(torch.cuda.is_available())"`
- Fix: reinstall from the CUDA-matched wheel index — `pip install --no-cache-dir --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` (the cu124 index didn't have 2.12.0; 2.6.0 was the newest available there)
- Lesson: always verify `torch.cuda.is_available()` is `True` right after install, on every new pod — driver versions vary and the default PyPI torch wheel targets the newest CUDA, not necessarily what the pod's driver supports
- Two trl 1.5.1 API breaks hit mid-run, both fixed: `DPOConfig` has no `max_prompt_length` (only `max_length`); `GRPOConfig` has no `max_new_tokens`/`max_prompt_length` (only `max_completion_length`)

---

## Run Log (original full-scale attempt, abandoned — config/scale superseded by run_fast.sh)

### Base Model (Round 0)

| Metric | Value |
|---|---|
| pass@1 (GSM8K 500-eval, greedy) | **0.4480** |
| GPU util during eval | ~20-25% |
| VRAM usage | ~1602 MiB |

- Base model is healthy — 44.8% is well above the ~25% threshold needed for viable pair construction
- Greedy eval on 500 problems ran silently (~5-8 min), no progress bar

### Probe Set Construction

- 500 problems × 4 rollouts
- **probe pair_rate = 0.552** (very healthy — >half of problems yielded both correct & incorrect rollouts)
- 80 pairs retained (capped) for squeeze tracking

---

### Round 1: DPO-VP

| Metric | Value |
|---|---|
| pair_rate | _TBD_ |
| pass@1 after round | _TBD_ |
| chosen_logprob (start → end) | _TBD_ |
| rejected_logprob (start → end) | _TBD_ |
| gap (start → end) | _TBD_ |
| Squeezing observed? | _TBD_ |

### Round 2: DPO-VP

| Metric | Value |
|---|---|
| pair_rate | _TBD_ |
| pass@1 after round | _TBD_ |
| chosen_logprob (start → end) | _TBD_ |
| rejected_logprob (start → end) | _TBD_ |
| gap (start → end) | _TBD_ |
| Squeezing observed? | _TBD_ |

### Round 3: DPO-VP

| Metric | Value |
|---|---|
| pair_rate | _TBD_ |
| pass@1 after round | _TBD_ |
| chosen_logprob (start → end) | _TBD_ |
| rejected_logprob (start → end) | _TBD_ |
| gap (start → end) | _TBD_ |
| Squeezing observed? | _TBD_ |

---

### GRPO Baseline

| Step | pass@1 |
|---|---|
| 0 (base) | 0.4480 |
| 250 | _TBD_ |
| 500 | _TBD_ |
| 750 | _TBD_ |
| 1000 | _TBD_ |

---

## Run Log (actual: `run_fast.sh`, 1200 train / 300 eval / 300 probe, 2026-06-21)

### Base Model (Round 0)

| Metric | Value |
|---|---|
| pass@1 (GSM8K 300-eval, greedy) | **0.4200** |

### Probe Set Construction

- 300 problems × 4 rollouts
- **probe pair_rate = 0.533** — healthy
- 80 pairs retained for squeeze tracking

### Round 1: DPO-VP

| Metric | Value |
|---|---|
| pair_rate | **0.717** (860/1200 — improved over probe's 0.533, model getting better at generating both correct & incorrect rollouts) |
| avg_rollout_len | 272.0 tokens |
| pass@1 after round | **0.4733** (avg completion len 282.1) — up from 0.4200 base |
| chosen_logprob (step 25 → 50) | -0.203 → -0.203 (flat) |
| rejected_logprob (step 25 → 50) | -0.261 → -0.262 (flat) |
| gap (step 25 → 50) | +0.058 → +0.059 (flat, no growth or collapse) |
| entropy (step 25 → 50) | 0.273 → 0.276 (stable) |
| kl_from_sft (step 25 → 50) | +0.0037 → +0.0042 (tiny, growing slowly) |
| Squeezing observed? | **No** — flat/stable trajectory within the round. Note: no step-0 pre-training probe reading exists (callback only fires on multiples of `probe_every`=25), so this is a within-round comparison, not vs. pre-round baseline. |

Training-batch stats (`logps/chosen` -53.7→-58.7, `logps/rejected` -75.8→-71.0 across logging steps)
moved in opposite directions from the fixed-probe-set readings — but those are raw per-minibatch sums
over whatever examples landed in that batch (length-confounded, not a fixed set), so they're not
directly comparable to the squeeze probe and shouldn't be over-interpreted.

### Round 2: DPO-VP

| Metric | Value |
|---|---|
| pair_rate | _TBD_ |
| pass@1 after round | _TBD_ |
| chosen_logprob (start → end) | _TBD_ |
| rejected_logprob (start → end) | _TBD_ |
| gap (start → end) | _TBD_ |
| entropy (start → end) | _TBD_ |
| kl_from_sft (start → end) | _TBD_ |
| Squeezing observed? | _TBD_ |

### Round 3: DPO-VP

| Metric | Value |
|---|---|
| pair_rate | _TBD_ |
| pass@1 after round | _TBD_ |
| chosen_logprob (start → end) | _TBD_ |
| rejected_logprob (start → end) | _TBD_ |
| gap (start → end) | _TBD_ |
| entropy (start → end) | _TBD_ |
| kl_from_sft (start → end) | _TBD_ |
| Squeezing observed? | _TBD_ |

### GRPO Baseline (actual scale: 600 steps, eval every 150)

| Step | pass@1 | entropy | kl_from_sft |
|---|---|---|---|
| 0 (base) | 0.4200 | — | — |
| 150 | _TBD_ | _TBD_ | _TBD_ |
| 300 | _TBD_ | _TBD_ | _TBD_ |
| 450 | _TBD_ | _TBD_ | _TBD_ |
| 600 | _TBD_ | _TBD_ | _TBD_ |

---

## Key Signal to Watch

**Squeezing effect:** both `chosen_logprob` AND `rejected_logprob` falling together during DPO training.  
- Healthy DPO: `rejected_logprob` falls, `chosen_logprob` stable → gap grows  
- Squeezing: both fall → gap collapses → model is forgetting how to produce either response type  

**Hypothesis:** for 0.5B, squeezing onset within round 1 or 2. For 7B+ (DPO-VP paper), reported stable for 3+ rounds.

---

## Findings

_TBD — fill in after run completes._
