# Experiment Observations: DPO-VP Falsification

**Model:** Qwen2.5-0.5B-Instruct  
**Dataset:** GSM8K (7000 train / 500 eval)  
**Pod:** RunPod, RTX 4090 24GB, CUDA 12.4  
**Date:** 2026-06-17  

---

## Environment Notes

- Pod image had torch 2.4.1+cu124 pre-installed
- `pip install --upgrade transformers trl` pulled torch 2.12.0, transformers 5.11.0, trl 1.5.1
- torchvision 0.19.1 (built for torch 2.4.1) broke `AutoProcessor` import via transformers → had to `pip uninstall torchvision torchaudio`
- Final working stack: torch 2.12.0, transformers 5.11.0, trl 1.5.1, datasets 5.0.0

---

## Run Log

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

## Key Signal to Watch

**Squeezing effect:** both `chosen_logprob` AND `rejected_logprob` falling together during DPO training.  
- Healthy DPO: `rejected_logprob` falls, `chosen_logprob` stable → gap grows  
- Squeezing: both fall → gap collapses → model is forgetting how to produce either response type  

**Hypothesis:** for 0.5B, squeezing onset within round 1 or 2. For 7B+ (DPO-VP paper), reported stable for 3+ rounds.

---

## Findings

_TBD — fill in after run completes._
