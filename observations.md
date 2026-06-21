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
| pair_rate | **0.670** (804/1200 — down slightly from Round 1's 0.717, still well above probe's 0.533) |
| avg_rollout_len | 274.3 tokens |
| pass@1 after round | **0.4633** (avg completion len 280.3) — slight dip from Round 1's 0.4733, still above 0.4200 base |
| chosen_logprob (step 25 → 50) | -0.206 → -0.206 (flat, ~same as Round 1's -0.203) |
| rejected_logprob (step 25 → 50) | -0.270 → -0.270 (flat, slightly more negative than Round 1's -0.262) |
| gap (step 25 → 50) | +0.064 → +0.064 (flat — grew vs. Round 1's +0.059, healthy direction) |
| entropy (step 25 → 50) | 0.283 → 0.282 (stable, ~same as Round 1) |
| kl_from_sft (step 25 → 50) | +0.0074 → +0.0076 (roughly doubled vs. Round 1's ~0.004, still tiny in absolute terms) |
| Squeezing observed? | **No** — chosen/rejected both flat within-round, gap grew round-over-round rather than collapsing. KL drift from SFT accumulating slowly but still small. |

### Round 3: DPO-VP

| Metric | Value |
|---|---|
| pair_rate | **0.657** (789/1200 — continued slow decline from 0.717→0.670→0.657, still well above probe's 0.533) |
| avg_rollout_len | 271.1 tokens |
| pass@1 after round | **0.4800** (avg completion len 283.4) — highest of all 3 rounds |
| chosen_logprob (step 25 → 50) | -0.208 → -0.208 (flat) |
| rejected_logprob (step 25 → 50) | -0.272 → -0.273 (flat) |
| gap (step 25 → 50) | +0.064 → +0.065 (flat within-round, continues the round-over-round growth) |
| entropy (step 25 → 50) | 0.280 → 0.281 (stable) |
| kl_from_sft (step 25 → 50) | +0.0096 → +0.0096 (continues monotonic growth: 0.004→0.008→0.0096) |
| Squeezing observed? | **No** — same flat-within-round, growing-across-rounds pattern as Rounds 1-2. |

---

## Cross-Round Summary (DPO-VP, all 3 rounds complete)

| Round | pass@1 | pair_rate | chosen_lp (end) | rejected_lp (end) | gap (end) | entropy (end) | kl_from_sft (end) |
|---|---|---|---|---|---|---|---|
| 0 (base) | 0.4200 | — | — | — | — | — | — |
| 1 | 0.4733 | 0.717 | -0.203 | -0.262 | +0.059 | 0.276 | 0.0042 |
| 2 | 0.4633 | 0.670 | -0.206 | -0.270 | +0.064 | 0.282 | 0.0076 |
| 3 | 0.4800 | 0.657 | -0.208 | -0.273 | +0.065 | 0.281 | 0.0096 |

**Verdict on the squeezing hypothesis (0.5B, GSM8K, 3 rounds DPO-VP): NOT supported.**

This is the textbook *healthy* DPO signature, not the squeezing failure mode:
- `chosen_logprob` stays nearly flat across all 3 rounds (-0.203 → -0.208, a drift of 0.005 over the whole run)
- `rejected_logprob` falls faster than chosen (-0.262 → -0.273, a drift of 0.011) — this is exactly the asymmetry squeezing is defined by the *absence* of
- the preference `gap` grows monotonically round over round (0.059 → 0.064 → 0.065), the opposite of collapse
- `entropy` stays essentially flat (0.276 → 0.282 → 0.281) — no sign of the policy degenerating toward a narrow/deterministic distribution
- `kl_from_sft` grows monotonically (0.0042 → 0.0076 → 0.0096) but stays small in absolute terms (~0.01 nats/token) — the model is drifting from its starting point as expected with iterative training, not collapsing
- `pass@1` trends upward overall (0.42 → 0.473 → 0.463 → 0.48) with normal eval-noise wobble on a 300-problem set, ending at its highest value
- `pair_rate` declines slowly (0.717 → 0.670 → 0.657) but stays far above the danger zone — the model isn't running out of solvable problems or losing rollout diversity

At this scale and configuration (Qwen2.5-0.5B-Instruct, GSM8K, 3 rounds, ~1200 train problems, ~700-800 pairs/round), the original hypothesis — that sub-1B models hit the DPO squeezing effect earlier/harder than the 7B+ models reported in the DPO-VP paper — **does not hold**. The 0.5B model's 3-round trajectory looks like the same healthy pattern reported for larger models, not an accelerated collapse.

Caveats before generalizing this too far:
- Only 3 rounds were run; squeezing could still onset later (round 4+) — untested here
- Only one model size (0.5B) was tested; no within-experiment comparison to a larger model to confirm the *relative* claim (sub-1B vs 7B+), only an absolute one (0.5B itself doesn't squeeze in 3 rounds)
- `beta=0.1`, `lr=5e-6` are mid-range hyperparameters; squeezing sensitivity to beta/lr was not swept
- GRPO baseline comparison now complete (see below) — confirms DPO-VP isn't uniquely unstable vs. online RL at this scale

### GRPO Baseline (actual scale: 600 steps, eval every 150)

| Step | pass@1 | entropy | kl_from_sft |
|---|---|---|---|
| 0 (base) | 0.4200 | — | — |
| 150 | (not captured this session — see results/grpo/results.json on pod) | | |
| 300 | 0.4967 | 0.239 | +0.0142 |
| 450 | (not captured this session — see results/grpo/results.json on pod) | | |
| 600 | 0.4933 | 0.238 | +0.0146 |
| 600 (final eval, post-training) | 0.4933 | — | — |

`train_runtime`: 4484s (~74.7 min), matching the live ETA estimate closely.

**GRPO is as stable as DPO-VP was.** Between step 300 and 600, the policy probe is essentially
flat: logprob -0.242→-0.242, entropy 0.239→0.238, kl_from_sft 0.0142→0.0146. No entropy collapse,
no runaway KL drift — same clean picture as the DPO-VP side, just via a different (online RL)
optimization path.

**DPO-VP vs GRPO, final comparison:**

| | Base | Final | Δ |
|---|---|---|---|
| DPO-VP (3 rounds) | 0.4200 | 0.4800 | +0.0600 |
| GRPO (600 steps) | 0.4200 | 0.4933 | +0.0733 |

Both methods improve pass@1 by a comparable margin at this scale, with neither showing
degenerate optimization (DPO squeezing or GRPO entropy collapse). No evidence that DPO-VP is
uniquely unstable relative to online RL for this model/dataset/scale.

---

## Follow-up Attempt: SmolLM2-135M-Instruct (`run_135m.sh`)

Goal: push the capacity hypothesis harder by testing an even smaller model than
0.5B, using the same protocol (3-round DPO-VP, 1200 train / 300 eval / 300 probe).

### Base Model (Round 0)

| Metric | Value |
|---|---|
| pass@1 (GSM8K 300-eval, greedy) | **0.0300** (3%) |
| avg completion len | 217.3 tokens |

### Probe Set Construction

- 300 problems × 4 rollouts
- **probe pair_rate = 0.057** (17 pairs from 300 problems) — far below the 0.5B run's 0.533
- Script printed a "CRITICAL: probe set is nearly empty" warning but had no hard stop, so the run continued

### Round 1: DPO-VP

| Metric | Value |
|---|---|
| pair_rate | **0.125** (150/1200) |
| avg_rollout_len | 225.3 tokens |
| pass@1 after round | **0.0300** — unchanged from base |
| loss | 0.692 (≈ ln 2, i.e. no signal — equivalent to random preference) |
| rewards/accuracies | 0.467 (≈ coin flip) |
| Squeeze probe | **Empty** (`"round_1": []`) — the 80-pair probe set couldn't even compute, because at this pair_rate there weren't enough valid chosen/rejected completions to populate it |

### Round 2: DPO-VP — crashed

Crashed mid-rollout-generation (~66% through, 25/38 batches) during `model.generate()`.
Moot given Round 1 already showed no usable signal — did not investigate or resume.

### Verdict: not a data point against the hypothesis

This is **not a squeezing result** — it's a capability floor. SmolLM2-135M cannot
reliably solve GSM8K problems (3% pass@1) or even reliably *fail* them in a way that
produces a diverse rollout pool; most generations collapse to similar wrong answers,
so there's nothing for the verifier to contrast into chosen/rejected pairs. DPO-VP
(and any preference-based method) needs a model capable enough to sometimes get the
right answer — without that, there's no preference signal to learn from, regardless
of squeezing dynamics.

**Takeaway:** the squeezing-onset question is gated by a prior capability question.
Testing the hypothesis at a smaller scale requires either a much easier dataset
(simple arithmetic, not GSM8K word problems) or a model with at least ~15-20%
base pass@1 — 135M on GSM8K doesn't clear that bar. Concluding the experiment here
rather than chasing this further with a different dataset.

---

## Key Signal to Watch

**Squeezing effect:** both `chosen_logprob` AND `rejected_logprob` falling together during DPO training.  
- Healthy DPO: `rejected_logprob` falls, `chosen_logprob` stable → gap grows  
- Squeezing: both fall → gap collapses → model is forgetting how to produce either response type  

**Hypothesis:** for 0.5B, squeezing onset within round 1 or 2. For 7B+ (DPO-VP paper), reported stable for 3+ rounds.

---

## Findings

**Experiment concluded 2026-06-21.**

1. **Squeezing hypothesis not supported at 0.5B.** Across 3 rounds of DPO-VP on
   Qwen2.5-0.5B-Instruct/GSM8K, chosen logprob stayed flat, rejected logprob fell
   faster, and the preference gap grew monotonically (0.059 → 0.064 → 0.065). This
   is the textbook healthy-DPO signature, not squeezing. No entropy collapse, no
   runaway KL drift from SFT.
2. **GRPO is equally stable at this scale.** The online-RL baseline (600 steps)
   showed the same flat entropy / small-KL-drift picture and a comparable pass@1
   gain (+0.073 vs DPO-VP's +0.060). No evidence DPO-VP is uniquely unstable
   relative to online RL for this model/dataset/scale.
3. **Capacity, not squeezing, is the real failure mode at the small end.**
   SmolLM2-135M-Instruct couldn't even generate the diverse correct/incorrect
   rollouts needed to construct preference pairs on GSM8K (3% pass@1, pair_rate
   0.057-0.125). The question "does squeezing hit smaller models harder" never
   got tested at 135M — it was preempted by a capability floor.
4. **Net result: a clean negative result.** The original hypothesis (sub-1B models
   squeeze earlier/harder than the 7B+ models in the DPO-VP paper) does not hold
   at 0.5B, and going smaller doesn't let you test it further on this dataset —
   you hit "model too weak to solve the task" before you hit "model squeezes."
   That boundary — capacity floor arrives before squeezing onset, at least for
   GSM8K-scale reasoning — is itself the interesting finding.

Caveats (see also Cross-Round Summary above): only 3 rounds tested (squeezing
could onset later), only one viable model size tested (no within-experiment
larger-model comparison), hyperparameters not swept, single dataset.
