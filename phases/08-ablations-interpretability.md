# Phase 8 — Ablations & interpretability

**Goal:** show which components are necessary and explain the reward (and its hacks) at byte level.
**Depends on:** Phase 7 gate (ablations may run on SML only — ask human). **Plan refs:** §6.4, §6.5, §7.

## Checklist

### Ablations (same protocol, ≥3 seeds each)
- [ ] **Ask human:** which games each ablation runs on
- [ ] Mask vs no mask
- [ ] No progress term vs each progress variant (a, b, c, d)
- [ ] Tokenization A (one token per step) vs B vs C (diff tokens) — implement A and C here
- [ ] Context length T (e.g. 4 / 8 / 16)
- [ ] Model size (1M / 5M / 25M)
- [ ] TAS vs human data (only if both exist)
- [ ] Likelihood vs curiosity sign (reuse Phase 6 runs if protocol identical)
- [ ] Ablation table + plots saved

### Interpretability
- [ ] Per-address attribution (attention rollout and/or gradient × input) for reward on expert and agent trajectories
- [ ] Top-N reward-driving addresses per game mapped to RAM-map meanings where known
- [ ] Sanity: attribution method passes a randomized-model check (attributions change when weights are randomized)

### Reward hacking catalogue
- [ ] `notes/reward-hacking.md` consolidated: each behavior → addresses responsible → which component fixes it
- [ ] Short clips for each catalogued behavior

### Off-distribution analysis
- [ ] Measure how often agent states are unseen in expert data (e.g. unseen byte values per address, per-step NLL spikes)
- [ ] Relate off-distribution episodes to hacking behaviors

### Compute
- [ ] Measured (not estimated) params and per-step FLOPs/latency for policy and reward model → replaces §7 estimates

## Exit gate (human sign-off)
- [ ] Ablation table complete
- [ ] Byte-level explanation for every catalogued hacking behavior

## Notes / decisions
-
