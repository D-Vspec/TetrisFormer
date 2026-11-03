# Phase 4 — Reward model v1

**Goal:** a ~5M-param action-free transformer predicting next RAM state, good on held-out expert play
and fast enough to sit inside the RL loop.
**Depends on:** Phase 3 gate. **Plan refs:** §5.4, §5.5, Milestone 4.

## Checklist

### Model (Tokenization B)
- [ ] Tokens: value embedding (256) + address embedding (K) + time embedding (T), summed
- [ ] **Ask human:** intra-step factorization — block-causal (predict all K bytes of s_{t+1} from s_{≤t}) vs byte-autoregressive within a step. Record choice
- [ ] Decoder-only causal transformer; cross-entropy over 256 values per byte
- [ ] Loss excludes volatile-masked bytes
- [ ] Configs exist for d=128/4L (~1M) and d=256/6L (~5M); param counts printed and recorded

### Sanity
- [ ] Overfit a single batch to ~0 loss with the 1M config
- [ ] Shape/causality test: changing s_{t+1} does not change predictions for s_{t+1}
- [ ] Trivial baselines computed on val: copy-last-state NLL, per-address unigram NLL

### Training (main, d=256, 6L, T=16, K=256)
- [ ] Training config logged (lr schedule, batch, steps, warmup, weight decay, seed)
- [ ] Train/val NLL curves saved
- [ ] Throughput measured (tokens/s); if attention at T×K=4096 is too slow → **ask human**: T=8 or factorized attention
- [ ] Final checkpoint frozen + hashed + versioned (`checkpoints/rm_<game>_v1`)

### Evaluation
- [ ] Held-out (test) NLL beats copy-last-state and unigram baselines (margins recorded)
- [ ] Per-address NLL breakdown saved (which bytes are hard)
- [ ] Likelihood ordering: expert test trajectories > Phase 2 baseline agent > random policy (per-step mean log p)
- [ ] Correlation of per-step reward with true progress (x-distance/level) on test data recorded

### RL readiness
- [ ] Batched inference across parallel envs with KV cache; matches non-cached output numerically
- [ ] Reward latency per env step measured; RL steps/s estimated with reward in the loop
- [ ] If too slow for RL → **ask human** (Option A, smaller T, factorized attention, batching changes)

## Exit gate (human sign-off)
- [ ] Beats trivial baselines on held-out NLL
- [ ] Expert > baseline agent > random likelihood ordering holds
- [ ] Inference speed acceptable for Phase 5

## Notes / decisions
-
