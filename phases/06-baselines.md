# Phase 6 — Learned-reward baselines

**Goal:** fill the controlled comparison table (policy fixed, only reward varies) for SML.
**Depends on:** Phase 5 gate. **Plan refs:** §6.2, Milestone 6.

## Checklist

### Shared protocol
- [ ] Same RAM policy, PPO config, address set, env-step budget, and seeds as Phase 5 Run 3
- [ ] Same evaluation script and metrics for every row

### Curiosity
- [ ] r_t = −log p(s_{t+1} | s_{≤t}), same frozen model, same normalization
- [ ] With mask and without mask (noisy-TV check: does it chase RNG/timer bytes?)
- [ ] ≥3 seeds; results + byte-level notes on what it chases

### VIPER (frames)
- [ ] **Ask human:** official VIPER code vs reimplementation
- [ ] Render frames for the *same* expert trajectories (pixels used only for this baseline — note in paper)
- [ ] Video model trained to convergence on held-out frame NLL; size/compute recorded vs our model
- [ ] Reward = frame likelihood (+ VIPER's own extras if their method includes them; recorded)
- [ ] Policy still RAM-input
- [ ] ≥3 seeds

### Rank-Then-Act (optional)
- [ ] **Ask human** whether to include (depends on code availability from Phase 0)
- [ ] If yes: their progress reward, RAM policy, ≥3 seeds

### Table
- [ ] SML rows filled: hand-engineered, ours, curiosity, VIPER, (RTA) — mean ± std over seeds
- [ ] Sample-efficiency plot with all rows

## Exit gate (human sign-off)
- [ ] SML comparison table complete with ≥3 seeds per row

## Notes / decisions
-
