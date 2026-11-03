# Phase 2 — Hand-engineered baseline (Super Mario Land)

**Goal:** prove a RAM-input PPO policy can learn SML with a hand-coded reward. This is a hard gate —
if it fails, the RAM policy is the bottleneck and nothing downstream is interpretable.
**Depends on:** Phase 1 gate. **Plan refs:** §5.7, §6.2, Milestone 2.

## Checklist

### Hand-coded reward
- [ ] SML RAM map sourced (cite URL in `notes/ram-maps.md`): x-position, level/world, lives, death/game-over, timer
- [ ] Each address verified in the emulator (move right → x increases; die → lives decrements)
- [ ] Reward terms written in one function: Δx progress, level-clear bonus, death penalty (coefficients recorded)
- [ ] Unit check: scripted input sequence produces expected reward signs

### Policy input
- [ ] Baseline address set chosen: bytes that vary during random/no-op rollouts minus volatile mask, plus RAM-map addresses (K recorded)
- [ ] Note: final address set comes from expert data in Phase 3 → this baseline must be re-run with that set (tracked in Phase 3)
- [ ] Encoding: one-hot or bit-unpacked bytes (which one recorded); input dim recorded

### PPO
- [ ] MLP policy + separate critic head; architecture recorded
- [ ] PPO implementation chosen (single-file, CleanRL-style unless human says otherwise)
- [ ] Hyperparameters logged in a config file (lr, γ, λ, clip, epochs, minibatch, entropy coef, envs, rollout length)
- [ ] Logging: episode return, max x-distance, levels cleared, episode length, steps/s
- [ ] Every run logs seed + config + git commit

### Runs
- [ ] **Success threshold written here before running** (e.g. "clears 1-1 in ≥X% of eval episodes by Y env steps")
- [ ] Random-policy reference measured on the same metrics
- [ ] Smoke run (short) to verify learning curve moves
- [ ] Full run, ≥3 seeds
- [ ] Eval: deterministic/greedy policy over fixed number of episodes; results table below
- [ ] Save a short gameplay video of the best policy

### If it fails
- [ ] **Do not tune indefinitely.** After the agreed tuning budget, stop and report to human with curves → any change of policy/arch/reward is a deviation

## Exit gate (human sign-off)
- [ ] Pre-registered threshold met across ≥3 seeds
- [ ] Results table + curves saved

## Notes / decisions
-
