# Phase 5 — RL with likelihood reward

**Goal:** an agent trained only on the frozen reward model's likelihood makes real progress in SML.
Expect and document failure modes first, then fix with mask + progress term.
**Depends on:** Phase 2 and Phase 4 gates. **Plan refs:** §5.6, §6.2, §6.5, Milestone 5.

## Checklist

### Reward plumbing
- [ ] r_t = Σ over unmasked bytes of log p(s_{t+1} | s_{≤t}), from the frozen Phase 4 checkpoint
- [ ] Running mean/std normalization (implementation + update rule recorded)
- [ ] Reward model weights verified frozen (no grads, checksum unchanged after a run)
- [ ] Policy, PPO, hyperparams, address set identical to Phase 2 — verified by config diff; only reward differs
- [ ] Hand-coded progress metrics still logged (not used for training)

### Run 1 — raw likelihood (no mask, no progress term)
- [ ] ≥1 seed run
- [ ] Failure modes recorded with video + byte-level evidence (expected: idling, menus, pausing)
- [ ] Each failure written to `notes/reward-hacking.md`: behavior, addresses responsible, how found

### Run 2 — + volatile mask
- [ ] ≥1 seed run; compare to Run 1; failure modes updated

### Progress term
- [ ] Implement candidate (a): novelty over auto-detected slow-changing bytes (no hand-named addresses); detection rule recorded
- [ ] Implement candidate (b): likelihood improvement over a no-op continuation
- [ ] Implement candidate (c): predicted temporal position within expert trajectories
- [ ] Implement candidate (d): VIPER-style exploration bonus (RND on RAM input), r = log p + β·r_expl (D-002)
- [ ] Short pilot of each (same seeds, same step budget) — results table below
- [ ] **Ask human:** pick the main progress term (open decision §10); others become Phase 8 ablations
- [ ] Coefficient for progress term vs likelihood recorded

### Run 3 — full method (likelihood + mask + chosen progress term)
- [ ] ≥3 seeds, same env-step budget as Phase 2
- [ ] Metrics: progress, % of hand-engineered performance, sample-efficiency curve
- [ ] Reward-vs-true-progress correlation during training
- [ ] New hacking behaviors appended to `notes/reward-hacking.md`
- [ ] Gameplay video of best policy

## Exit gate (human sign-off)
- [ ] Full method shows measurable progress across ≥3 seeds (number vs hand-engineered recorded)
- [ ] Failure modes of Runs 1–2 documented at byte level

## Notes / decisions
-
