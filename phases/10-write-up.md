# Phase 10 — Write-up

**Goal:** paper + blog post that claim exactly what the experiments support.
**Depends on:** Phase 8 gate (+ Phase 9 if run). **Plan refs:** §2, §3, §11, Milestone 10.

## Checklist

### Before writing
- [ ] **Ask human:** method name (open decision §10)
- [ ] Final novelty re-check (repeat Phase 0 searches for anything new since)
- [ ] Review `DEVIATIONS.md`: every APPROVED deviation is reflected in the method/experiments text

### Paper
- [ ] Abstract placeholders filled from real results: [METHOD NAME], [TAS / human], [N], [X%], [Y]
- [ ] Transfer sentence matches Phase 9 verdict (or removed if skipped)
- [ ] Claims limited to PLAN.md §3; each claim → the table/figure that supports it
- [ ] "WRAM is near-complete game-logic state, not full machine state" caveat included
- [ ] CPU registers "considered and excluded" mentioned
- [ ] VIPER baseline's use of pixels stated (our method uses none)
- [ ] Compute section uses measured numbers; training-time-only framing
- [ ] Related work final; all venues verified
- [ ] Figures: system diagram, comparison table, sample-efficiency curves, ablations, byte-attribution, hacking catalogue
- [ ] Every number traceable to a run (seed, config, commit)

### Reproducibility
- [ ] Code release cleaned; README with exact commands
- [ ] Configs, seeds, ROM SHA-1s, save-state hashes published (no ROMs)
- [ ] Reward-model checkpoints released (if allowed)

### Blog / LinkedIn
- [ ] Hook from PLAN.md §11; avoid "no reward function" wording
- [ ] Numbers match the paper

### Submission
- [ ] Venue format, page limit, anonymization checked
- [ ] Human final read-through and approval

## Exit gate (human sign-off)
- [ ] Human approves submission

## Notes / decisions
-
