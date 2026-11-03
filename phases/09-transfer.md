# Phase 9 — Transfer (stretch)

**Goal:** test whether one reward model works across titles. Report honestly: full / partial / none.
**Depends on:** Phase 8 gate. **Human decides whether to run this phase at all.** **Plan refs:** §6.6, Milestone 9.

## Checklist

- [ ] **Ask human:** run this phase? (yes / skip / defer)
- [ ] Shared address handling across games decided with human (per-game address sets + game-ID token, or tokenization C)
- [ ] Multi-game reward model with game-ID token; size (likely d=512/8L) recorded
- [ ] Per-game held-out NLL: multi-game vs single-game models
- [ ] RL on each game with the shared model, same protocol as Phase 5 Run 3, ≥3 seeds
- [ ] (Optional, ask) Held-out game: train on N−1 games, evaluate on the remaining one
- [ ] Verdict per game written: full / partial / none, with numbers
- [ ] Abstract §2 transfer sentence drafted to match the verdict — no stronger

## Exit gate (human sign-off)
- [ ] Verdict recorded and accepted

## Notes / decisions
-
