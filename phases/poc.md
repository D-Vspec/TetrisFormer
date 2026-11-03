# PoC — 2-day proof of concept on Tetris (D-004)

**Goal:** show end to end that a likelihood reward over RAM, learned from expert RAM traces, can train a RAM policy
that beats random play. Single seed and rough numbers are fine. Replaces Phases 1–10 until further notice.
**Game:** Tetris (GB), ROM from `~/Documents/Programming/Projects/dvBoyC/roms/tetris.gb`, copied into `roms/` (gitignored).

## Checklist

### P1 — Environment
- [ ] venv with pyboy + torch; versions recorded in `requirements.txt`
- [ ] ROM SHA-1 recorded in `notes/roms.md`
- [ ] Env: headless PyBoy, 8 actions (↑ ↓ ← → A B Start, no-op), frame-skip 24, fixed start state at the beginning of a game
- [ ] Observation = WRAM 0xC000–0xDFFF uint8; CPU registers excluded
- [ ] Termination: game over or max-steps (value recorded)
- [ ] Determinism check: same actions → identical RAM trace
- [ ] Vectorized envs (≥8); throughput recorded
- [ ] Volatile mask from no-op + random rollouts; saved; count recorded

### P2 — Hand-engineered baseline
- [ ] Tetris RAM addresses for lines / score / game over verified in the emulator (source cited)
- [ ] Hand reward: lines cleared + small survival term (coefficients recorded)
- [ ] MLP policy + critic on selected bytes (one-hot), PPO; config logged
- [ ] Random-policy reference: lines and episode length
- [ ] Train (1 seed, time budget recorded); evaluate vs random
- [ ] **If the agent barely beats random → ask human** (e.g. scripted heuristic expert as data source instead)

### P3 — Expert data
- [ ] N rollouts of the baseline agent, keep top-k by lines/length (N, k recorded)
- [ ] Store `[episode, step, 8192]` uint8; train/val split by episode
- [ ] Address selection: bytes that vary in expert data minus volatile mask → K=128 (rule + list saved)
- [ ] Data volume recorded (steps, tokens)

### P4 — Reward model
- [ ] Option B tokens (value + address + time emb), T=8, K=128, ~1M params, causal; volatile bytes excluded from the loss
- [ ] Overfit-one-batch check
- [ ] Val NLL vs copy-last-state baseline
- [ ] Likelihood ordering: expert val > random play
- [ ] Batched reward inference speed recorded

### P5 — RL on the learned reward
- [ ] r = Σ log p over unmasked bytes, running normalization, + β·RND bonus (D-002 candidate d)
- [ ] Same policy/PPO/budget as P2; only the reward differs
- [ ] Runs (1 seed each): full method; no mask; curiosity (−log p)
- [ ] Failure modes / hacks noted with responsible addresses (`notes/reward-hacking.md`)

### P6 — Report
- [ ] Table: random, hand-engineered, ours, ours-no-mask, curiosity → lines, episode length
- [ ] Top reward-driving addresses (simple per-address log p contribution)
- [ ] `RESULTS.md` with honest caveats (circular data, 1 seed, Tetris instead of SML)

## Exit gate (human sign-off)
- [ ] RESULTS.md reviewed by human

## Notes / decisions
-
