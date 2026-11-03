# PoC — 2-day proof of concept on Tetris (D-004)

**Goal:** show end to end that a likelihood reward over RAM, learned from expert RAM traces, can train a RAM policy
that beats random play. Single seed and rough numbers are fine. Replaces Phases 1–10 until further notice.
**Game:** Tetris (GB), ROM from `~/Documents/Programming/Projects/dvBoyC/roms/tetris.gb`, copied into `roms/` (gitignored).

## Checklist

### P1 — Environment
- [x] venv with pyboy + torch; versions recorded in `requirements.txt` — pyboy 2.9.0, torch 2.14.0+cu130 (CUDA ok), py3.14
- [x] ROM SHA-1 recorded in `notes/roms.md`
- [x] Env: headless PyBoy, 8 actions (↑ ↓ ← → A B Start, no-op), frame-skip 24, fixed start state at the beginning of a game — `env.py`; start = `reset_game(timer_div)`, timer_div random per episode (seeds piece RNG) for diversity, fixed in tests
- [x] Observation = WRAM 0xC000–0xDFFF uint8; CPU registers excluded
- [x] Termination: game over or max-steps (value recorded) — MAX_STEPS=2000; no-op game ends at ~174 steps
- [x] Determinism check: same actions → identical RAM trace — `python env.py` asserts it (151×8192 trace)
- [x] Vectorized envs (≥8); throughput recorded — 8 envs 1.5k steps/s, 12 envs 1.7k steps/s (~6k frames/s single env)
- [x] Volatile mask from no-op + random rollouts; saved; count recorded — 8 bytes (`notes/tetris_volatile.json`)

### P2 — Hand-engineered baseline
- [x] Tetris RAM addresses for lines / score / game over verified in the emulator (source cited) — PyBoy `GameWrapperTetris` (reads RAM internally); lines=0 under random play, game_over fires
- [x] Hand reward: lines cleared + small survival term (coefficients recorded) — 1.0·Δlines + 0.01/step
- [x] MLP policy + critic on selected bytes (one-hot), PPO; config logged — `ppo.py`, K=271 policy bytes, `runs/*/config.json`
- [x] Random-policy reference: lines and episode length — 0.0 lines, 465 steps (48 eps)
- [~] Train (1 seed, time budget recorded); evaluate vs random — smoke run 400k steps / ~6 min: 0 lines, pause hack (H-001)
- [ ] **If the agent barely beats random → ask human** (e.g. scripted heuristic expert as data source instead)

### P3 — Expert data
- [x] N rollouts of the baseline agent, keep top-k by lines/length (N, k recorded) — replaced by scripted expert (D-005): N=k=100 episodes, all kept; `notes/scripted-expert.md`
- [x] Store `[episode, step, 8192]` uint8; train/val split by episode — npz per episode; val = every 10th episode (10), train = 90; + 20 random-play episodes
- [x] Address selection: bytes that vary in expert data minus volatile mask → K=128 (rule + list saved) — K=289 (D-007), `notes/rm_addresses.json`
- [x] Data volume recorded (steps, tokens) — 183,044 expert steps ≈ 52.9M tokens

### P4 — Reward model
- [x] Option B tokens (value + address + time emb), T=8, K=128, ~1M params, causal; volatile bytes excluded from the loss — `rm.py`, d=128 4L 4H, 0.90M params, block-causal, K=289 (D-007)
- [x] Overfit-one-batch check — 5.88 → 0.03 nats/byte in 300 its
- [x] Val NLL vs copy-last-state baseline — 0.040 vs copy 0.505 (ε=change rate 0.053) vs unigram 0.715; `runs/rm/eval.json`
- [x] Likelihood ordering: expert val > random play — NLL 0.040 expert vs 0.417 random
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
- P1 mask rule: no byte changes on ≥95% of steps at 24-frame sampling (frame counter / RNG live in HRAM/IO, not WRAM). Used **no-op change rate ≥ 0.6** instead → 7 sound-driver bytes (0xDF94–0xDFC4) + stack byte 0xCFED. Random-play rates not thresholded because Start pauses Tetris and halves the rates. Only 278/8192 WRAM bytes ever change in play.
