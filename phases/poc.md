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
- [x] Train (1 seed, time budget recorded); evaluate vs random — final `runs/hand_final` (300k steps, D-008/D-009 config): 0 lines, 178-step eps vs random 448; earlier runs hacked (H-001, H-002)
- [x] **If the agent barely beats random → ask human** (e.g. scripted heuristic expert as data source instead) — asked; D-005

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
- [x] Batched reward inference speed recorded — ~300 windows/s (T·K=2312 tokens), compute-bound (batch 12→192 no gain; SDPA rewrite no gain); 48 ms/step for 12 envs (+8 ms RND) → RL ≈ 200 steps/s ≈ 4 h per 3M-step run

### P5 — RL on the learned reward
- [x] r = Σ log p over unmasked bytes, running normalization, + β·RND bonus (D-002 candidate d) — `rl.py`, β=0.5 on normalized scale; done-transitions get r=0; history padded with first state
- [x] Same policy/PPO/budget as P2; only the reward differs — `rl.py` calls `ppo.train` unchanged, 3M steps; RM checksum asserted unchanged after run
- [x] Runs (1 seed each): full method; no mask; curiosity (−log p) — `runs/pilot_a` (= ours), `runs/ours_nomask`, `runs/curiosity`, 300k steps (D-009)
- [x] Failure modes / hacks noted with responsible addresses (`notes/reward-hacking.md`) — H-001…H-004; addresses in RESULTS.md

### P6 — Report
- [x] Table: random, hand-engineered, ours, ours-no-mask, curiosity → lines, episode length — RESULTS.md
- [x] Top reward-driving addresses (simple per-address log p contribution) — `runs/analysis.json`, RESULTS.md
- [x] `RESULTS.md` with honest caveats (circular data, 1 seed, Tetris instead of SML)

## Exit gate (human sign-off)
- [ ] RESULTS.md reviewed by human

## Notes / decisions
- P1 mask rule: no byte changes on ≥95% of steps at 24-frame sampling (frame counter / RNG live in HRAM/IO, not WRAM). Used **no-op change rate ≥ 0.6** instead → 7 sound-driver bytes (0xDF94–0xDFC4) + stack byte 0xCFED. Random-play rates not thresholded because Start pauses Tetris and halves the rates. Only 278/8192 WRAM bytes ever change in play.
- P5: curiosity run keeps the RND bonus (β=0.5) so it differs from "ours" only in the sign of log p.
- D-010/D-011 (2026-11-04/05): all proposed 3M runs done — ours ×3, hand ×3, no-mask, curiosity (`runs/*_3m_s*`, `runs/analysis_3m.json`). KL early stop added mid-experiment; ours s1/s2 ran without it. REPORT.md, RESULTS.md and README updated to 3M.
