# PoC results — learned RAM-likelihood reward on Game Boy Tetris

**Date:** 2026-11-03 · **Scope:** 2-day PoC (D-004), Tetris instead of Super Mario Land (no SML ROM on hand) ·
**Single seed, 300k RL steps per run (D-009). Treat as a feasibility check, not a result.**

## TL;DR
- ✅ **The reward model works as a model.** A 0.9M-param byte-token transformer trained only on RAM traces
  (no actions, no pixels, no reward code) predicts held-out expert RAM at 0.040 nats/byte, 12× better than
  copying the last state. It scores random play ~8–10× less likely per step than expert play.
- ✅ **The learned reward ranks behaviour sensibly.** Expert ≫ ours > hand-reward agent > random, and
  pausing is the worst thing you can do. The pause trick that broke both of my hand-written rewards
  (H-001, H-002) is punished by the learned reward.
- ✅ **Curiosity (same model, opposite sign) fails the predictable way.** It pauses the game on 76% of steps
  (the RAM "noisy TV"). This is clean evidence that the *sign*, not the model, drives behaviour.
- ❌ **No agent clears lines.** Our method, its ablations, and the hand-engineered baseline all clear ~0 lines in
  300k steps. Our agent's states become *expert-like by the reward model's measure*, but it doesn't play Tetris yet.
- ⚠️ **The comparison is mostly circular.** "Ours scores best under the reward model" is partly by construction.
  The only independent metric (lines) shows no difference.

## Setup
| Component | What |
|---|---|
| Env | PyBoy 2.9, Tetris, WRAM 0xC000–0xDFFF snapshot every 24 frames, 8 buttons incl. Start & no-op (`env.py`) |
| Volatile mask | 8 bytes changing on ≥60% of no-op steps: sound driver 0xDF94–0xDFC4 + stack 0xCFED |
| Expert data | Scripted placement heuristic (D-005, `notes/scripted-expert.md`): 100 games, 183k steps, 53 lines/game |
| Reward model | Option B tokens (value + address + time emb), K=289 bytes × T=8 steps = 2,312 tokens, d=128, 4 layers, **0.90M params**, block-causal (`rm.py`) |
| Reward | r = norm(Σ_bytes log p(s_{t+1} ∣ s_{t-7..t})) + 0.5·norm(RND) (`rl.py`) |
| Policy | MLP on one-hot RAM (271 bytes), 17.8M params (almost all input embedding), PPO, entropy 0.05 (D-008), 12 envs |

## Reward model (P4)
| Data | NLL (nats/byte) |
|---|---|
| Expert val (10 held-out games) | **0.040** |
| Copy last state (best ε-smoothing) | 0.505 |
| Per-address unigram | 0.715 |
| Random-play traces | 0.417 |

The no-mask model (K=297) reaches the same val NLL (0.040).

## RL comparison (P5) — 300k steps, 1 seed
Lines and length: 24 eval episodes, sampled actions. Log p: the masked reward model's per-step score over
8 eval episodes, excluding the terminal transition (expert = 10 held-out games).

| Run | Reward | Lines/game | Ep. length | log p/step mean | **median** | Main behaviour |
|---|---|---|---|---|---|---|
| Expert (scripted) | — | 53.3 | 1830 | −13.8 | **−5.1** | rotate → shift → drop |
| Random | — | 0.0 | 448 | −109 | −106 | |
| Hand-engineered | Δlines + 0.1/piece | 0.0 | 178 | −72 | −33 | mixed, ↓ 29% |
| **Ours** | likelihood + mask + RND | 0.0 | 159 | **−38** | **−12.6** | ↓ 59% (fast drops) |
| Ours, no mask | likelihood + RND, K incl. volatile | 0.0 | 822 | −81 | −107 | ← 76% |
| Curiosity | −likelihood + RND | 0.04 | 525 | −197 | −144 | **Start 76%** (pauses) |

How to read it:
- **Ours ranks first on the reward model's own scale.** Its median step is near expert level. But a policy
  trained on that reward *should* score well on it, so this is the circular part.
- **The hand-engineered agent was never trained on log p, yet it scores between random and ours.** That
  weakly supports the reward model tracking "reasonable play".
- **No mask → worse.** Under the masked model, the no-mask agent's states are about as unlikely as random play,
  and its largest gaps are on sound-driver and stack bytes (0xDFBE, 0xDFAA, 0xDF9E, 0xCFF1). Hypothesis (not
  verified): without the mask, the reward was dominated by sound-engine state, and the agent learned to
  steer the sound effects (constant ← presses each trigger the move sound) rather than the board.
- **Training trend:** our raw log p improved −39 → −29/step over 300k steps and was still rising, so
  longer runs might get further.

## Which bytes drive the gap to expert (P6)
Top per-address log-p gaps (expert − agent), under the masked reward model:

| Agent | Top addresses (gap in nats) | Likely meaning |
|---|---|---|
| Ours | 0xC203 (1.67), 0xC018, 0xC01C, 0xC014, 0xC010 (~1.0) | 0xC010–0xC01D = shadow OAM, i.e. falling-piece sprite positions; 0xC203 unidentified |
| Hand | 0xC014, 0xC018, 0xC010, 0xC203, 0xC01C (~2.5) | same: the falling piece moves unlike the expert's |
| Random | 0xDFBE, 0xCFF1, 0xDFAE, 0xC018 (~3.5) | sound state + stack + piece |
| No mask | 0xDFBE, 0xDFAA, 0xDF9E, 0xCFF1 (~3) | sound driver + stack |
| Curiosity | 0xC014, 0xC018, 0xC210, 0xCFF1, 0xC010 (~7.5) | piece sprites (frozen/hidden while paused) |

The interpretability story from PLAN.md holds even in this tiny PoC. Each failure mode maps onto a small,
nameable set of addresses: piece sprites for "moves the piece unlike the expert", sound/stack for "unmasked
volatile bytes", sprites for "paused".

## Reward hacking catalogue (`notes/reward-hacking.md`)
| ID | Reward | Exploit |
|---|---|---|
| H-001 | hand v1 (lines + survival) | pause with Start, farm survival reward |
| H-002 | hand v2 (lines + per-piece) | pause overlay fooled my piece detector |
| H-003 | likelihood (entropy 0.01) | collapse to "always →", the best *trivial* policy (exploration failure, not a bug) |
| H-004 | curiosity | pause: never-seen screen = maximal surprise |

## Caveats — please read before quoting anything
1. **One seed, 300k steps** (10× less than planned). All lines-cleared numbers are 0 ± noise.
2. **Circular evaluation.** Log-p columns use the same reward model that trained "ours". The only independent
   metric (lines) shows nothing yet.
3. **The expert is a hand-written heuristic** (it reads the board from the screen tilemap). The learned reward
   contains no reward code, but it imitates engineered play. It's not human/TAS data.
4. **Tetris, not SML.** Tetris has a weak "progress" signal and no level/x-distance; PLAN.md's main target was untested.
5. **PPO config was tuned once for the learned reward** (entropy 0.01 → 0.05 after the collapse, D-008), then
   applied to every run, hand included.
6. **No-mask interpretation** (sound effects) is a hypothesis.
7. **0xC203's meaning** is unidentified.

## What I'd do next (in order)
1. Run ours vs hand for 3M+ steps, 3 seeds: does lines-cleared separate at all?
2. Behaviour-clone warm-start or a coarser action space (placement-level). Button-level PPO is the bottleneck,
   not the reward.
3. An independent metric beyond lines: e.g. holes/height on the board, computed outside the reward model.
4. Get an SML ROM and run the original plan's first target.

## Where everything is
`DEVIATIONS.md` (D-001–D-009) · `notes/log.md` (every discrepancy) · `notes/reward-hacking.md` ·
`notes/scripted-expert.md` · `runs/analysis.json` · `phases/poc.md` (checklist with evidence)
