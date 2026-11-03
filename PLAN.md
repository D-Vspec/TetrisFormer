# Learning Rewards from Machine State

**Working title:** Learning Rewards from Machine State: Predictive Likelihood as a Reward Signal for Agents Acting on Console Memory
**Method name:** [METHOD NAME]
**Status:** Planning

---

## 1. One-line summary

Train an autoregressive transformer on Game Boy RAM traces of competent play, then use its predictive likelihood over future memory states as the reward for an RL agent that also acts purely on RAM. No hand-written reward code, no pixels anywhere in the pipeline, no action labels.

## 2. Abstract (draft)

Reward specification remains a bottleneck in reinforcement learning: hand-engineered rewards require per-environment domain knowledge and invite reward hacking, while learned rewards over high-dimensional observations are opaque and hard to debug. We study this problem in a minimal, exactly observable setting: a game console, whose game-logic state is a compact, discrete byte array (work RAM). We introduce [METHOD NAME], which trains an autoregressive transformer on [TAS / human] trajectories of competent play and uses its predictive likelihood over future memory states as a dense reward — requiring no per-game reward code, no pixel observations, and no action labels. Masking suppresses reward from stochastic memory regions (RNG, timers), while a progress term prevents degenerate high-likelihood behavior such as idling in menus. Because the state is exact, the learned reward is inspectable at byte granularity: we identify which addresses drive reward and characterize reward hacking in terms of specific memory locations rather than opaque latent features. On [N] Game Boy titles, including Super Mario Land and Pokémon Red, agents trained solely with predictive rewards reach [X%] of hand-engineered performance and outperform VIPER and curiosity baselines on [Y] of [N] games. We further examine [whether / how far] a single reward model transfers across titles.

> Before submission: only state the transfer result as strongly as the experiments support.

## 3. Core claims (keep to these)

1. **Method:** Predictive likelihood over RAM is a usable dense reward that replaces per-game reward engineering.
2. **Interpretability:** Because RAM is exact and discrete, the reward and its failure modes (reward hacking) can be explained at the level of specific addresses.
3. **Supporting:** Masking + a progress term are needed to make likelihood rewards work; cross-game transfer is explored, not promised.

Framing: *we trade human reward-engineering effort for training-time compute; the deployed agent is just the policy and costs nothing extra.*

## 4. Related work

### Closest analogue
- **VIPER** — Escontrela et al., *Video Prediction Models as Rewards for RL*, NeurIPS 2023. Autoregressive transformer trained on expert video; prediction likelihood used as reward. **Our method is VIPER with RAM instead of frames.** Also our main learned-reward baseline.

### Transformers / sequence models as reward models
- **Preference Transformer** — Kim et al., ICLR 2023. Transformer reward model with non-Markovian rewards from human preferences.
- **RUDDER** — Arjona-Medina et al., NeurIPS 2019. LSTM predicts return; reward redistributed along the sequence.
- **Reward Bag Transformer** — RL from Bagged Reward, 2024 (arXiv 2402.03771).
- **DTRD** — Zhu et al., IJCAI 2023. Reward redistribution for Decision Transformer under long-delayed rewards.

### Prediction error as reward (curiosity) — our ablation lineage
- Schmidhuber, 1990/1991 — curiosity from prediction error / prediction improvement.
- **ICM** — Pathak et al., ICML 2017.
- Burda et al., *Large-Scale Study of Curiosity-Driven Learning*, ICLR 2019 (verify).
- Known failure mode: "noisy TV" — unpredictable transitions give endless reward. RAM equivalent: RNG / timer bytes.

### RAM as state representation
- **AtariARI** — Anand et al., *Unsupervised State Representation Learning in Atari*, NeurIPS 2019.
- **OCAtari** — object-centric Atari via RAM, Reinforcement Learning Journal.

### Game Boy RL
- **PokeRL** — Mudireddy et al., 2026 (arXiv 2604.10812). PyBoy + hand-designed hierarchical RAM rewards. Example of the engineering we replace.
- **Rank-Then-Act** — 2026 (arXiv 2607.01897). Reward-free control from expert video progress ranking; evaluated on PyBoy (Catrap, Kirby). **Nearest competitor** — pixels + ranking vs our RAM + likelihood.

### Transformer world models (contrast, verify venues)
- IRIS (Micheli et al., ICLR 2023), TWM (Robine et al., ICLR 2023), STORM (NeurIPS 2023). They learn action-conditioned dynamics for planning; we learn an action-free model of *good play* for reward.

### LLM / VLM rewards (contrast only)
- Kwon et al., *Reward Design with Language Models*, ICLR 2023; Motif (ICLR 2024); VLM-RMs (Rocamonde et al., ICLR 2024); Eureka (ICLR 2024).

> TODO: Semantic Scholar sweep for "RAM" + "reward model" + "transformer" before claiming novelty.

## 5. Method

### 5.1 System overview

```
emulator (PyBoy) ──RAM──▶ policy (RAM input) ──action──▶ emulator
        │
        └──RAM history──▶ reward transformer (frozen) ──r_t──▶ PPO
```

Two separate networks:

| | Policy | Reward transformer |
|---|---|---|
| Input | Current RAM state | Last T RAM states |
| Output | Softmax over 8 actions (↑ ↓ ← → A B Start, no-op) | Next RAM state, 256-way softmax per byte |
| Trained by | PPO | Supervised next-state prediction |
| Used at deployment | Yes | No |

The policy is (near-)Markov in RAM; the reward is history-conditioned.

### 5.2 Data

- **Source options:**
  - **TAS** (TASVideos): input files replay deterministically → exact RAM traces for free. Use **glitchless categories only** — TAS runs exploit glitches, RNG manipulation, wrong-warps, ACE, which would teach the reward model unreachable states.
  - **Human recordings**: more representative of reachable competent play; scarcer.
- **Pipeline:** replay inputs in PyBoy → snapshot RAM every agent step (e.g. every ~24 frames) → store `[episode, step, K]` uint8 arrays.
- **Actions are discarded** — the reward model is action-free.
- **Rough data target:** ~20 tokens/param → ~100M tokens for a 5M model (~25k windows of 16 steps).

### 5.3 State selection and masking

- Memory: WRAM 0xC000–0xDFFF (8 KB) holds game-logic state. HRAM / IO optional.
- **Exclude CPU registers** (PC etc. reflect snapshot timing, not game state). Mention as considered and excluded.
- Note for paper: WRAM is *near-complete* game-logic state, not full machine state (VRAM, OAM, cartridge SRAM, MBC bank registers, PPU/timer internals live elsewhere).
- **Volatile byte mask:** find bytes that change every step regardless of input (RNG, frame counters, timers) by running random / no-op rollouts; exclude them from loss and reward.
- **Address selection (Option B):** K ≈ 128–512 addresses chosen from (a) bytes that vary during expert play, (b) published RAM maps where available.

### 5.4 Tokenization options

| Option | Tokens | Seq length | Pros | Cons |
|---|---|---|---|---|
| A: one token per step | RAM vector → linear embed | T | Cheapest (~CNN-level compute) | Model must find relevant bytes in a flat vector |
| **B: byte tokens over subset** (prototype) | value emb + address emb + time emb | T × K | Interpretable attention over addresses | Expensive attention at T×K |
| C: diff tokens | `(address, new_value)` for changed bytes | variable | Full 8 KB in scope, most novel | Variable length, packing |

Plan: **prototype with B**, upgrade to **C** as the "full RAM" variant, keep **A** as the cheap ablation.

Bytes are categorical (map ID 37 is not "more" than 36): embed, don't feed as scalars.

### 5.5 Reward transformer

- Decoder-only, causal, next-state prediction; cross-entropy over 256 values per byte.
- **Prototype size:** d_model = 256, 6 layers ≈ 5M params; T = 16, K = 256.

| Config | Params | Use |
|---|---|---|
| d=128, 4 layers | ~1M | Pipeline debugging |
| d=256, 6 layers | ~5M | Main prototype |
| d=512, 8 layers | ~25M | If underfitting / multi-game |

- If attention at 4,096 tokens is slow: drop T to 8, or factorize attention (within-step over addresses, then across time).
- **Frozen during RL**; reward computed in batches across parallel envs with KV caching.

### 5.6 Reward

- **Likelihood reward (main):** `r_t = log p(s_{t+1} | s_{≤t})`, summed over unmasked bytes, normalized (running mean/std).
- **Curiosity reward (ablation):** `r_t = −log p(s_{t+1} | s_{≤t})` — same model, opposite sign.
- **Progress term:** counters the failure mode where idling in predictable states (menus, standing still) scores high likelihood. Candidates:
  - novelty over a small set of slow-changing bytes (map ID, level counter) without naming them by hand — e.g. auto-detected as bytes that change rarely in expert data;
  - reward for likelihood *improvement* over a no-op continuation;
  - predicted temporal position within expert trajectories (how "far along" the state looks).
  Decide one; ablate the rest.

### 5.7 Policy

- RAM input, same selected addresses, one-hot / bit-unpacked bytes.
- Architecture: MLP (default) or small transformer encoder over byte tokens.
- **Do not share the encoder with the reward model** (coupling eases hacking and muddies interpretability).
- PPO with a critic; frame-skip ~24; 16–64 parallel PyBoy envs.

## 6. Experiments

### 6.1 Games
- Super Mario Land (linear progress, short episodes — good first target)
- Pokémon Red (long horizon, exploration-heavy)
- [1–2 more: e.g. Kirby's Dream Land, Tetris] — overlap with Rank-Then-Act titles helps comparison.

### 6.2 Controlled comparison — policy fixed, only reward varies

| Run | Policy | Reward |
|---|---|---|
| Hand-engineered | RAM policy | Hand-coded RAM reward (PokeRL-style) |
| **Ours** | RAM policy | Transformer likelihood (+ mask + progress) |
| Curiosity | RAM policy | Transformer prediction error |
| VIPER | RAM policy | Likelihood over rendered frames |
| (opt.) Rank-Then-Act | RAM policy | Their progress reward |

**Run the hand-engineered baseline first.** If it trains well, the RAM policy is not the bottleneck — this pre-empts the "RAM policies are weak" objection.

### 6.3 Metrics
- Game progress (Mario: level/x-distance; Pokémon: badges, maps visited, story flags)
- % of hand-engineered performance
- Sample efficiency (progress vs env steps)
- Reward model held-out NLL; correlation of reward with true progress

### 6.4 Ablations
- No mask vs mask
- No progress term vs each progress variant
- Tokenization A vs B vs C
- Context length T
- Model size
- TAS vs human data
- Likelihood vs curiosity sign

### 6.5 Interpretability & reward hacking
- Attention / gradient attribution per address → which bytes drive reward; map to known RAM-map meanings.
- Catalogue reward-hacking behaviors found during training; for each, identify the addresses responsible.
- Show where the model is off-distribution (agent states never seen in expert data).

### 6.6 Transfer (stretch)
- Single reward model across games with a game-ID token.
- Report honestly: full / partial / none.

## 7. Compute budget (rough)

| Component | Params | Per-step compute |
|---|---|---|
| Typical CNN-PPO policy (reference) | ~1–2M | ~20 MFLOPs |
| RAM MLP policy | <1M | negligible |
| Reward transformer, Option B, KV-cached | ~5M | ~2–3 GFLOPs |
| Reward transformer, Option A | ~5M | ~CNN-level |

Reward model dominates training cost (~100× policy for Option B). State this in the paper with the training-time-only framing.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Likelihood reward rewards idling / menus | Progress term; ablation shows necessity |
| RNG/timer bytes dominate reward | Volatile-byte mask |
| TAS data teaches glitch states | Glitchless categories; compare with human data |
| TAS input files don't sync in PyBoy (made for other emulators) | Verify sync early; fall back to human recordings |
| RAM policy underperforms | Hand-engineered baseline on same policy first |
| Reward model too slow for RL loop | Option A, smaller T, factorized attention, batching |
| Off-distribution states → reward hacking | Turn into a contribution via byte-level analysis |
| Transfer fails | Keep as exploratory; don't headline |
| Prior work already does this | Literature sweep before writing intro |

## 9. Milestones

1. **Environment:** PyBoy wrapper, RAM snapshotting, parallel envs, volatile-byte detection.
2. **Baseline:** RAM-policy PPO with hand-coded reward on Super Mario Land. Must learn.
3. **Data:** RAM traces from TAS / human play; verify replay sync.
4. **Reward model v1:** Option B, ~5M params, held-out NLL sanity check.
5. **RL with likelihood reward:** observe failure modes (expect idling) → add mask + progress term.
6. **Baselines:** curiosity, VIPER.
7. **Pokémon Red** + 1–2 more titles.
8. **Ablations + interpretability analysis.**
9. **Transfer experiment** (stretch).
10. **Write-up** + blog post.

## 10. Open decisions

- [ ] TAS vs human data (or both)
- [ ] Which progress term
- [ ] Policy: MLP vs transformer encoder
- [ ] Final game list
- [ ] Target venue and deadline
- [ ] Method name

## 11. Notes for the blog / LinkedIn

- Accurate hook: *"I didn't teach a transformer to play Pokémon. I taught it to tell an AI when it was playing well."*
- Alt: *"The reward function for my Pokémon agent isn't code. It's a transformer."*
- Avoid "no reward function" — the transformer *is* the reward function; what's gone is the *hand-written* one.
- Fun fact: Super Mario Land's ROM is ~64 KB; a 5M-param model is ~20 MB. The transformer isn't compressing the game — it's learning which of the game's possible trajectories count as playing well, information the ROM doesn't contain.
