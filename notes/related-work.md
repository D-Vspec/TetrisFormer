# Related work — Phase 0 sweep (2026-11-03)

## Novelty verdict

**No paper found that uses likelihood over console RAM / emulator memory as a learned reward.**
Searches: "RAM + reward model + transformer", "learned reward from emulator memory", "Game Boy RAM transformer",
"VIPER follow-ups non-pixel", "RAM learned reward Atari demonstrations", "action-free reward from expert
observations / density". Caveat: web search only, not a full Semantic Scholar citation crawl of VIPER.

**But "log-likelihood of expert *states* as reward" is not new in itself.** Observation-only density rewards
exist for low-dimensional continuous states (IL-flOw, density-based reward modeling, f-IRL). So the novelty is
the *combination*: exact discrete machine state + autoregressive transformer + byte-level interpretability of
reward and hacking + Game Boy. Claim 1 must be framed as "VIPER on exact machine state", not "likelihood as
reward" in general. → proposed **D-003**.

## Close reading

### VIPER — Escontrela et al., NeurIPS 2023 (arXiv 2305.14343)
Venue confirmed via NeurIPS 2023 poster page + mlanthology. (danijar.com says "NeurIPS 2024" — wrong.)
- **Reward:** `r_t = ln p(x_{t+1} | x_{max(1,t-k):t}) + β · r_expl` (Eq. 6–7). Derived from KL(q‖p) =
  cross-entropy − entropy; **the exploration term is part of the method**, not optional. They use
  Plan2Explore and RND as `r_expl`.
- They name our exact failure mode: the most likely sequence ≠ typical behavior (MNIST blank image, coin flips);
  dense likelihood alone → local optima. RLBench: model gave high likelihood to *stationary* trajectories →
  fixed by temporal subsampling ×4 (train and reward).
- **Atari scoreboard masking** improved RL: unseen scores (opponent scoring in Pong) → noisy reward. This is the
  pixel analogue of our volatile-byte / off-distribution story, but hand-specified.
- Model: VQ-GAN (64×64 → 16×16 codes for Atari) + VideoGPT-style AR transformer in raster order, 16 frames,
  one-hot task label (multi-task model per suite). RL: DreamerV3 (Atari/RLBench), DrQ + Dv3 (DMC).
- **Data:** expert videos = top-k of 1000 rollouts from oracle RL agents trained on true reward; k=100 for
  Atari. 7 Atari games.
- Limitations they state: stochastic envs → agent prefers low-uncertainty states; short context → myopic.
- Code + data: ViperRL.com.

**Implications for us**
1. Our "progress term" overlaps VIPER's exploration term. Reviewers will ask why ours over theirs. → **D-002**:
   add VIPER's exploration bonus (RND) as progress-term candidate (d), and run the VIPER baseline *with* its
   exploration term.
2. Our automatic volatile mask is a real differentiator vs their hand-chosen scoreboard mask — say so.
3. Data: VIPER didn't use human/TAS data; they used oracle-agent rollouts. → **D-001** (see data note below).
4. Temporal subsampling: our 24-frame step already does this; cite it as the reason.

### Rank-Then-Act — Maksyuta, Bredis, Rakhimov, Gavrilov, arXiv 2607.01897 (ICML 2026 workshop)
- VLM (Qwen2.5-VL-7B) fine-tuned with GRPO to order shuffled expert frames; Spearman ρ with true time = reward.
- **Pixels only**, no RAM. Policy is the VLM agent (VL-DAC); MLP+PPO is an ablation on Catrap.
- Games via PyBoy: Catrap L2/4/6, Kirby's Dream Land first checkpoint. Metrics: success rate, pass@5.
  Kirby: RTA 0.07 vs oracle 0.40.
- Data: human Catrap playthroughs + 70 YouTube GB playthroughs. Baselines: GVL, VLM-RM, Rank2Reward. **No VIPER.**
- Compute: 4×H100 stage 1, 1×H100 ≤24h stage 2. Code: github.com/corl-team/rank-then-act.
- Implications: Kirby is the overlap title. Their reward ≈ our candidate (c) "temporal position".
  Their scorer is a 7B VLM; ours is ~5M — a compute contrast worth stating.

### PokeRL — Mudireddy & Patibandla (Texas A&M), arXiv 2604.10812
- PyBoy, early game only: exit house, reach tall grass, win first rival battle.
- RAM used for reward/termination, **policy is SB3 CnnPolicy on pixels** (RAM not given to policy).
- Hierarchical rewards (micro/meso/macro), anti-loop tile-visit penalties (>3 / >5 visits), map masking.
- Code: github.com/reddheeraj/PokemonRL. Reward details from secondary summaries — verify in PDF in Phase 7.
- Implication: our "PokeRL-style" baseline is an adaptation (RAM policy), not a reproduction. Say so.

### Other close priors (add to paper)
- **IL-flOw** — Chang et al., NeurIPS 2021 Robot Learning Workshop (arXiv 2205.09251). Normalizing-flow density
  over expert state→state transitions as a stationary, non-adversarial reward. Closest *idea* to ours outside VIPER.
- **f-IRL** (Ni et al., CoRL 2020, PMLR v155) — state-marginal matching from expert states.
- **Density-based reward modeling** — `imitation` library; notes KDE fails in high dimensions.
- **XIPER** — arXiv 2606.03201. VIPER + domain translation, DreamerV3, still pixels.
- **Diffusion Reward** — arXiv 2312.14134. Video diffusion rewards; claims better generalization than VIPER.
- **Ibarz et al. 2018** — reward learning from preferences + demos in Atari (pixels).
- **Sygnowski & Michalewski 2016** — "Learning from the memory of Atari 2600": RAM-input policies (support for RAM policies).

## Data-volume warning (feeds Phase 3)
- Plan target ≈ 400k agent steps. One SML TAS ≈ 12 min ≈ 43k frames ≈ **~1.8k agent steps** at 24 frames/step.
  TAS alone is ~200× short.
- Published SML TAS (TheAxeMan lineage, VBA-rr) uses a screen-warp trick on autoscroll levels — may not count as glitchless.
- VIPER's answer: top-k rollouts from an oracle agent. → D-001.

## Citation verification
| Entry | Status |
|---|---|
| VIPER — NeurIPS 2023 | ✔ verified |
| Burda et al. — ICLR 2019 | ✔ verified |
| IRIS — Micheli et al., ICLR 2023 (oral) | ✔ verified |
| TWM — Robine et al., ICLR 2023 | ✔ verified |
| STORM — Zhang et al., NeurIPS 2023 | ✔ verified |
| Rank-Then-Act — arXiv 2607.01897, ICML 2026 workshop | ✔ verified (arXiv + repo) |
| PokeRL — arXiv 2604.10812 | ✔ verified (arXiv) |
| Preference Transformer — ICLR 2023 | ✔ verified |
| RUDDER — NeurIPS 2019 | ✔ verified |
| Reward Bag Transformer — **TMLR 2025** (arXiv 2402.03771); an ICML 2026 poster with same title also exists | ✔ verified — PLAN.md says "2024" |
| DTRD — Zhu et al., IJCAI 2023 | ✔ verified |
| ICM — Pathak et al., ICML 2017 | ✔ verified |
| AtariARI — Anand et al., NeurIPS 2019 | ✔ verified |
| OCAtari — Delfosse et al., RLJ vol. 1, 2024 (RLC 2024) | ✔ verified |
| Kwon et al. — ICLR 2023 | ✔ verified |
| Motif — Klissarov et al., ICLR 2024 | ✔ verified |
| VLM-RMs — Rocamonde et al., ICLR 2024 | ✔ verified |
| Eureka — ICLR 2024 | ✔ verified |
| Schmidhuber 1990/1991 | ☐ not verified (classic; low risk) |

BibTeX: `notes/refs.bib`.
