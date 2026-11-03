# Phase 0 — Literature sweep & decisions

**Goal:** confirm the idea isn't already published and lock the decisions needed before building.
**Depends on:** nothing. **Plan refs:** §4, §10.

## Checklist

### Novelty sweep
- [~] Semantic Scholar / Google Scholar / arXiv search: "RAM" + "reward model" + "transformer" — web search only, no hits (`notes/related-work.md`); a proper Semantic Scholar crawl is not done
- [x] Search: "memory state" / "console RAM" + "learned reward" / "likelihood reward" — no hits; found IL-flOw / density-reward prior art
- [~] Search: papers citing VIPER (Escontrela et al., NeurIPS 2023) for non-pixel variants — XIPER, Diffusion Reward found (both pixels); full citation list not crawled
- [~] Search: papers citing AtariARI and OCAtari that use RAM for *reward*, not just representation — keyword search only, no hits
- [x] Re-check Rank-Then-Act (arXiv 2607.01897) and PokeRL (arXiv 2604.10812) for follow-ups / code releases — both have code; RTA at ICML 2026 workshop
- [x] Write findings to `notes/related-work.md`: each hit → one line on how it differs from us — commit 9899619
- [x] **Ask human:** does anything found kill or force a reframe of the novelty claim? — "Proceed + approve D-003" (2026-11-03)

### Close reading
- [x] VIPER: reward definition, normalization, exploration term, model arch, data size; note code availability — `notes/related-work.md` §VIPER (from PDF)
- [x] Rank-Then-Act: games, metrics, reward definition, compute; note code availability — `notes/related-work.md` §RTA (arXiv HTML)
- [x] PokeRL: hand-designed reward terms (template for our Pokémon baseline) — from abstract + secondary summaries; re-verify in PDF in Phase 7
- [x] Note any detail that contradicts PLAN.md §5 in `notes/related-work.md` (→ deviation if we change the plan) — D-001, D-002, D-003 proposed

### Citation verification
- [x] Burda et al. 2019 venue (ICLR) confirmed
- [x] IRIS, TWM, STORM venues confirmed — ICLR 2023 / ICLR 2023 / NeurIPS 2023
- [~] All §4 entries have full bibliographic info in `notes/refs.bib` — all except Schmidhuber 1990/91; Reward Bag Transformer is TMLR 2025, not "2024"

### Decisions (PLAN.md §10) — human picks, Claude proposes
- [x] Target venue and deadline → write into this file — NeurIPS 2027; official CFP not out yet, planning deadline **2027-05-03** (NeurIPS 2026 abstract deadline was May 4; trackers estimate May 14–21, 2027). Re-check neurips.cc when the CFP is posted
- [x] TAS vs human data: decide now or mark "decide in Phase 3" — decide in Phase 3; oracle-agent rollouts allowed as an extra source (D-001)
- [x] Policy MLP vs transformer encoder: default MLP for Phases 2–5 unless human says otherwise — MLP (default; human has not overridden it)
- [x] Final game list: mark "decide in Phase 7"
- [x] Progress term: mark "decide in Phase 5" — now 4 candidates incl. RND (D-002)
- [x] Method name: mark "decide before Phase 10"

## Exit gate (human sign-off)
- [x] Novelty sweep done and summarized; human confirms we proceed — 2026-11-03
- [~] Venue + deadline set; working timeline per phase written below — proposed timeline below, **awaiting human sign-off**

## Notes / decisions
- 2026-11-03: novelty sweep found no RAM-likelihood reward paper; closest priors are VIPER (pixels) and IL-flOw (low-dim state density). See `notes/related-work.md`.
- 2026-11-03: D-001, D-002, D-003 approved and applied.
- Citation fix in PLAN.md §4: Reward Bag Transformer → TMLR 2025 (checking venues was part of this phase).

### Proposed timeline (26 weeks → 2027-05-03), sequential phases
| Phase | Weeks | Dates |
|---|---|---|
| 1 Environment | 2 | Nov 3 – Nov 15 |
| 2 Baseline | 3 | Nov 16 – Dec 6 |
| 3 Data | 3 | Dec 7 – Dec 27 |
| 4 Reward model | 3 | Dec 28 – Jan 17 |
| 5 RL likelihood | 3 | Jan 18 – Feb 7 |
| 6 Baselines | 2 | Feb 8 – Feb 21 |
| 7 More games | 4 | Feb 22 – Mar 21 |
| 8 Ablations | 2 | Mar 22 – Apr 4 |
| 9 Transfer | 0 | only if ahead of schedule |
| 10 Write-up | 4 | Apr 5 – May 2 |

Risks: Phase 7 is the tightest; 4 weeks for Pokémon Red + 1–2 titles is ambitious. The schedule has no slack, so running Phase 3 TAS/human collection alongside Phase 2 would buy ~2 weeks (needs approval: CLAUDE.md §0.6).
