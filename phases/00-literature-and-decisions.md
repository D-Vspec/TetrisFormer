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
- [ ] **Ask human:** does anything found kill or force a reframe of the novelty claim?

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
- [ ] Target venue and deadline → write into this file
- [ ] TAS vs human data: decide now or mark "decide in Phase 3"
- [ ] Policy MLP vs transformer encoder: default MLP for Phases 2–5 unless human says otherwise
- [ ] Final game list: mark "decide in Phase 7"
- [ ] Progress term: mark "decide in Phase 5"
- [ ] Method name: mark "decide before Phase 10"

## Exit gate (human sign-off)
- [ ] Novelty sweep done and summarized; human confirms we proceed
- [ ] Venue + deadline set; working timeline per phase written below

## Notes / decisions
-
