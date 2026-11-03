# Phase 0 — Literature sweep & decisions

**Goal:** confirm the idea isn't already published and lock the decisions needed before building.
**Depends on:** nothing. **Plan refs:** §4, §10.

## Checklist

### Novelty sweep
- [ ] Semantic Scholar / Google Scholar / arXiv search: "RAM" + "reward model" + "transformer"
- [ ] Search: "memory state" / "console RAM" + "learned reward" / "likelihood reward"
- [ ] Search: papers citing VIPER (Escontrela et al., NeurIPS 2023) for non-pixel variants
- [ ] Search: papers citing AtariARI and OCAtari that use RAM for *reward*, not just representation
- [ ] Re-check Rank-Then-Act (arXiv 2607.01897) and PokeRL (arXiv 2604.10812) for follow-ups / code releases
- [ ] Write findings to `notes/related-work.md`: each hit → one line on how it differs from us
- [ ] **Ask human:** does anything found kill or force a reframe of the novelty claim?

### Close reading
- [ ] VIPER: reward definition, normalization, exploration term, model arch, data size; note code availability
- [ ] Rank-Then-Act: games, metrics, reward definition, compute; note code availability
- [ ] PokeRL: hand-designed reward terms (template for our Pokémon baseline)
- [ ] Note any detail that contradicts PLAN.md §5 in `notes/related-work.md` (→ deviation if we change the plan)

### Citation verification
- [ ] Burda et al. 2019 venue (ICLR) confirmed
- [ ] IRIS, TWM, STORM venues confirmed
- [ ] All §4 entries have full bibliographic info in `notes/refs.bib`

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
