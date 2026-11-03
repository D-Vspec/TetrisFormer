# Deviations Log

Every departure from a phase checklist or `PLAN.md` — proposed, approved, or rejected — is recorded here.
Newest at the bottom. IDs are never reused. Only a human can approve; quote them.

## Template

```
## D-000 — <short title>
- Date proposed: YYYY-MM-DD
- Phase / item: phases/0X-....md → "<checklist item text>"
- Checklist says: <what the plan specifies>
- Proposed instead: <what we'd do>
- Reason: <evidence: error, measurement, paper, constraint>
- Downstream impact: <phases/items/claims affected>
- Status: PROPOSED | APPROVED | REJECTED | SUPERSEDED by D-XXX
- Human decision: "<verbatim quote>" — YYYY-MM-DD
- Outcome: <what actually happened after applying it>
```

---

<!-- entries below -->

## D-001 — Add oracle-agent rollouts as an expert-data source
- Date proposed: 2026-11-03
- Phase / item: phases/03-data.md → "Ask human: TAS, human recordings, or both"; PLAN.md §5.2
- Checklist says: data sources are TAS or human recordings.
- Proposed instead: add a third option — top-k episodes from the Phase 2 hand-engineered agent (VIPER's protocol: top-100 of 1000 rollouts per Atari game).
- Reason: one SML TAS ≈ 1.8k agent steps vs ~400k target (~200× short); SML TAS uses a screen-warp trick (may not be glitchless). VIPER did not use human/TAS data at all.
- Downstream impact: weakens claim 1 if the *only* data comes from a hand-rewarded agent (the reward model would learn the engineered reward's behavior — circular). Would need to be framed as "same data protocol as VIPER", with TAS/human as the non-circular variant. Affects Phases 3, 8 (TAS vs human vs oracle ablation), 10.
- Status: APPROVED
- Human decision: "Allow as extra source (Recommended)" — 2026-11-03 (selected answer to Phase 0 question)
- Outcome:

## D-002 — VIPER exploration term: add as progress-term candidate and include in VIPER baseline
- Date proposed: 2026-11-03
- Phase / item: phases/05-rl-likelihood.md → "Progress term" candidates (a)–(c); phases/06-baselines.md → VIPER reward
- Checklist says: three progress-term candidates; VIPER reward "+ VIPER's own extras if their method includes them".
- Proposed instead: (1) add candidate (d): VIPER-style exploration bonus (RND) on RAM; (2) state explicitly that the VIPER baseline uses r = log p + β·r_expl.
- Reason: VIPER Eq. 6 makes the exploration/entropy term part of the method, derived from KL matching; it exists to fix the same "most-likely ≠ typical / idling" problem our progress term targets. Reviewers will ask why ours beats theirs.
- Downstream impact: one more pilot in Phase 5, one more ablation row in Phase 8.
- Status: APPROVED
- Human decision: "Approve both (Recommended)" — 2026-11-03 (selected answer to Phase 0 question)
- Outcome:

## D-003 — Narrow novelty framing of claim 1
- Date proposed: 2026-11-03
- Phase / item: phases/00 → novelty sweep; PLAN.md §3 claim 1, §4
- Checklist says: claim 1 "Predictive likelihood over RAM is a usable dense reward that replaces per-game reward engineering".
- Proposed instead: keep claim, but add to PLAN.md §4 an "observation-only density rewards" group (IL-flOw, f-IRL, density reward modeling) and frame novelty as exact discrete machine state + AR transformer + byte-level interpretability, not likelihood-as-reward itself.
- Reason: IL-flOw (NeurIPS 2021 workshop) already uses expert state-transition log-density as reward on low-dim states. No RAM-likelihood reward paper found.
- Downstream impact: PLAN.md §4 edit; intro framing in Phase 10. Claims list unchanged.
- Status: APPROVED
- Human decision: "Proceed + approve D-003 (Recommended)" — 2026-11-03 (selected answer to Phase 0 question)
- Outcome:

## D-004 — Compress project to a ~2-day proof of concept
- Date proposed: 2026-11-03 (requested by human)
- Phase / item: phases/00 → proposed 26-week timeline; all of Phases 1–10
- Checklist says: 10 sequential phases toward NeurIPS 2027 (2027-05-03).
- Proposed instead: PoC in ~2 days on **Tetris** (only usable ROM on disk). Scope: PyBoy env + volatile mask; short hand-engineered PPO baseline; expert data = top-k rollouts of that agent (D-001); ~1M reward model (Option B, T=8, K=128); PPO on likelihood + mask + RND; compare vs hand-engineered, random, curiosity (1 seed each); mask on/off; short results README. Cut: TAS, VIPER, extra games, other ablations, transfer, paper. Checklist: phases/poc.md.
- Reason: human: "Change in timeline, I need it done in a couple days (doesn't have to be perfect just something as a Poc)"
- Downstream impact: most of Phases 6–10 cut; multi-seed, ablations, VIPER, and extra games likely dropped; PoC results not paper-grade. Constraint found: no Super Mario Land ROM on this machine (only Tetris and Pokémon Yellow in ~/Documents/Programming/Projects/dvBoyC/roms/).
- Status: APPROVED
- Human decision: "Change in timeline, I need it done in a couple days (doesn't have to be perfect just something as a Poc)" — 2026-11-03; game: "Tetris (available now)"; scope: "Approve as proposed (Recommended)" — 2026-11-03
- Outcome:
