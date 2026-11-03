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
- Status: PROPOSED
- Human decision:
- Outcome:

## D-002 — VIPER exploration term: add as progress-term candidate and include in VIPER baseline
- Date proposed: 2026-11-03
- Phase / item: phases/05-rl-likelihood.md → "Progress term" candidates (a)–(c); phases/06-baselines.md → VIPER reward
- Checklist says: three progress-term candidates; VIPER reward "+ VIPER's own extras if their method includes them".
- Proposed instead: (1) add candidate (d): VIPER-style exploration bonus (RND) on RAM; (2) state explicitly that the VIPER baseline uses r = log p + β·r_expl.
- Reason: VIPER Eq. 6 makes the exploration/entropy term part of the method, derived from KL matching; it exists to fix the same "most-likely ≠ typical / idling" problem our progress term targets. Reviewers will ask why ours beats theirs.
- Downstream impact: one more pilot in Phase 5, one more ablation row in Phase 8.
- Status: PROPOSED
- Human decision:
- Outcome:

## D-003 — Narrow novelty framing of claim 1
- Date proposed: 2026-11-03
- Phase / item: phases/00 → novelty sweep; PLAN.md §3 claim 1, §4
- Checklist says: claim 1 "Predictive likelihood over RAM is a usable dense reward that replaces per-game reward engineering".
- Proposed instead: keep claim, but add to PLAN.md §4 an "observation-only density rewards" group (IL-flOw, f-IRL, density reward modeling) and frame novelty as exact discrete machine state + AR transformer + byte-level interpretability, not likelihood-as-reward itself.
- Reason: IL-flOw (NeurIPS 2021 workshop) already uses expert state-transition log-density as reward on low-dim states. No RAM-likelihood reward paper found.
- Downstream impact: PLAN.md §4 edit; intro framing in Phase 10. Claims list unchanged.
- Status: PROPOSED
- Human decision:
- Outcome:
