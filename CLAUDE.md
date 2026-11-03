# CLAUDE.md

Project: **Learning Rewards from Machine State** — a RAM-only reward transformer for Game Boy RL.
Full plan: `PLAN.md`. Work is driven by the phase checklists in `phases/`.

---

## 0. The checklist is the contract (highest priority — overrides everything below)

1. **Never deviate from the checklist without explicit human approval.** A deviation is anything that
   differs from what a checklist item says: skipping, reordering, substituting a method/tool/library,
   changing a number (model size, T, K, frame-skip, seeds, thresholds), adding scope, merging items, or
   starting a later phase before the current phase's exit gate is passed.
2. **Stop and ask before deviating.** Explain: what the item says, what you want to do instead, why,
   and what it affects downstream. Then wait. Do not proceed "provisionally".
3. **Always write deviations down** in `DEVIATIONS.md` — proposed, approved, *and* rejected — using the
   template there. Record it *when proposed*, then update the status once the human answers. Quote the
   human's approval verbatim. You may never mark a deviation "approved" yourself.
4. **Branches inside an item** ("if X, do Y; else Z") are not deviations, but *which* branch was taken
   and why gets noted under that phase's `Notes / decisions` section. If an item says *ask human*, ask.
5. **Check a box only with evidence.** `[x]` means done *and verified* — add a short pointer to the
   evidence (file, commit, log, number) after the item. Unverified = unchecked. Partially done = `[~]`
   with a note.
6. **One phase at a time.** Work the lowest-numbered phase whose exit gate is not passed. Gates are
   checked by the human, not self-certified: report gate status and ask for sign-off.
7. **Open decisions** (PLAN.md §10) are the human's. Present options + a recommendation; don't pick.
8. If the checklist turns out wrong, impossible, or contradicts `PLAN.md`, that is a deviation: stop,
   explain, log, ask.

### Phase index

| # | Phase | File |
|---|---|---|
| 0 | Literature sweep & decisions | `phases/00-literature-and-decisions.md` |
| 1 | Environment | `phases/01-environment.md` |
| 2 | Hand-engineered baseline (SML) | `phases/02-baseline.md` |
| 3 | Expert RAM data | `phases/03-data.md` |
| 4 | Reward model v1 | `phases/04-reward-model.md` |
| 5 | RL with likelihood reward | `phases/05-rl-likelihood.md` |
| 6 | Learned-reward baselines | `phases/06-baselines.md` |
| 7 | More games | `phases/07-more-games.md` |
| 8 | Ablations & interpretability | `phases/08-ablations-interpretability.md` |
| 9 | Transfer (stretch) | `phases/09-transfer.md` |
| 10 | Write-up | `phases/10-write-up.md` |

Research notes go in `notes/`.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted (then log it as a proposed deviation).
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request or a checklist item.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant
clarification. Each checklist item should end in something checkable; if it doesn't, ask what "done" means.

---

## 5. Research hygiene (project-specific)

- Every experiment: fixed seeds, logged config, logged git commit. No numbers in the paper without a run behind them.
- Never commit ROMs. Record ROM SHA-1 hashes instead.
- Don't overstate results — especially transfer (PLAN.md §2 note, §6.6).
- Keep the claims to PLAN.md §3.

**Tradeoff:** these guidelines bias toward caution over speed. For trivial tasks (typos, formatting a
note) use judgment — but the checklist rules in §0 always apply.
