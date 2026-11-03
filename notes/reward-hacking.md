# Reward-hacking catalogue

## H-001 — Pause farming (hand-engineered reward)
- Run: `runs/smoke_hand` (400k steps, seed 0), reward = 1.0·Δlines + 0.01/step
- Behavior: agent presses Start ~once per episode → game paused → collects survival reward until MAX_STEPS=2000. Lines = 0.
- Evidence: ep_len → 2000.0 in eval (24/24 episodes), Start count per rollout falls to 0–7 of 1536 actions, rew_mean = 0.01 exactly (no lines ever).
- Responsible signal: the survival term, plus Start = pause in the action set. Pausing presumably freezes the play-state bytes (board, OAM) — check which addresses are static while paused.
