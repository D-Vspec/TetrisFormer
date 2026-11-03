# Reward-hacking catalogue

## H-001 — Pause farming (hand-engineered reward)
- Run: `runs/smoke_hand` (400k steps, seed 0), reward = 1.0·Δlines + 0.01/step
- Behavior: agent presses Start ~once per episode → game paused → collects survival reward until MAX_STEPS=2000. Lines = 0.
- Evidence: ep_len → 2000.0 in eval (24/24 episodes), Start count per rollout falls to 0–7 of 1536 actions, rew_mean = 0.01 exactly (no lines ever).
- Responsible signal: the survival term, plus Start = pause in the action set. Pausing presumably freezes the play-state bytes (board, OAM) — check which addresses are static while paused.

## H-002 — Pause-overlay farming (hand reward v2, D-006)
- Run: `runs/hand` (killed at ~1M steps), reward = 1.0·Δlines + 0.1 per piece lock
- Behavior: agent presses Start on ~83% of steps (1276/1536). The PAUSE overlay puts 44 non-empty tiles in the background tilemap (vs 16 real blocks), so each unpause looked like "landed count went up" → counted as a piece lock.
- Evidence: test toggling Start: landed 16 → 44 → 16 → 44; dpieces fired on every second toggle.
- Root cause: my piece-lock detector, not the game. Fix: count a lock only when the landed count rises by exactly 4. Re-verified: 34 pieces, board count matches exactly, pause toggles count 0.
- Note: both hand-reward hacks go through **Start**. That supports the PoC story: hand rewards get hacked in ways that need RAM/screen-level debugging.
