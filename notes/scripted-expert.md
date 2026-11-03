# Scripted Tetris expert (D-005)

Code: `expert.py`. Data: `collect.py` → `data/traces/expert_*.npz` (full WRAM per agent step).

## Why it exists
The PoC plan took expert data from rollouts of the hand-engineered PPO agent (D-001). In a smoke run, that agent
cleared 0 lines and learned to pause the game (H-001). A PPO agent pressing raw buttons won't learn line
clears within the 2-day budget, so a scripted player supplies the "competent play" RAM traces instead.

## How it works
1. **Perception, read from the emulator screen state, not WRAM.**
   - Landed blocks: background tilemap, columns 2–11 × rows 0–17 (`pb.tilemap_background`). Tile 47 = empty.
   - Falling piece: `game_area()` (background + sprites) minus the background → the 4 sprite cells.
2. **Planning, once per piece.** When a new piece appears (the landed-cell count changed, so the previous piece
   locked), it takes the falling piece's shape, generates its distinct 90° rotations (1–4), and tries every
   rotation × column. Each placement is hard-dropped on a copy of the board, full rows are cleared, and the
   result is scored with a linear heuristic:

   `score = −0.510·aggregate_height + 0.761·lines − 0.357·holes − 0.184·bumpiness`

   (weights from Yiyuan Lee, "Tetris AI – The (Near) Perfect Bot"). The highest score wins. No lookahead on the next piece.
3. **Execution, closed loop, one button per 24-frame env step.**
   - If the on-screen shape ≠ target rotation → press **A** (rotate).
   - else if the left edge is right of / left of the target column → **←** / **→**
   - else → **↓** (soft drop) until it locks.

   It never presses Start, B or ↑. Because it compares against the observed shape, it doesn't need to know
   GB Tetris's rotation centres.
4. It acts through `TetrisEnv.step`, the same action set, frame-skip and RAM snapshot timing as the RL policy.
   The traces therefore match what the reward model sees during RL. Actions are not stored (the reward model is action-free).

## Performance (100 episodes, timer_div = 0…99 seeds the piece sequence, max 2000 steps)
| Metric | Value |
|---|---|
| Lines / episode | mean 53.3, median 58, min 2, max 75 |
| Steps / episode | mean 1830 (71 hit the 2000 cap, 29 topped out) |
| Lines / 100 steps | 2.9 |
| Uniform random policy | 0.0 lines, ~465 steps |
| Total data | 183,044 steps ≈ 52.9M tokens at K=289 |

## Limitations / caveats
- **It's engineered knowledge.** The heuristic and the board parsing are hand-written. The learned reward doesn't
  read any reward code, but it learns to imitate a hand-designed player. That's less circular than D-001
  (no RL reward involved), but it isn't "human play". RESULTS.md must say so.
- **Weak by Tetris-AI standards.** 29% of games top out, from the 1-button-per-24-frames control rate (pieces
  fall while it's still rotating/shifting at higher levels) and from having no next-piece lookahead.
- **Narrow behaviour.** It always rotates first, then shifts, then soft-drops. The reward model will learn this
  stereotyped style, not Tetris in general. Expect agent states that deviate from the style (e.g. shifting
  before rotating) to get lower likelihood even when they're fine.
- **It never pauses**, so pause states are off-distribution for the reward model. Whether the learned reward
  rewards pausing (static RAM is very predictable) or punishes it (never seen) is an open PoC question.
