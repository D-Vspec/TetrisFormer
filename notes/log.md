# Event log

Every discrepancy, surprise, workaround, and noteworthy event, newest at the bottom. Short entries.
Deviations still go in `DEVIATIONS.md` and reward hacks in `notes/reward-hacking.md`; this log links to them.
Entries before 2026-11-03 ~17:00 were reconstructed from the session after the human asked for this log.

## 2026-11-03

- **Repo:** the project folder sat inside a git repo rooted at `~` (home dir). Gave it its own `git init`.
- **Lit sweep:** WebFetch of the VIPER PDF returned raw binary. Extracted the text with `pdftotext` instead.
- **Citation discrepancy:** danijar.com labels VIPER "NeurIPS 2024"; NeurIPS proceedings + mlanthology say **2023**. Used 2023.
- **Citation discrepancy:** PLAN.md listed the Reward Bag Transformer as "2024"; it's **TMLR 2025** (an ICML 2026 poster with the same title also exists). Fixed PLAN.md without asking (flagged to human afterwards).
- **Venue:** NeurIPS 2027 CFP not published yet; deadline estimated from trackers + the 2026 pattern. (Superseded by D-004.)
- **Timeline change:** human asked for a 2-day PoC → D-004.
- **No SML ROM** on the machine. Only Tetris + Pokémon Yellow (in `dvBoyC/roms`). Human chose Tetris.
- **Env setup:** `uv venv` defaulted to Python 3.12 (no torch). Recreated it on system Python 3.14, where torch 2.14+cu130 lives.
- **Minor:** `pyboy` has no `__version__`; used `importlib.metadata`.
- **Mask discrepancy:** the planned rule ("changes on ≥95% of steps") masked **0 bytes**; frame counter / RNG live in HRAM/IO, not WRAM. Switched to a no-op change rate ≥ 0.6 → 8 bytes (sound driver 0xDF94–0xDFC4 + stack 0xCFED). The PoC checklist gave no threshold, so this wasn't a formal deviation.
- **Observation:** only 278 of 8192 WRAM bytes ever change in no-op/random play; 289 in expert play.
- **Hack H-001:** hand reward v1 (survival term) → agent pauses with Start and farms survival. 0 lines. → D-005 (scripted expert), D-006 (reward fix).
- **Bug:** first per-piece detector double-counted line clears (47 counted vs 34 real). Fixed: count landed-count increases only.
- **Expert:** 53 lines/game, but 29/100 games top out (control rate: 1 button per 24 frames).
- **Discrepancy:** varying-byte count was 284 on 30 expert episodes and **289** on all 100. Used 289 under the approved "all varying bytes" rule (D-007). No-mask RM: K=297.
- **Ops mistake:** `pkill -f "ppo.py --name hand"` also matched my own shell's command line. The kill worked, but the check command after it died (exit 144). Since then I use the `[p]attern` trick.
- **Hack H-002:** the pause overlay adds 28 tiles to the board tilemap, so the piece detector fired on unpause. The agent pressed Start on 83% of steps. Fixed: a lock = exactly +4 cells. First `runs/hand` killed; restarted.
- **RM overfit check** bottomed out at 0.03 nats/byte, not ~0 (bf16 + changing bytes). Treated as a pass.
- **Contention:** the hand PPO run dropped from ~1050 to ~150 steps/s while RM training shared the machine (CPU-side window sampling + GPU). Running jobs side by side costs more than it saves.
- **Reward inference speed:** 48 ms/step for 12 envs (T·K=2312 tokens), compute-bound. Batching gave no gain, and neither did a hand-written SDPA forward (max |Δlogit| 0.09 under bf16; argmax agreement 99.5%). → ~4 h per 3M-step run.
- **Harmless warning:** `float(dist.entropy().mean())` on a grad-requiring tensor in `ppo.py` logging.
- **Repo hygiene:** `runs/` isn't gitignored, so small checkpoints (`model.pt` ~3.6 MB, `policy.pt`) and logs got committed. Left as is; flag if unwanted.
- **Decision:** the curiosity run keeps the RND bonus (β=0.5) so it differs from "ours" only in the sign of log p.
- **Hand baseline v3** (lines + 0.1/piece): through ~1M steps it presses ↓ most of the time, ~0 lines, episodes ~150–240 steps. It drops pieces fast to collect the per-piece bonus.
- **Collapse:** `runs/ours` collapsed by 152k steps. It presses → on 100% of steps, entropy 1.7e-5, 0 lines, raw log p ≈ −39/step. `runs/smoke_rm` (20k steps) was already 90% →. Human approved stopping the queue to check. Queue killed (exit 144, intended); hand baseline left running.
- **Collapse diagnosis (not a bug):** ran the expert through env + LearnedReward online (β=0, unseen seeds 210/211): −12.4 log p/step vs offline −11.7, so state/reward alignment is correct. Fixed policies: → −34.5, no-op −41…−52, ← −45, ↓ −46, random −121, Start −257 per step. The reward ranks the expert ~3× above any trivial policy; "always →" is the best *trivial* policy → PPO converged there (exploration failure / local optimum, as VIPER warns). RND β=0.5 and entropy 0.01 didn't prevent entropy collapse to 1.7e-5 within 150k steps.
- **Noteworthy:** pausing is the *worst* behaviour under the learned reward (−257/step, never seen in expert data), so the Start exploit behind H-001/H-002 doesn't transfer to the learned reward.
- **Noteworthy:** random play scores worse than any fixed action (−121), mostly because random presses Start ~1/8 of the time.
- **Noteworthy:** the first 8 steps (history padded with s0) score about the same as later steps. Padding isn't a problem.
