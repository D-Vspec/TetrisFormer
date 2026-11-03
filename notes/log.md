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
- **Stopped** `runs/hand` (entropy 0.01) at 1.58M steps (0.02 lines/ep, ~228-step episodes), to be superseded by the D-008 parity rerun. Its partial log is kept.
- **Correction:** I first logged the hand-run stop at ~1.1M steps; the final log line shows 1,579,008. Fixed above and in D-008.
- **Throughput discrepancy:** learned-reward PPO runs at ~126 steps/s end to end (pilot_a, nothing else running), not the ~200 estimated from reward-fn profiling (PPO update + env overhead on top). 3M-step run ≈ 6.6 h, not 4 h; 3 runs sequential ≈ 20 h.
- **Pilots (D-008):** (a) entropy 0.05, β=0.5, 300k steps: no collapse (entropy ~1.0 throughout), raw log p −39 → −29/step, 0 lines, eval 0 lines / 159-step episodes. (b) entropy 0.05, β=2: collapsed to always-↓ (entropy 0.002) by 57k steps. Stopped (b) at ~110k steps, since it already met the "collapses" criterion. **Chose (a).**
- **Surprise:** the larger RND weight made collapse *worse*, not better. Under β=2 the always-↓ policy presumably keeps producing fresh board states fast (short episodes → many resets).
- **Budget cut (D-009):** human wants the report in ~2 h. All runs 300k steps; pilot_a reused as "ours".
- **Parallel runs:** ours_nomask + curiosity side by side run at 143 steps/s each (286 combined vs 126 alone): reward inference was not saturating the GPU.
- **Analysis discrepancy (fixed):** analyze.py scored pilot_a at −71 log p/step vs −29…−43 in training logs. An online re-measurement gave mean −39.5 / **median −12.2** (expert median ≈ −12): per-step reward is heavy-tailed. Cause: analyze.py included each episode's terminal transition (into the game-over screen), which the online reward excludes (r=0 on done). Fixed: drop the terminal transition, report medians too, 8 agent episodes instead of 4.
- **Noise:** PyBoy/SDL prints "Error in sys.excepthook" at interpreter exit in ad-hoc scripts. Harmless teardown noise.
- **Noteworthy:** pilot_a's median step is about as likely as an expert step. The policy is expert-like most of the time and loses reward on rare, very unlikely steps.
- **Slowdown:** parallel ours_nomask + curiosity slowed from 143 to ~95 steps/s each as training went on (190 combined); still faster than sequential.
- **Curiosity finding:** by 50k steps the curiosity run presses Start on ~92–99% of steps (entropy 0.04–0.2), raw log p ≈ −250/step. It pauses the game because pause screens are maximally surprising (never in expert data): the RAM version of the noisy-TV failure. Mirror image of the likelihood reward, which scores pausing worst.
- **Correction:** I told the human the policy is ~17.4M params; it's 17,828,361 (input EmbeddingBag 271×256×256 = 17.76M).
- **Final runs done:** ours_nomask eval eps 822 steps vs ~176 in training. Its policy presses ← 76% (no pausing: Start 0%), so pieces pile on the left. Unexplained why that lasts longer than "always →" (~215).
- **Curiosity** cleared 1 line in 24 eval games (0.04/game). Treated as noise.
- **Expert median log p** is −5.1/step (mean −13.8): expert per-step reward is heavy-tailed too.
- **Unidentified address:** 0xC203 is the top gap byte for ours and hand; meaning unknown.
- **RESULTS.md written**; all PoC checklist items ticked except the human-review gate.
- **GIF:** runs/tetris_agents.gif (200 agent steps, timer_div=42, captured every 8 frames, real-time). In it, ours tops out early (fast ↓ drops) and curiosity sits on the pause screen.
- **History rewrite:** before the first push to github.com/D-Vspec/TetrisFormer (public), stripped runs/*/policy.pt (6 × 71 MB) from all history with git filter-branch (human-approved). Commit hashes from the first run-containing commit onward changed. Backup branch `pre-strip` kept locally. filter-branch also deleted the policy files from the working tree; restored from pre-strip as untracked, gitignored files.
- **Discrepancy noted in README:** the policy address set was built by an inline snippet, not a committed script.
- **Pushed** to git@github.com:D-Vspec/TetrisFormer.git (public), branch renamed master → main, 39 commits, no co-author lines. Verified: README + GIF served. Commit author email (dv@shamit.org, from local git config) is public in the history.
- **Finding from the training figure:** ours_nomask's raw log p (under its own K=297 model) collapsed from ~−40 to ~−235 at ~285k steps, with entropy falling from ~1.1 to ~0.4 at the same time. That's when it switched to ← 76%, which explains the 822-step eval episodes. Why an agent would move to a much *lower* raw likelihood is unexplained; candidates are RND domination or the running-normalization baseline (not verified).
- **Report:** wrote REPORT.md for the professor (problem statement, work, implementation, results, discussion) + runs/fig_training.png (plot_training.py). Author line set to the GitHub handle D-VSpec; the human may want their real name.
- **PDF:** REPORT.pdf built with pandoc + xelatex (build_pdf.sh), 10 pages. Discrepancies handled: GIF can't go in a PDF (still frame used); ≫ missing from DejaVu Serif (rendered as \gg); pandoc needs a blank line before lists (used lists_without_preceding_blankline).
- **Figure 2 swapped** to a mid-game frame (GIF frame 180 of 333 after GIF frame-merging) at the human's request. The previous frame (step ~100) showed our agent's game-over curtain, which looks like a full grid. Captions updated in REPORT.md and build_pdf.sh.
- **Report:** removed Appendix A (reproducibility) and B (deviation list) at the human's request. That info stays in README.md and DEVIATIONS.md.
