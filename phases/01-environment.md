# Phase 1 — Environment

**Goal:** fast, deterministic, parallel PyBoy environment that exposes RAM, plus the volatile-byte mask.
**Depends on:** Phase 0 gate. **Plan refs:** §5.1, §5.3, §5.7, Milestone 1.

## Checklist

### Setup
- [ ] Python env created; versions of python, pyboy, torch pinned in a lockfile / requirements
- [ ] Legally obtained ROMs placed outside git (`roms/` in `.gitignore`); SHA-1 + ROM revision recorded in `notes/roms.md`
- [ ] `.gitignore` covers ROMs, save states, checkpoints, large data

### Single env wrapper
- [ ] PyBoy runs headless with unlimited speed; single-env raw FPS measured and recorded
- [ ] Action set = 8: ↑ ↓ ← → A B Start, no-op
- [ ] Press/release semantics defined and documented (how long a button is held within the 24-frame step)
- [ ] Frame-skip = 24 frames per agent step
- [ ] `reset()` starts from a fixed save state past boot/intro (one per game); state file hash recorded
- [ ] `step()` returns RAM observation, done flag, info
- [ ] Episode termination: game over detection + max-steps limit (value recorded)

### RAM observation
- [ ] Read WRAM 0xC000–0xDFFF (8192 bytes) as uint8 each step
- [ ] HRAM 0xFF80–0xFFFE optional — off by default, flag exists only if needed later (ask before adding)
- [ ] CPU registers excluded (documented why: snapshot-timing artifacts)
- [ ] Verified reads against a known address from a published SML RAM map (e.g. lives / x-pos changes as expected)

### Determinism
- [ ] Test: same start state + same action sequence → byte-identical RAM trace (run twice, compare)
- [ ] Test passes across separate processes (not just in-process)

### Parallel envs
- [ ] Vectorized env over subprocesses, 16 envs working
- [ ] Throughput measured at 16 / 32 / 64 envs (agent steps/s) and recorded below
- [ ] No cross-env state leakage (each env's trace matches a single-env replay)

### Volatile-byte mask
- [ ] Rollouts: N no-op episodes + N random-action episodes from the same start state (N recorded)
- [ ] Flag bytes that change on (nearly) every step regardless of input (threshold recorded)
- [ ] Flag bytes that differ across no-op runs from identical state (RNG-like) — expect none if deterministic; investigate if any
- [ ] Save mask as `data/masks/<game>_volatile.npy` + JSON metadata (threshold, N, ROM hash)
- [ ] Spot-check flagged addresses against published RAM map (frame counter, timer, RNG expected)
- [ ] Report: number of masked bytes / 8192

## Exit gate (human sign-off)
- [ ] Determinism tests pass
- [ ] ≥16 parallel envs with measured throughput
- [ ] SML volatile mask saved and spot-checked

## Notes / decisions
-
