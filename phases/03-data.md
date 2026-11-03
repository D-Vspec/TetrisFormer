# Phase 3 — Expert RAM data

**Goal:** exact RAM traces of competent play, split, with the final address set.
**Depends on:** Phase 1 gate (can overlap Phase 2 only with human approval). **Plan refs:** §5.2, §5.3, Milestone 3.

## Checklist

### Source decision
- [ ] **Ask human:** TAS, human recordings, or both (open decision §10) — record answer

### TAS path
- [ ] List glitchless-category TASVideos runs for SML (and Pokémon Red for later): URL, category, emulator, ROM revision, movie format
- [ ] ROM revision of each TAS matches our ROM SHA-1 (if not → ask)
- [ ] Parser for the movie format → per-frame input list; tested on a known file
- [ ] Replay in PyBoy from power-on (match boot ROM / initial conditions the TAS assumes)
- [ ] **Sync check:** compare against checkpoints from the run (level-end frames, final frame count, final screen) — record pass/fail per run
- [ ] If desync: document what was tried; **stop and ask** before falling back to human data

### Human-recording path
- [ ] Recording tool: play in PyBoy, log inputs + RAM per frame
- [ ] Recording protocol written (who plays, how many runs, what counts as "competent")
- [ ] Recordings replay deterministically from logged inputs

### Trace extraction
- [ ] Snapshot full WRAM every agent step (24 frames), same frame phase as the env wrapper
- [ ] Store as `[episode, step, 8192]` uint8 + metadata (source, ROM hash, frame offset); actions discarded
- [ ] Verify: replaying a trace's inputs through the Phase 1 env reproduces the stored RAM exactly

### Data volume
- [ ] Count steps available vs target (~25k windows × 16 steps ≈ 400k steps ≈ 100M tokens at K=256)
- [ ] **If short (likely for TAS): report to human with options** (more runs, human data, frame-offset subsampling, smaller model) — chosen option logged as deviation if it changes the plan

### Address selection (Option B)
- [ ] Bytes that vary during expert play, minus volatile mask
- [ ] Union with published RAM-map addresses
- [ ] Trim/pad to K ≈ 128–512 (K=256 default); selection rule + final list saved to `data/addresses/<game>.json`
- [ ] Per-address note where RAM map gives a meaning (feeds Phase 8)
- [ ] Re-run Phase 2 baseline with this address set (≥1 seed) and confirm it still learns

### Splits
- [ ] Train/val/test split **by episode/run**, not by window (ratios recorded)
- [ ] Dataset card `data/README.md`: sources, ROM hashes, steps per split, K, mask version

## Exit gate (human sign-off)
- [ ] ≥1 synced source with verified traces
- [ ] Data volume reported and accepted by human
- [ ] Address list + splits saved; baseline re-run with final addresses still learns

## Notes / decisions
-
