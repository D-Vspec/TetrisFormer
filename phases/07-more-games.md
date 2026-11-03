# Phase 7 — More games

**Goal:** repeat the full pipeline on Pokémon Red and 1–2 more titles.
**Depends on:** Phase 6 gate. **Plan refs:** §6.1, Milestone 7.

## Checklist

### Game list
- [ ] **Ask human:** final game list (open decision §10). Candidates: Kirby's Dream Land, Tetris; prefer overlap with Rank-Then-Act titles
- [ ] ROMs obtained, hashes recorded in `notes/roms.md`

### Per game — copy this block for each title

#### <Game>
- [ ] Start save state + episode length / termination rule defined
- [ ] Progress metric defined (Pokémon: badges, maps visited, story flags) and verified in emulator
- [ ] Volatile mask (Phase 1 procedure)
- [ ] Hand-engineered reward (Pokémon: PokeRL-style terms) — Phase 2 procedure, pre-registered threshold, ≥3 seeds
- [ ] Expert data (Phase 3 procedure): sync check, volume check, address set, splits
- [ ] Reward model (Phase 4 procedure): beats trivial baselines, likelihood ordering holds
- [ ] Full method RL (Phase 5 Run 3 protocol), ≥3 seeds
- [ ] Curiosity + VIPER baselines (Phase 6 protocol), ≥3 seeds
- [ ] Reward-hacking behaviors added to `notes/reward-hacking.md`
- [ ] Any per-game change to the shared pipeline → deviation entry

### Cross-game
- [ ] Full table: N games × all reward rows, mean ± std
- [ ] "% of hand-engineered" and "beats VIPER/curiosity on Y of N games" computed

## Exit gate (human sign-off)
- [ ] All chosen games have complete rows

## Notes / decisions
-
