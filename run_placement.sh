#!/bin/sh
# D-012: placement-level actions, 1M placements, 1 seed each, in parallel
cd "$(dirname "$0")"
.venv/bin/python rl.py --name ours_place_s0 --placement --ent 0.05 --beta 0.5 --steps 1000000 --seed 0 > runs_ours_place_s0.log 2>&1 &
.venv/bin/python ppo.py --name hand_place_s0 --placement --ent 0.05 --steps 1000000 --seed 0 > runs_hand_place_s0.log 2>&1 &
wait
echo ALL_DONE
