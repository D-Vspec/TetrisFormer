#!/bin/sh
# PoC P5 runs, sequential (GPU-bound reward inference). Same PPO config/budget as runs/hand.
set -e
.venv/bin/python rl.py --name ours        --rm runs/rm/model.pt        --steps 3000000 > runs_ours.log 2>&1
.venv/bin/python rl.py --name ours_nomask --rm runs/rm_nomask/model.pt --steps 3000000 > runs_ours_nomask.log 2>&1
.venv/bin/python rl.py --name curiosity   --rm runs/rm/model.pt --curiosity --steps 3000000 > runs_curiosity.log 2>&1
