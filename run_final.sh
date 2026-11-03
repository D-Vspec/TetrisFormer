#!/bin/sh
# D-009: 300k steps each, D-008 config. ours = runs/pilot_a (already done).
.venv/bin/python rl.py --name ours_nomask --rm runs/rm_nomask/model.pt --ent 0.05 --beta 0.5 --steps 300000 > runs_ours_nomask.log 2>&1 &
.venv/bin/python rl.py --name curiosity --rm runs/rm/model.pt --curiosity --ent 0.05 --beta 0.5 --steps 300000 > runs_curiosity.log 2>&1 &
wait
.venv/bin/python ppo.py --name hand_final --ent 0.05 --steps 300000 > runs_hand_final.log 2>&1
