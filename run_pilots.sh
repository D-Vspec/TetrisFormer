#!/bin/sh
# D-008 pilots: full method, 300k steps each, sequential
.venv/bin/python rl.py --name pilot_a --ent 0.05 --beta 0.5 --steps 300000 > runs_pilot_a.log 2>&1
.venv/bin/python rl.py --name pilot_b --ent 0.05 --beta 2.0 --steps 300000 > runs_pilot_b.log 2>&1
