#!/bin/sh
# D-010: all proposed 3M-step runs (D-008 config). 3 jobs at a time, longest first.
cd "$(dirname "$0")"
R=".venv/bin/python rl.py --ent 0.05 --beta 0.5 --steps 3000000"
H=".venv/bin/python ppo.py --ent 0.05 --steps 3000000"
xargs -P 3 -I{} sh -c '{}' <<JOBS
$R --name ours_3m_s0 --seed 0 > runs_ours_3m_s0.log 2>&1
$R --name ours_3m_s1 --seed 1 > runs_ours_3m_s1.log 2>&1
$R --name ours_3m_s2 --seed 2 > runs_ours_3m_s2.log 2>&1
$R --name nomask_3m_s0 --seed 0 --rm runs/rm_nomask/model.pt > runs_nomask_3m_s0.log 2>&1
$R --name curiosity_3m_s0 --seed 0 --curiosity > runs_curiosity_3m_s0.log 2>&1
$H --name hand_3m_s0 --seed 0 > runs_hand_3m_s0.log 2>&1
$H --name hand_3m_s1 --seed 1 > runs_hand_3m_s1.log 2>&1
$H --name hand_3m_s2 --seed 2 > runs_hand_3m_s2.log 2>&1
JOBS
echo ALL_DONE
