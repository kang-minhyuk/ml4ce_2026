#!/bin/bash
cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
sleep 5
while pgrep -f "q_v1.sh" >/dev/null; do sleep 20; done
OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/informed_bayes.py algos/v2_capF.py algos/v2_capR.py algos/v2_capFR.py algos/v2_capF1.py --split off_big --j 20 --quiet > logs/session3/v2_offbig.log 2>&1
date >> logs/session3/v2_offbig.log
