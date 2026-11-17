#!/bin/bash
cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
while pgrep -f "evaluate.py algos/p1_c221.py algos/q1_fine.py" >/dev/null; do sleep 20; done
OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/informed_bayes.py algos/t1_cp9.py algos/t1_cp2.py algos/t1_cp9m1.py algos/t1_cp9_c221.py --split off_big --j 20 --quiet > logs/session3/t1_offbig.log 2>&1
date >> logs/session3/t1_offbig.log
