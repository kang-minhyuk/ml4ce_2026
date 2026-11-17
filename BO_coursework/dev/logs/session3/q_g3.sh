#!/bin/bash
cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
sleep 5; while pgrep -f "q_w1.sh" >/dev/null; do sleep 20; done
OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/generic_bayes.py algos/g3_cp9.py algos/g3_cp9_m221.py --split off_big --j 20 --quiet 2>&1 | grep -v frozen > logs/session3/g3_offbig.log
date >> logs/session3/g3_offbig.log
