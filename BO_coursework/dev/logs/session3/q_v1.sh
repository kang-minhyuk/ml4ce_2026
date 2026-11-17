#!/bin/bash
cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
while pgrep -f "q_t1.sh" >/dev/null; do sleep 20; done
OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/informed_bayes.py algos/v1_f03.py algos/v1_f03L.py algos/v1_f03L1.py algos/v1_f05L.py algos/v1_f07L.py --split off_big --j 20 --quiet > logs/session3/v1_offbig.log 2>&1
date >> logs/session3/v1_offbig.log
OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/informed_bayes.py algos/u1_bs05.py algos/u1_bs065.py --split off_big --j 20 --quiet > logs/session3/u1_offbig.log 2>&1
date >> logs/session3/u1_offbig.log
