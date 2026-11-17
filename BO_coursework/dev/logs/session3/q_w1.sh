#!/bin/bash
cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
for s in off_big dev off_dev hold off_hold off_low off_wide practice20; do
  OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/informed_bayes.py algos/w1_a.py algos/w1_b.py algos/w1_c.py --split $s --j 20 --quiet 2>&1 | grep -v frozen
done > logs/session3/w1_all.log
date >> logs/session3/w1_all.log
