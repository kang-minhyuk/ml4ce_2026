#!/bin/bash
cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
for s in off_big dev off_dev hold off_hold off_low off_wide practice20; do
  OMP_NUM_THREADS=1 PYTHONPATH=. ../.venv/bin/python evaluate.py final/informed_bayes2.py algos/x4_m20.py algos/x4_m20b.py algos/x2_b05.py --split $s --j 20 --quiet 2>&1 | grep -v frozen | grep "runs per\|paired"
done
PYTHONPATH=. ../.venv/bin/python dbg/pooled.py final/informed_bayes2.py algos/x4_m20.py algos/x4_m20b.py algos/x2_b05.py 2>&1 | grep -v frozen
date
