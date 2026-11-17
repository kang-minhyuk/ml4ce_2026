#!/bin/bash
cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
for s in dev off_dev hold off_hold off_low off_wide off_fresh; do
  OMP_NUM_THREADS=1 PYTHONPATH=. ../.venv/bin/python evaluate.py final/generic_bayes2.py algos/g6_ls15.py algos/g6_nl96.py algos/g6_lsd.py --split $s --j 20 --quiet 2>&1 | grep -v frozen | grep "runs per\|paired"
done
PYTHONPATH=. ../.venv/bin/python dbg/pooled.py final/generic_bayes2.py algos/g6_ls15.py algos/g6_nl96.py algos/g6_lsd.py 2>&1 | grep -v frozen
date
