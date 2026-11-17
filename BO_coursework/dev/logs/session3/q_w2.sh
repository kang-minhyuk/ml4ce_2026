#!/bin/bash
cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
sleep 5; while pgrep -f "q_g3.sh" >/dev/null; do sleep 20; done
for s in off_big dev off_dev hold off_hold off_low off_wide practice20; do
  OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py algos/w1_c.py algos/w2_c3.py algos/w2_s07.py algos/w2_lf2.py algos/w2_m221.py --split $s --j 20 --quiet 2>&1 | grep -v frozen
done > logs/session3/w2_all.log
PYTHONPATH=. OMP_NUM_THREADS=1 ../.venv/bin/python dbg/pooled.py algos/w1_c.py algos/w2_c3.py algos/w2_s07.py algos/w2_lf2.py algos/w2_m221.py 2>&1 | grep -v frozen >> logs/session3/w2_all.log
date >> logs/session3/w2_all.log
