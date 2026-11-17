#!/bin/bash
cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
sleep 5; while pgrep -f "q_re[g].sh" >/dev/null; do sleep 20; done
OMP_NUM_THREADS=1 ../.venv/bin/python policy_search.py final/informed_bayes2.py 80 --seed 2 --j 20 2>&1 | grep -v frozen > logs/policy_search/ps2.log
date >> logs/policy_search/ps2.log
