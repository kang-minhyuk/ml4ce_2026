cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/informed_bayes.py algos/m1_m221.py algos/m2_*.py --split off_big --j 10 --quiet 2>&1 | grep -v "RuntimeWarning\|^  " > logs/session3/m2_offbig.log
