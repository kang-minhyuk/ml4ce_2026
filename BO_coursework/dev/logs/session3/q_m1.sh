cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
until [ "$(grep -c 'paired diff' logs/session3/l6_sig_offbig.log 2>/dev/null)" -ge 3 ]; do sleep 20; done
OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/informed_bayes.py algos/m1_*.py --split off_big --j 9 --quiet 2>&1 | grep -v "RuntimeWarning\|^  " > logs/session3/m1_knobs_offbig.log
