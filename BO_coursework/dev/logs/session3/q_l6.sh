cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
until [ "$(grep -c 'paired diff' logs/session3/l5_race_offbig.log 2>/dev/null)" -ge 3 ]; do sleep 20; done
OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/informed_bayes.py algos/l6_s1.py algos/l6_s2.py algos/l6_s1_inc.py --split off_big --j 6 --quiet 2>&1 | grep -v "RuntimeWarning\|^  " > logs/session3/l6_sig_offbig.log
