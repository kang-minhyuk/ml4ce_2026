cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
until [ "$(grep -c 'paired diff' logs/session3/l3_sig_offbig.log)" -ge 3 ]; do sleep 20; done
OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/informed_bayes.py algos/l4_inc.py algos/l4_inc5.py algos/l4_inc_s1.py --split off_big --j 6 --quiet 2>&1 | grep -v "RuntimeWarning\|^  " > logs/session3/l4_inc_offbig.log
