cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
until [ -s logs/session3/m2_offbig.log ] && [ "$(grep -c 'paired diff' logs/session3/m2_offbig.log)" -ge 8 ]; do sleep 30; done
OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/informed_bayes.py algos/n1_*.py --split off_big --j 10 --quiet 2>&1 | grep -v "RuntimeWarning\|^  " > logs/session3/n1_offbig.log
