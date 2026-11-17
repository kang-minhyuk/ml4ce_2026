cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
until [ "$(grep -c 'paired diff' logs/session3/p1_offbig.log 2>/dev/null)" -ge 3 ] && [ "$(grep -c 'paired diff' logs/session3/r1_offbig.log 2>/dev/null)" -ge 3 ]; do sleep 30; done
OMP_NUM_THREADS=1 ../.venv/bin/python policy_search.py final/informed_bayes.py 100 --seed 1 --j 20 2>&1 | grep -v "RuntimeWarning\|^  warn\|frozen" > logs/policy_search/ps1.log
