cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
sleep 120
while pgrep -f "policy_search.py final/informed_bayes.py 100 --seed 1" > /dev/null; do sleep 30; done
OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py algos/p1_c221.py algos/q1_fine.py algos/q1_voix.py --split off_big --j 18 --quiet 2>&1 | grep -v "RuntimeWarning\|^  " > logs/session3/q1_offbig.log
