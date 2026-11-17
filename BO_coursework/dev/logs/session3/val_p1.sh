cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
for s in dev off_dev hold off_hold off_low off_wide practice20; do echo "== $s"; OMP_NUM_THREADS=1 ../.venv/bin/python evaluate.py final/informed_bayes.py algos/p1_c221.py --split $s --j 2 --quiet 2>&1 | grep -v "RuntimeWarning\|^  "; done; echo "== DONE"
