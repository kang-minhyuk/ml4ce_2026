#!/bin/bash
# Re-evaluates every algorithm listed in ALGORITHMS.md (results are cached by file content, so this is cheap to rerun).
cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
export OMP_NUM_THREADS=1
PY=/home/bruno/Documents/ml4ce_2026-main/BO_coursework/.venv/bin/python
REF="../student/algorithms/reference_mfbo.py ../student/algorithms/reference_screen_then_confirm.py algos/mfgp_structured.py"
FIN="final/generic_funnel.py final/generic_bayes.py final/generic_bayes2.py final/informed_funnel.py final/informed_library.py final/informed_bayes.py final/informed_bayes2.py"
MIL="algos/i3_funnel.py algos/i6_funnel.py algos/i10_funnel.py algos/i14_funnel.py algos/i17_funnel.py algos/k5_bayes.py algos/k7_voi_s2.py"
# k5/k7 also on off_big (cached)
REF2="../student/algorithms/reference_screen_then_confirm.py algos/mfgp_structured.py"  # reference_mfbo (~160 s/run): off_big + practice20 only
for s in off_big dev off_dev hold off_hold off_low off_wide practice20; do
  echo "== $s"
  R=$REF2; if [ $s = off_big ] || [ $s = practice20 ]; then R=$REF; fi
  $PY evaluate.py $R $FIN --split $s --j 12 --quiet 2>&1 | grep -v "RuntimeWarning\|^  "
done
for s in dev off_dev practice20; do
  echo "== milestones $s"
  $PY evaluate.py $MIL --split $s --j 12 --quiet 2>&1 | grep -v "RuntimeWarning\|^  "
done
echo "== milestones off_big"
$PY evaluate.py algos/k5_bayes.py algos/k7_voi_s2.py --split off_big --j 12 --quiet 2>&1 | grep -v "RuntimeWarning\|^  "
echo "== DONE"
