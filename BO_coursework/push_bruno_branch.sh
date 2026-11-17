#!/bin/bash
# Push the whole ml4ce_2026-main folder to https://github.com/kang-minhyuk/ml4ce_2026 on branch "bruno".
# Needs: git installed, your git name/email set, and GitHub access (gh auth login, or an SSH key / token).
# Excluded (local or regenerable): .venv, scenario_suite/results, scenario_suite/.labs, dev/.labs_off, __pycache__.
set -euo pipefail

SRC=/home/bruno/Documents/ml4ce_2026-main
REPO=https://github.com/kang-minhyuk/ml4ce_2026.git
BRANCH=bruno

command -v git >/dev/null || { echo "git is not installed: sudo apt install git"; exit 1; }
git config user.name >/dev/null && git config user.email >/dev/null || {
  echo "Set your git identity first: git config --global user.name ... ; git config --global user.email ..."; exit 1; }

WORK=$(mktemp -d)
git clone "$REPO" "$WORK/repo"
cd "$WORK/repo"
if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git checkout -b "$BRANCH" "origin/$BRANCH"
else
  git checkout -b "$BRANCH"
fi

rsync -a \
  --exclude='/.git/' \
  --exclude='/BO_coursework/.venv/' \
  --exclude='/BO_coursework/scenario_suite/results/' \
  --exclude='/BO_coursework/scenario_suite/.labs/' \
  --exclude='/BO_coursework/dev/.labs_off/' \
  --exclude='__pycache__/' \
  --exclude='.ipynb_checkpoints/' \
  "$SRC/" "$WORK/repo/"

git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit: branch $BRANCH already matches the folder."
else
  echo "Files to commit: $(git diff --cached --name-only | wc -l)"
  git commit -q -F - <<'MSG'
BO coursework: algorithm development folder (dev/) and final algorithms

dev/final/informed_bayes2.py is the best informed algorithm (0.742 on 400 untouched
official-style scenarios) and dev/final/generic_bayes2.py the best generic one (0.478).
dev/ALGORITHMS.md is the score register, dev/SUMMARY.md the summary and dev/NOTES.md
the full development log. Local caches and the virtualenv are not included.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
fi
git push -u origin "$BRANCH"
echo "Pushed to https://github.com/kang-minhyuk/ml4ce_2026/tree/$BRANCH (work copy: $WORK/repo)"
