#!/usr/bin/env bash
set -euo pipefail

readonly WORKSPACE="/home/jovyan/work"
readonly TEMPLATES="/opt/course/templates"

mkdir -p "${WORKSPACE}" "${WORKSPACE}/files"
mkdir -p "${WORKSPACE}" "${WORKSPACE}/files/outputs"
.
copy_tree_if_missing() {
    local source="$1"
    local destination="$2"

    if [[ ! -e "${destination}" ]]; then
        cp -R --no-preserve=mode "${source}" "${destination}"
    fi
}

if [[ ! -f "${WORKSPACE}/RL_coursework.ipynb" ]]; then
    cp --no-preserve=mode \
        "${TEMPLATES}/RL_coursework.ipynb" \
        "${WORKSPACE}/RL_coursework.ipynb"
fi

copy_tree_if_missing \
    "${TEMPLATES}/algorithms" \
    "${WORKSPACE}/algorithms"

copy_tree_if_missing \
    "${TEMPLATES}/data" \
    "${WORKSPACE}/data"

# Repair workspaces that were seeded read-only by an earlier version of this image.
chmod -R u+w "${WORKSPACE}/algorithms" "${WORKSPACE}/RL_coursework.ipynb" 2>/dev/null || true

exec "$@"