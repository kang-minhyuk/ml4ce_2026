#!/usr/bin/env bash
set -euo pipefail

readonly WORKSPACE="/home/jovyan/work"
readonly TEMPLATES="/opt/course/templates"

mkdir -p "${WORKSPACE}" "${WORKSPACE}/outputs"

# Docker may create missing bind-mount directories as root. Permit the host
# user and the unprivileged notebook user to work without matching numeric UIDs.
chmod a+rwx "${WORKSPACE}" "${WORKSPACE}/outputs"

copy_tree_if_missing() {
    local source="$1"
    local destination="$2"

    if [[ ! -e "${destination}" ]]; then
        cp -R --no-preserve=mode "${source}" "${destination}"
    fi
}

copy_file_if_missing() {
    local source="$1"
    local destination="$2"

    if [[ ! -f "${destination}" ]]; then
        cp --no-preserve=mode "${source}" "${destination}"
    fi
}

copy_file_if_missing "${TEMPLATES}/README.md" "${WORKSPACE}/README.md"
copy_file_if_missing "${TEMPLATES}/RL_coursework.ipynb" "${WORKSPACE}/RL_coursework.ipynb"

copy_tree_if_missing \
    "${TEMPLATES}/algorithms" \
    "${WORKSPACE}/algorithms"

# Under Compose, data/training is a read-only bind mount. This fallback keeps
# the image usable with docker run without trying to replace a mount point.
copy_tree_if_missing \
    "${TEMPLATES}/data" \
    "${WORKSPACE}/data"

# Repair workspaces that were seeded read-only by an earlier version of this image.
chmod a+rw "${WORKSPACE}/RL_coursework.ipynb"
chmod -R a+rwX "${WORKSPACE}/algorithms" "${WORKSPACE}/outputs"

exec /usr/local/bin/start.sh "$@"
