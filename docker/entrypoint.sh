#!/usr/bin/env bash
set -euo pipefail

CURRENT_UID="$(id -u)"
CURRENT_GID="$(id -g)"

if [ "${CURRENT_UID}" != "0" ] && { ! getent passwd "${CURRENT_UID}" >/dev/null || ! getent group "${CURRENT_GID}" >/dev/null; }; then
    NSS_WRAPPER_DIR="${NSS_WRAPPER_DIR:-/tmp/enpm690_nss_wrapper}"
    NSS_WRAPPER_PASSWD="${NSS_WRAPPER_PASSWD:-${NSS_WRAPPER_DIR}/passwd}"
    NSS_WRAPPER_GROUP="${NSS_WRAPPER_GROUP:-${NSS_WRAPPER_DIR}/group}"
    CONTAINER_USER_NAME="${CONTAINER_USER_NAME:-enpm690}"
    CONTAINER_GROUP_NAME="${CONTAINER_GROUP_NAME:-enpm690}"

    mkdir -p "${NSS_WRAPPER_DIR}"
    cp /etc/passwd "${NSS_WRAPPER_PASSWD}"
    cp /etc/group "${NSS_WRAPPER_GROUP}"

    if ! getent passwd "${CURRENT_UID}" >/dev/null; then
        printf '%s:x:%s:%s:ENPM690 runtime user:%s:/bin/bash\n' \
            "${CONTAINER_USER_NAME}" \
            "${CURRENT_UID}" \
            "${CURRENT_GID}" \
            "${HOME:-/tmp/enpm690_cache/home}" \
            >> "${NSS_WRAPPER_PASSWD}"
    fi

    if ! getent group "${CURRENT_GID}" >/dev/null; then
        printf '%s:x:%s:\n' \
            "${CONTAINER_GROUP_NAME}" \
            "${CURRENT_GID}" \
            >> "${NSS_WRAPPER_GROUP}"
    fi

    export NSS_WRAPPER_PASSWD
    export NSS_WRAPPER_GROUP
    export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}/usr/lib/x86_64-linux-gnu/libnss_wrapper.so"
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate unitree_sim_env

# Defaults mirror the Dockerfile but remain overrideable at `docker run` time.
# Keeping them here as well protects interactive shells launched with `bash`
# from losing the project/IsaacLab import context.
export ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/IsaacLab}"
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/project}"
export OMNI_KIT_ALLOW_ROOT="${OMNI_KIT_ALLOW_ROOT:-1}"
export OMNI_KIT_DISABLE_STARTUP="${OMNI_KIT_DISABLE_STARTUP:-1}"
export UNITREE_TASKS_IMPORT_FILTER="${UNITREE_TASKS_IMPORT_FILTER:-tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3}"
export UNITREE_G1_TASKS_IMPORT_FILTER="${UNITREE_G1_TASKS_IMPORT_FILTER:-cgc_topdown_curriculum_g1_29dof_dex3}"

CACHE_ROOT="${CACHE_ROOT:-/tmp/enpm690_cache}"
CONTAINER_HOME="${CONTAINER_HOME:-${CACHE_ROOT}/home}"
CONTAINER_XDG_CACHE_HOME="${CONTAINER_XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
# Isaac Sim and matplotlib write a large amount of cache/log state on first
# launch.  Put that state under /tmp by default so running the image does not
# mutate the copied project tree or require a writable root-owned home.
mkdir -p \
    "${CONTAINER_HOME}" \
    "${CONTAINER_XDG_CACHE_HOME}" \
    "${CACHE_ROOT}/kit" \
    "${CACHE_ROOT}/omni" \
    "${CACHE_ROOT}/logs" \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/pip"

export HOME="${CONTAINER_HOME}"
export XDG_CACHE_HOME="${CONTAINER_XDG_CACHE_HOME}"
export OMNI_KIT_CACHE_DIR="${OMNI_KIT_CACHE_DIR:-${CACHE_ROOT}/kit}"
export OMNI_CACHE_ROOT="${OMNI_CACHE_ROOT:-${CACHE_ROOT}/omni}"
export OMNI_LOGS_DIR="${OMNI_LOGS_DIR:-${CACHE_ROOT}/logs}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"

cd "${PROJECT_ROOT}"

if [ "$#" -eq 0 ]; then
    exec bash
fi

exec "$@"
