#!/bin/bash
# Copyright (c) 2026 Bingyu Liu. All rights reserved.

CUR_DIR=$(dirname $(readlink -f "$0"))
set -e

export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
THIRD_PARTY_DIR="${CUR_DIR}/../vendor/"
DEPS_CSV="${CUR_DIR}/dependencies.csv"

echo -e "Starting dependency download..."

while getopts 'T:F:' opt; do
    case "$opt" in
    T)
        THIRD_PARTY_DIR=$(readlink -f "${OPTARG}")
        ;;
    F)
        DEPS_CSV="${OPTARG}"
        ;;
    *)
        echo "Invalid command"
        exit 1
        ;;
    esac
done

if [ ! -d "${THIRD_PARTY_DIR}" ]; then
  mkdir -p "${THIRD_PARTY_DIR}"
fi

function git_clone_open_src() {
    local name="$1"
    local tag="$2"
    local repo="$3"
    local savepath="$THIRD_PARTY_DIR"

    echo -e "=== download opensrc ${name}-${tag} to ${savepath} from $repo ... ==="
    cd "$savepath"
    if [ -d "$name" ] && [ "$(ls -A "$name")" ]; then
        echo -e "${name} has already been downloaded to ${savepath} and is not empty."
        return 0
    else
        rm -rf "$name"
    fi
    mkdir "$name" && cd "$name"

    # 尝试使用tag克隆
    if git lfs clone --depth 1 -b $tag --recursive $repo . 2>/dev/null; then
        echo "Successfully cloned ${name} with tag ${tag}"
    else
        # 如果tag不存在，使用默认分支
        echo "Tag ${tag} not found, cloning default branch..."
        cd ..
        rm -rf "$name"
        mkdir "$name" && cd "$name"
        git lfs clone --depth 1 --recursive $repo .
    fi

    if [ -f ".gitmodules" ]; then
      echo "Initializing submodules for ${name}..."
      git submodule update --init --recursive --depth 1 || echo "Warning: submodule update failed for ${name}"
    fi
}

download_a_repo() {
    local name=$1
    local tag=$2
    local downloadType=$3
    local repo=$4

    echo "begin download $name"

    # 由于所有都是gitCloneOpenSrc类型，直接调用
    if [ "$downloadType" = "gitCloneOpenSrc" ]; then
        git_clone_open_src $name $tag $repo
    else
        echo "download type($downloadType) not supported"
        exit 1
    fi
}

# 检查CSV文件是否存在
if [ ! -f "${DEPS_CSV}" ]; then
    echo -e "Dependencies file not found: ${DEPS_CSV}"
    exit 1
fi

pids=()
while IFS=',' read -r name tag downloadType repo || [ -n "$name" ]; do
    # 跳过空行
    [[ -z "$name" ]] && continue

    # Start background process for each task.
    download_a_repo "$name" "$tag" "$downloadType" "$repo" &
    pid=$!
    pids+=("$pid")
    echo "Task PID ${pid}: download repo $repo"
done < "${DEPS_CSV}"

# Wait all task and handle errors
for pid in "${pids[@]}"; do
    wait "${pid}" || echo "Task with PID ${pid} failed"
done

echo "All downloads completed!"