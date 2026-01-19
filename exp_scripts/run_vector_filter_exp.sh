#!/bin/bash

PYTHON_SCRIPT="${1:-./vector_filter.py}"
OUTPUT_BASE_DIR="${2:./output}"
LOG_DIR="${3:-./logs}"
INV_DATASET_PATH="/mnt/vdb/data/718_1025"  # Huawei internal dataset (not for public release)

KEY_WORDS=(
    "Israelite"
    "obabilities"
    "summarized"
    "loadbalancing"
    "kubernetes"
    "loadbalancer"
    "Pharaoh's"
    "Moses"
    "getCommenterImage"
    "commenterNickname"
)

clear_cache() {
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
}

# Inverted Index Test
inv_log="${LOG_DIR}/inverted_index.log"
echo "Starting inverted index test..." > "${inv_log}"
echo "Dataset: ${INV_DATASET_PATH} (Huawei internal data)" >> "${inv_log}"

for keyword in "${KEY_WORDS[@]}"; do
    echo "Testing keyword: ${keyword}" >> "${inv_log}"
    python "${PYTHON_SCRIPT}" inv -d "${INV_DATASET_PATH}" -k "${keyword}" >> "${inv_log}" 2>&1
    clear_cache
done

echo "Inverted index test completed: ${inv_log}"

# HNSW Test
hnsw_path="${OUTPUT_BASE_DIR}/hnsw_test"
hnsw_log="${LOG_DIR}/hnsw.log"

echo "Starting HNSW test..." > "${hnsw_log}"
echo "Dataset: ${hnsw_path}" >> "${hnsw_log}"

python "${PYTHON_SCRIPT}" hnsw -d "${hnsw_path}" >> "${hnsw_log}" 2>&1
clear_cache

echo "HNSW test completed: ${hnsw_log}"