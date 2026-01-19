#!/bin/bash

FORMATS=(
    "muller"
    "deeplake"
    "lance"
    "parquet"
)

DATASETS=(
    "core"
    "bi"
    "classic"
    "geo"
    "log"
    "ml"
    "tpch"
)

CONVERTED_DIR="${1:-./output}"
PARQUET_DIR="${2:-./data/parquet}"
LOG_DIR="${3:-./logs}"
ITERATIONS=5

mkdir -p "${LOG_DIR}"

echo "Configuration:"
echo "  Converted data directory: ${CONVERTED_DIR}"
echo "  Parquet data directory: ${PARQUET_DIR}"
echo "  Log directory: ${LOG_DIR}"
echo "  Iterations: ${ITERATIONS}"
echo ""

for format in "${FORMATS[@]}"; do
    log_file="${LOG_DIR}/random_access_${format}.log"

    echo "Running ${format} random access tests..." > "${log_file}"
    echo "========================================" >> "${log_file}"
    echo "Converted dir: ${CONVERTED_DIR}" >> "${log_file}"
    echo "Parquet dir: ${PARQUET_DIR}" >> "${log_file}"
    echo "" >> "${log_file}"

    echo "Testing format: ${format}"
    echo "========================================"

    for dataset in "${DATASETS[@]}"; do
        echo "" >> "${log_file}"
        echo "Dataset: ${dataset}" >> "${log_file}"
        echo "----------------------------------------" >> "${log_file}"

        echo "Testing dataset: ${dataset}"

        for iter in $(seq 1 ${ITERATIONS}); do
            echo "" >> "${log_file}"
            echo "Iteration ${iter}" >> "${log_file}"
            echo "Running iteration ${iter}/${ITERATIONS}..."

            python random_access.py \
                --format "${format}" \
                --dataset "${dataset}" \
                --converted_dir "${CONVERTED_DIR}" \
                --parquet_dir "${PARQUET_DIR}" \
                >> "${log_file}" 2>&1

            if [ $? -ne 0 ]; then
                echo "Error occurred during iteration ${iter}" >> "${log_file}"
                echo "Error occurred, check log file for details"
            fi

            if [ ${iter} -lt ${ITERATIONS} ]; then
                echo "Clearing cache..."
                echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
                sleep 2
            fi
        done

        echo "" >> "${log_file}"
        echo "Finished dataset ${dataset}" >> "${log_file}"
        echo "========================================" >> "${log_file}"
        echo "Finished dataset ${dataset}"
        echo ""
    done

    echo "" >> "${log_file}"
    echo "All tests completed for format ${format}" >> "${log_file}"
    echo "Format ${format} completed. Check ${log_file} for results."
    echo ""
done

echo "All random access tests completed!"
echo "Results are in ${LOG_DIR}/"