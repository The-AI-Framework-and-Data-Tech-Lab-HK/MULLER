#!/bin/bash

PYTHON_SCRIPT="${1:-./full_scan.py}"
OUTPUT_BASE_DIR="${2:-./output}"
LOG_DIR="${3:-./logs}"
PARQUET_DIR="${4:-./data/parquet}"

mkdir -p "${LOG_DIR}"

FORMATS=("muller" "dl" "lance" "parquet")

END_IDX_LIST=(10000 200000 300000 400000 500000 600000 700000 800000 900000 1000000)

DATASET_NAME="tpch"

get_dataset_path() {
    local format=$1
    local base_name=$2

    if [ "$format" == "parquet" ]; then
        echo "${PARQUET_DIR}/lineitem_duckdb_double.parquet"
    else
        echo "${OUTPUT_BASE_DIR}/${format}_${base_name}"
    fi
}

echo "========================================"
echo "Full Scan Benchmark"
echo "========================================"
echo "Python Script: ${PYTHON_SCRIPT}"
echo "Output Base Dir: ${OUTPUT_BASE_DIR}"
echo "Log Dir: ${LOG_DIR}"
echo "Dataset: ${DATASET_NAME}"
echo "Formats: ${FORMATS[@]}"
echo "End Index List: ${END_IDX_LIST[@]}"
echo "========================================"
echo ""

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "Error: Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

for format in "${FORMATS[@]}"; do
    echo ""
    echo "========================================"
    echo "Testing format: ${format}"
    echo "========================================"

    dataset_path=$(get_dataset_path "${format}" "${DATASET_NAME}")

    echo "Dataset path: ${dataset_path}"

    if [ ! -e "${dataset_path}" ]; then
        echo "Warning: Dataset path does not exist: ${dataset_path}"
        echo "Skipping ${format}..."
        continue
    fi

    log_file="${LOG_DIR}/full_scan_${format}_${DATASET_NAME}.log"

    {
        echo "========================================"
        echo "Full Scan Benchmark - ${format}"
        echo "========================================"
        echo "Dataset: ${DATASET_NAME}"
        echo "Path: ${dataset_path}"
        echo "Start Time: $(date)"
        echo "========================================"
        echo ""
    } > "${log_file}"

    for end_idx in "${END_IDX_LIST[@]}"; do
        start_idx=0
        rows=$((end_idx - start_idx))

        echo ""
        echo "----------------------------------------"
        echo "Testing: start_idx=${start_idx}, end_idx=${end_idx}, rows=${rows}"
        echo "----------------------------------------"

        {
            echo "----------------------------------------"
            echo "Test: start_idx=${start_idx}, end_idx=${end_idx}, rows=${rows}"
            echo "Time: $(date)"
            echo "----------------------------------------"
        } >> "${log_file}"

        if python "${PYTHON_SCRIPT}" \
            --format "${format}" \
            --path "${dataset_path}" \
            --start-idx ${start_idx} \
            --end-idx ${end_idx} 2>&1 | tee -a "${log_file}"; then

            echo "✓ Test completed successfully"
            echo "" >> "${log_file}"
        else
            echo "✗ Test failed"
            {
                echo "ERROR: Test failed"
                echo ""
            } >> "${log_file}"
        fi

        echo "Clearing memory and cache..."
        sync
        sleep 1

        echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

        sleep 2

        echo ""
    done

    {
        echo "========================================"
        echo "Format ${format} completed"
        echo "End Time: $(date)"
        echo "========================================"
    } >> "${log_file}"

    echo ""
    echo "Format ${format} completed. Log saved to: ${log_file}"
done

echo ""
echo "========================================"
echo "All tests completed!"
echo "========================================"
echo "Logs are saved in: ${LOG_DIR}"
echo ""

summary_file="${LOG_DIR}/summary.log"
echo "Generating summary report: ${summary_file}"

{
    echo "========================================"
    echo "Full Scan Benchmark - Summary"
    echo "========================================"
    echo "Generated at: $(date)"
    echo ""

    for format in "${FORMATS[@]}"; do
        log_file="${LOG_DIR}/full_scan_${format}_${DATASET_NAME}.log"

        if [ -f "${log_file}" ]; then
            echo "Format: ${format}"
            echo "----------------------------------------"

            grep "RESULT:" "${log_file}" | while read -r line; do
                IFS=',' read -r fmt start end time <<< "${line#RESULT: }"
                rows=$((end - start))
                echo "  Rows: ${rows}, Time: ${time}ms"
            done

            echo ""
        fi
    done

    echo "========================================"
} > "${summary_file}"

echo "Summary report generated!"
cat "${summary_file}"