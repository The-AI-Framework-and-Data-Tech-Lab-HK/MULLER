#!/bin/bash

PYTHON_SCRIPT="projection.py"
LOG_DIR="./logs/projection"
FORMATS=("parquet" "deeplake" "muller")
COLUMN_COUNTS=(10 100 500 1000 5000 10000 20000)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

mkdir -p "${LOG_DIR}"

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

generate_datasets() {
    print_info "Starting dataset generation..."

    log_file="${LOG_DIR}/generation.log"

    echo "Dataset Generation Log" > "${log_file}"
    echo "======================" >> "${log_file}"
    echo "Start time: $(date)" >> "${log_file}"
    echo "" >> "${log_file}"

    python "${PYTHON_SCRIPT}" gen >> "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        print_info "Dataset generation completed successfully"
        echo "" >> "${log_file}"
        echo "Generation completed successfully" >> "${log_file}"
        echo "End time: $(date)" >> "${log_file}"
    else
        print_error "Dataset generation failed. Check ${log_file} for details"
        exit 1
    fi

    echo ""
}

test_format() {
    local format=$1
    local log_file="${LOG_DIR}/test_${format}.log"

    print_info "Testing format: ${format}"
    echo "=========================================="

    echo "Projection Test Log - ${format}" > "${log_file}"
    echo "======================================" >> "${log_file}"
    echo "Start time: $(date)" >> "${log_file}"
    echo "Format: ${format}" >> "${log_file}"
    echo "" >> "${log_file}"

    for num_cols in "${COLUMN_COUNTS[@]}"; do
        print_info "Testing ${format} with ${num_cols} columns..."

        echo "" >> "${log_file}"
        echo "Testing with ${num_cols} columns" >> "${log_file}"
        echo "----------------------------------------" >> "${log_file}"

        python "${PYTHON_SCRIPT}" test -f "${format}" -n "${num_cols}" >> "${log_file}" 2>&1

        if [ $? -eq 0 ]; then
            print_info "  ✓ ${num_cols} columns test completed"
        else
            print_error "  ✗ ${num_cols} columns test failed"
            echo "Error occurred during ${num_cols} columns test" >> "${log_file}"
        fi

        if [ ${num_cols} -ne ${COLUMN_COUNTS[-1]} ]; then
            print_info "  Clearing cache..."
            echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1
            sleep 2
        fi
    done

    echo "" >> "${log_file}"
    echo "All tests completed for ${format}" >> "${log_file}"
    echo "End time: $(date)" >> "${log_file}"

    print_info "Format ${format} completed. Results in ${log_file}"
    echo ""
}

test_all_formats() {
    print_info "Starting projection tests for all formats..."
    echo ""

    for format in "${FORMATS[@]}"; do
        test_format "${format}"
    done

    print_info "All projection tests completed!"
    print_info "Results are in ${LOG_DIR}/"
}

generate_summary() {
    local summary_file="${LOG_DIR}/summary.log"

    print_info "Generating summary report..."

    echo "Projection Test Summary" > "${summary_file}"
    echo "======================" >> "${summary_file}"
    echo "Generated at: $(date)" >> "${summary_file}"
    echo "" >> "${summary_file}"

    for format in "${FORMATS[@]}"; do
        local log_file="${LOG_DIR}/test_${format}.log"

        if [ -f "${log_file}" ]; then
            echo "Format: ${format}" >> "${summary_file}"
            echo "----------------------------------------" >> "${summary_file}"

            grep "time:" "${log_file}" >> "${summary_file}" 2>/dev/null

            echo "" >> "${summary_file}"
        fi
    done

    print_info "Summary report generated: ${summary_file}"
}

show_help() {
    cat << EOF
Usage: $0 [COMMAND]

Commands:
    gen     Generate datasets for parquet, deeplake, and MULLER formats
    test    Run projection tests for all formats
    all     Generate datasets and run all tests
    help    Show this help message

Examples:
    $0 gen          # Only generate datasets
    $0 test         # Only run tests (datasets must exist)
    $0 all          # Generate datasets and run tests

Logs will be saved to: ${LOG_DIR}/
EOF
}

main() {
    local command=${1:-help}

    case "${command}" in
        gen)
            generate_datasets
            ;;
        test)
            test_all_formats
            generate_summary
            ;;
        all)
            generate_datasets
            test_all_formats
            generate_summary
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: ${command}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    print_error "Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

main "$@"