#!/bin/bash

PARQUET_FILES=(
    "bi.parquet"
    "classic.parquet"
    "core.parquet"
    "geo.parquet"
    "log.parquet"
    "ml.parquet"
    "lineitem_duckdb_double.parquet"
)

FORMATS=(
    "deeplake"
    "muller"
    "lance"
)

PARQUET_DIR="${1:-./data/parquet}"
OUTPUT_BASE_DIR="${2:-./output}"

for parquet_file in "${PARQUET_FILES[@]}"; do
    base_name="${parquet_file%.parquet}"

    for format in "${FORMATS[@]}"; do
        input_path="${PARQUET_DIR}/${parquet_file}"
        output_path="${OUTPUT_BASE_DIR}/${format}_${base_name}"

        echo "Converting ${parquet_file} to ${format} format..."
        echo "Input: ${input_path}"
        echo "Output: ${output_path}"
        echo "----------------------------------------"

        python convert_parquet.py \
            --input "${input_path}" \
            --format "${format}" \
            --output "${output_path}"

        if [ $? -eq 0 ]; then
            echo "Successfully converted ${parquet_file} to ${format}"
        else
            echo "Error converting ${parquet_file} to ${format}"
        fi
        echo "========================================"
        echo ""
    done
done

echo "All conversions completed!"