#!/bin/bash

# Build the benchmark
cargo build --example bench --release

# Generate datasets
for dataset in tpch core bi classic geo log ml; do
    echo "Generating dataset: $dataset"
    sudo sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    ./target/release/examples/bench random-access -- $dataset
    echo -e "\n"
done

echo "Dataset generation completed."