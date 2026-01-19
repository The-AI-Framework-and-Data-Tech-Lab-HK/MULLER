# MULLER Experiment Reproduction Guide

Scripts to reproduce the experimental results from the MULLER paper, comparing various data file formats across multiple performance dimensions.

## Directory Structure
```
.
├── data/  
│   └── parquet/                    # Source parquet datasets  
│       ├── bi.parquet  
│       ├── classic.parquet  
│       ├── core.parquet  
│       ├── geo.parquet  
│       ├── log.parquet  
│       ├── ml.parquet  
│       └── lineitem_duckdb_double.parquet  
│
├── output/                         # Generated test datasets
│   ├── deeplake_*/                 # Deeplake format datasets
│   ├── muller_*/                   # Muller format datasets
│   ├── lance_*/                    # Lance format datasets
│   ├── hnsw_test/                  # HNSW index for vector search
│   ├── benchmark_vc/               # Version control test dataset
│   ├── 8w_rows_parquet/           # Projection test datasets
│   ├── 8w_rows_deeplake/
│   └── 8w_rows_muller/
│
├── logs/                           # Experiment results
│   ├── random_access_*.log
│   ├── vector_filter_*.log
│   ├── full_scan_*.log
│   ├── projection_*.log
│   └── version_control_*.log
.
```
## Prerequisites

- MULLER
- Rust (for F3/FFF format, see [F3 build instructions](https://github.com/future-file-format/F3?tab=readme-ov-file#build-instructions))
- Required Python packages:
  ```bash
  pip install pyarrow pandas deeplake pylance faiss-cpu

## Reproduction Steps

### Step 0: Prepare Source Datasets
Clone and build the F3 repository to generate the source parquet files.
```
git clone https://github.com/future-file-format/F3.git
cd F3
cargo build --example bench --release
```
Replace `F3/exp_scripts/random_access.sh` with the `data_generate.sh` script to generate parquet datasets and corresponding FFF format data.
```
./exp_scripts/data_generate.sh
```
Copy the generated parquet files to ./data/parquet/ in this repository.


### Step 1: Convert Datasets to Different Formats
Convert parquet files to Deeplake, Muller, and Lance formats:
```
./exp_scripts/convert_all.sh
```
This script processes all seven parquet files and generates datasets in the three target formats under `./output/`.

### Step 2: Run Experiments
#### Experiment 1: Random Access (Figure 3a)
Test random access performance across Muller, Deeplake, Lance, and Parquet formats:
```
./exp_scripts/run_random_access_exp.sh
```
For F3/FFF format results, run from the F3 repository:
```
./exp_scripts/random_access.sh
```

#### Experiment 2: Hybrid Search (Figure 3b)
```
./exp_scripts/run_vector_filter_exp.sh
```

#### Experiment 3: Full Scan (Figure 3c)
```
./exp_scripts/run_full_scan_exp.sh
```

#### Experiment 4: Projection (Figure 3d)
Test column projection performance for Parquet, Deeplake, and Muller:
```
./exp_scripts/run_projection_exp.sh all
```
For F3/FFF format results, run from the F3 repository:
```
cargo run --example metadata_test_v2 --release gen
./exp_scripts/projection.sh
```
```
./exp_scripts/projection.sh
```
#### Experiment 5: Version Control (Table 2)
Test version control operations:
```
python version_control.py
```

### Result Analysis
All experiment results are stored in the ./logs/ directory. Each log file contains:

- Execution times for each operation
- File I/O statistics
- Detailed performance metrics

You can analyze the results using the provided log files to reproduce the figures and tables from the paper.