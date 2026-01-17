#!/usr/bin/env python3
import gc
import muller
import time
import sys
from pathlib import Path
import lance
import deeplake
import pyarrow.parquet as pq
import subprocess
import argparse

COLUMN_NAMES = {
    "tpch": "l_shipdate",
}


def clear_cache():
    """Clear system cache and Python objects"""
    gc.collect()

    try:
        subprocess.run(['sync'], check=False)
        subprocess.run(['sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], check=False)
        print("System cache cleared")
    except Exception as e:
        print(f"Warning: Could not clear system cache: {e}")


def calculate_time(func):
    """Decorator to calculate execution time"""

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        return result, elapsed_ms

    return wrapper


@calculate_time
def muller_full_scan(path, tensor, start_idx, end_idx):
    ds = muller.dataset(path)
    result = ds[tensor][start_idx:end_idx].numpy_continuous()
    return result


@calculate_time
def pq_full_scan(path, col, start_idx, end_idx):
    table = pq.read_table(path, columns=[col])
    result = table.slice(start_idx, end_idx - start_idx)
    return result


@calculate_time
def lance_full_scan(path, col, start_idx, end_idx):
    dataset = lance.dataset(path)
    result = dataset.take(list(range(start_idx, end_idx)), columns=[col])
    return result


@calculate_time
def deeplake_full_scan(path, col, start_idx, end_idx):
    dataset = deeplake.open(path)
    result = dataset[col][start_idx:end_idx]
    return result


def get_num_rows(format_name, dataset_path):
    """Get number of rows for different formats"""
    try:
        if format_name == "muller":
            ds = muller.dataset(dataset_path)
            return len(ds)
        elif format_name == "parquet":
            table = pq.read_table(dataset_path)
            return table.num_rows
        elif format_name == "lance":
            dataset = lance.dataset(dataset_path)
            print(f"lance count rows: {dataset.count_rows()}")
            return dataset.count_rows()
        elif format_name == "dl":
            dataset = deeplake.open(dataset_path)
            return len(dataset)
        else:
            raise ValueError(f"Unknown format: {format_name}")
    except Exception as e:
        print(f"Error getting num_rows for {format_name}: {e}")
        return None


def run_full_scan(format_name, dataset_path, start_idx, end_idx):
    """Run full scan test on a dataset"""
    print(f"Loading dataset ({format_name}) from {dataset_path}")

    # Get column name
    col = COLUMN_NAMES["tpch"]

    # Get number of rows
    num_rows = get_num_rows(format_name, dataset_path)
    if num_rows is None:
        print(f"Failed to get num_rows, skipping...")
        return None

    print(f"Dataset has {num_rows} rows")
    print(f"Scanning from {start_idx} to {end_idx} ({end_idx - start_idx} rows)")

    # Clear cache before scan
    print("Clearing cache before scan...")
    clear_cache()
    time.sleep(0.5)

    # Measure full scan time based on format
    try:
        if format_name == "muller":
            result, elapsed_ms = muller_full_scan(dataset_path, col, start_idx, end_idx)
        elif format_name == "parquet":
            result, elapsed_ms = pq_full_scan(dataset_path, col, start_idx, end_idx)
        elif format_name == "lance":
            result, elapsed_ms = lance_full_scan(dataset_path, col, start_idx, end_idx)
        elif format_name == "dl":
            result, elapsed_ms = deeplake_full_scan(dataset_path, col, start_idx, end_idx)
        else:
            raise ValueError(f"Unknown format: {format_name}")

        print(f"Full scan time: {elapsed_ms:.3f}ms")

        # Clear memory after scan
        del result
        gc.collect()

        return elapsed_ms

    except Exception as e:
        error_msg = f"Error during full scan: {e}"
        print(error_msg, file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Full scan benchmark for different formats")
    parser.add_argument("-f", "--format", required=True,
                        choices=["muller", "dl", "lance", "parquet"],
                        help="Format to test")
    parser.add_argument("-p", "--path", required=True,
                        help="Path to dataset")
    parser.add_argument("-s", "--start-idx", type=int, default=0,
                        help="Start index (default: 0)")
    parser.add_argument("-e", "--end-idx", type=int, required=True,
                        help="End index")

    args = parser.parse_args()

    # Validate path exists
    if not Path(args.path).exists():
        print(f"Error: Dataset path {args.path} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.start_idx < 0 or args.end_idx <= args.start_idx:
        print(f"Error: Invalid index range [{args.start_idx}, {args.end_idx})", file=sys.stderr)
        sys.exit(1)

    # Run the scan
    elapsed_ms = run_full_scan(args.format, args.path, args.start_idx, args.end_idx)

    if elapsed_ms is None:
        sys.exit(1)

    # Output result in a parseable format
    print(f"RESULT: {args.format},{args.start_idx},{args.end_idx},{elapsed_ms:.3f}")


if __name__ == "__main__":
    main()