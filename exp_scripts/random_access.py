#!/usr/bin/env python3
import muller
import time
import random
import sys
from pathlib import Path
import lance
import deeplake
import pyarrow.parquet as pq
import argparse

COLUMN_NAMES = {
    "core": "column16",
    "bi": "column16",
    "classic": "column16",
    "geo": "column16",
    "log": "column16",
    "ml": "column16",
    "tpch": "l_shipinstruct",
}


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000 / 10
        return result, elapsed_ms

    return wrapper


@calculate_time
def muller_random_access(path, tensor, index_list):
    ds = muller.dataset(path)
    line_list = []
    line_list.extend(ds[tensor].numpy_batch_random_access(index_list=index_list, parallel=None))
    return line_list


@calculate_time
def pq_random_access(path, col, index_list):
    table = pq.read_table(path, columns=[col])
    line_list = []
    line_list.extend(table.take(index_list))
    return line_list


@calculate_time
def lance_random_access(path, col, index_list):
    dataset = lance.dataset(path)
    line_list = []
    line_list.extend(dataset.take(index_list, columns=[col]))
    return line_list


@calculate_time
def deeplake_random_access(path, col, index_list):
    dataset = deeplake.open(path)
    line_list = []
    for idx in index_list:
        line_list.extend(dataset[idx].to_dict()[col])
    return line_list


def get_num_rows(format_name, dataset_path):
    try:
        if format_name == "muller":
            ds = muller.dataset(dataset_path)
            return len(ds)
        elif format_name == "parquet":
            table = pq.read_table(dataset_path)
            return table.num_rows
        elif format_name == "lance":
            dataset = lance.dataset(dataset_path)
            return dataset.count_rows()
        elif format_name == "deeplake":
            dataset = deeplake.open(dataset_path)
            return len(dataset)
        else:
            raise ValueError(f"Unknown format: {format_name}")
    except Exception as e:
        print(f"Error getting num_rows: {e}", file=sys.stderr)
        return None


def run_random_access(format_name, dataset_name, dataset_path):
    print(f"Loading dataset {dataset_name} ({format_name}) from {dataset_path}")

    col = COLUMN_NAMES[dataset_name]

    num_rows = get_num_rows(format_name, dataset_path)
    if num_rows is None:
        print("Failed to get num_rows, skipping...")
        return

    print(f"Dataset has {num_rows} rows")

    index_list = sorted(random.sample(range(num_rows), 10))
    print(f"row_id_list: {index_list}")

    try:
        if format_name == "muller":
            result, elapsed_ms = muller_random_access(dataset_path, col, index_list)
        elif format_name == "parquet":
            result, elapsed_ms = pq_random_access(dataset_path, col, index_list)
        elif format_name == "lance":
            result, elapsed_ms = lance_random_access(dataset_path, col, index_list)
        elif format_name == "deeplake":
            result, elapsed_ms = deeplake_random_access(dataset_path, col, index_list)
        else:
            raise ValueError(f"Unknown format: {format_name}")

        print(f"Random access time: {elapsed_ms:.3f}ms")

    except Exception as e:
        print(f"Error during random access: {e}", file=sys.stderr)
        sys.exit(1)


def get_dataset_path(format_name, dataset_name, converted_dir, parquet_dir):
    if format_name == "parquet":
        if dataset_name == "tpch":
            return f"{parquet_dir}/lineitem_duckdb_double.parquet"
        else:
            return f"{parquet_dir}/{dataset_name}.parquet"
    else:
        if dataset_name == "tpch":
            return f"{converted_dir}/{format_name}_lineitem_duckdb_double"
        else:
            return f"{converted_dir}/{format_name}_{dataset_name}"


def main():
    parser = argparse.ArgumentParser(description='Random access benchmark for different data formats')
    parser.add_argument('--format', type=str, required=True,
                        choices=['muller', 'deeplake', 'lance', 'parquet'],
                        help='Data format to test')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['core', 'bi', 'classic', 'geo', 'log', 'ml', 'tpch'],
                        help='Dataset name to test')
    parser.add_argument('--converted_dir', type=str, default='./output',
                        help='Directory containing converted datasets (muller, deeplake, lance)')
    parser.add_argument('--parquet_dir', type=str, default='./data/parquet',
                        help='Directory containing original parquet files')

    args = parser.parse_args()

    dataset_path = get_dataset_path(args.format, args.dataset,
                                    args.converted_dir, args.parquet_dir)

    if not Path(dataset_path).exists():
        print(f"Error: Dataset path {dataset_path} does not exist", file=sys.stderr)
        sys.exit(1)

    run_random_access(args.format, args.dataset, dataset_path)


if __name__ == "__main__":
    main()