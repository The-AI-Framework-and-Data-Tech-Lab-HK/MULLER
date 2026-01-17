#!/usr/bin/env python3
import argparse
import os
import random
import time
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def column_name(i: int) -> str:
    return f"column_{i}"


def generate_batch(num_columns: int, num_rows: int) -> pa.RecordBatch:
    fields = []
    for i in range(num_columns):
        fields.append(pa.field(column_name(i), pa.float64(), nullable=False))

    schema = pa.schema(fields)

    array = pa.array([42.0] * num_rows, type=pa.float64())
    columns = [array] * num_columns

    return pa.RecordBatch.from_arrays(columns, schema=schema)


def write_parquet(batches: List[pa.RecordBatch], parquet_path: str) -> None:
    table = pa.Table.from_batches(batches)
    pq.write_table(table, parquet_path)


def parquet_decompress_from(
        parquet_file,
        projections: Optional[List[int]] = None,
) -> int:
    columns = None
    if projections is not None:
        columns = [column_name(i) for i in projections]

    table = parquet_file.read(columns=columns)

    nbytes = table.nbytes

    return nbytes


def generate_datasets(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)

    column_counts = [10, 100, 500, 1000, 5000, 10000, 20000]
    num_rows = 80000

    for num_columns in column_counts:
        parquet_path = os.path.join(data_dir, f"{num_columns}.parquet")

        print(f"Generating Parquet file with {num_columns} columns...")
        batch = generate_batch(num_columns, num_rows)

        write_parquet([batch], parquet_path)

        print(f"✓ Generated Parquet file with {num_columns} columns at {parquet_path}")


def test_parquet_read(data_dir: str, num_columns: int) -> None:
    num_rows = 80000
    parquet_path = os.path.join(data_dir, f"{num_columns}.parquet")

    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found: {parquet_path}")
        return

    if num_columns == 1000:
        projections = [156, 183, 374, 445, 596, 598, 731, 779, 796, 950]
    else:
        random.seed(42)
        projections = sorted(random.sample(range(num_columns), min(10, num_columns)))

    parquet_file = pq.ParquetFile(parquet_path)

    start_time = time.perf_counter()
    columns = None
    if projections is not None:
        columns = [column_name(i) for i in projections]

    parquet_file.read(columns=columns)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"Parquet num_rows: {num_rows}, num_cols: {num_columns}, "
          f"projections: {len(projections)}, time: {elapsed_ms:.3f} ms")


def write_deeplake(num_columns: int, num_rows: int, deeplake_path: str) -> None:
    import deeplake

    print(f"Creating Deeplake dataset with {num_columns} columns...")

    if os.path.exists(deeplake_path):
        import shutil
        shutil.rmtree(deeplake_path)

    ds = deeplake.create(deeplake_path)
    print(f"  Dataset created successfully")

    for i in range(num_columns):
        col_name = column_name(i)
        ds.add_column(col_name, deeplake.types.Float64())

    row_batch_size = 1
    total_batches = (num_rows + row_batch_size - 1) // row_batch_size
    print(f"  Writing {total_batches} batches of data...")

    for batch_idx in range(total_batches):
        current_batch_size = min(row_batch_size, num_rows - batch_idx * row_batch_size)

        data_dict = {}
        for i in range(num_columns):
            col = column_name(i)
            data_dict[col] = [42.0] * current_batch_size
        ds.append(data_dict)

    ds.commit()
    print(f"  Dataset committed")


def write_muller(num_columns: int, num_rows: int, muller_path: str) -> None:
    import muller

    print(f"Creating MULLER dataset with {num_columns} columns...")

    ds = muller.dataset(muller_path, overwrite=True)

    for i in range(num_columns):
        ds.create_tensor(column_name(i), htype='generic', dtype='float64')
        getattr(ds, column_name(i)).extend([42.0] * num_rows)

        if (i + 1) % 100 == 0 or i == num_columns - 1:
            print(f"  Created and populated {i + 1}/{num_columns} tensors")

    ds.commit()
    print(f"  Dataset committed")


def generate_deeplake_datasets(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)

    column_counts = [10, 100, 500, 1000, 5000, 10000, 20000]
    num_rows = 80000

    for num_columns in column_counts:
        deeplake_path = os.path.join(data_dir, f"{num_columns}")
        write_deeplake(num_columns, num_rows, deeplake_path)

        print(f"✓ Generated Deeplake dataset with {num_columns} columns at {deeplake_path}\n")


def generate_muller_datasets(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)

    column_counts = [10, 100, 500, 1000, 5000, 10000, 20000]
    num_rows = 80000

    for num_columns in column_counts:
        muller_path = os.path.join(data_dir, f"{num_columns}")
        write_muller(num_columns, num_rows, muller_path)

        print(f"✓ Generated MULLER dataset with {num_columns} columns at {muller_path}\n")


def test_deeplake_read(data_dir: str, num_columns: int) -> None:
    import deeplake

    num_rows = 80000
    deeplake_path = os.path.join(data_dir, f"{num_columns}")

    if not os.path.exists(deeplake_path):
        print(f"Error: Deeplake dataset not found: {deeplake_path}")
        return

    if num_columns == 1000:
        projections = [156, 183, 374, 445, 596, 598, 731, 779, 796, 950]
    else:
        random.seed(42)
        projections = sorted(random.sample(range(num_columns), min(10, num_columns)))

    ds = deeplake.open(deeplake_path)

    start_time = time.perf_counter()

    for proj_idx in projections:
        col_name = column_name(proj_idx)
        _ = ds.schema
        _ = len(ds[col_name])

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"Deeplake num_rows: {num_rows}, num_cols: {num_columns}, "
          f"projections: {len(projections)}, time: {elapsed_ms:.3f} ms")


def test_muller_read(data_dir: str, num_columns: int) -> None:
    import muller

    num_rows = 80000
    muller_path = os.path.join(data_dir, f"{num_columns}")

    if not os.path.exists(muller_path):
        print(f"Error: MULLER dataset not found: {muller_path}")
        return

    if num_columns == 1000:
        projections = [156, 183, 374, 445, 596, 598, 731, 779, 796, 950]
    else:
        random.seed(42)
        projections = sorted(random.sample(range(num_columns), min(10, num_columns)))

    start_time = time.perf_counter()

    for proj_idx in projections:
        col_name = column_name(proj_idx)
        res = muller.get_col_info(muller_path, col_name)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"MULLER num_rows: {num_rows}, num_cols: {num_columns}, "
          f"projections: {len(projections)}, time: {elapsed_ms:.3f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Projection benchmark tool for Parquet, Deeplake, and MULLER formats"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("gen", help="Generate datasets for all formats")
    gen_parser.add_argument(
        "--parquet-dir",
        default="8w_rows_parquet",
        help="Directory for Parquet datasets (default: 8w_rows_parquet)"
    )
    gen_parser.add_argument(
        "--deeplake-dir",
        default="8w_rows_deeplake",
        help="Directory for Deeplake datasets (default: 8w_rows_deeplake)"
    )
    gen_parser.add_argument(
        "--muller-dir",
        default="8w_rows_muller",
        help="Directory for MULLER datasets (default: 8w_rows_muller)"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test read performance")
    test_parser.add_argument(
        "-f", "--format",
        required=True,
        choices=["parquet", "deeplake", "muller"],
        help="Format to test"
    )
    test_parser.add_argument(
        "-n", "--num-columns",
        type=int,
        required=True,
        choices=[10, 100, 500, 1000, 5000, 10000, 20000],
        help="Number of columns to test"
    )
    test_parser.add_argument(
        "--data-dir",
        help="Data directory (default: auto-detected based on format)"
    )

    args = parser.parse_args()

    if args.command == "gen":
        print("=" * 60)
        print("Generating datasets for all formats")
        print("=" * 60)
        print()

        print("Step 1/3: Generating Parquet datasets")
        print("-" * 60)
        generate_datasets(args.parquet_dir)
        print()

        print("Step 2/3: Generating Deeplake datasets")
        print("-" * 60)
        generate_deeplake_datasets(args.deeplake_dir)
        print()

        print("Step 3/3: Generating MULLER datasets")
        print("-" * 60)
        generate_muller_datasets(args.muller_dir)
        print()

        print("=" * 60)
        print("All datasets generated successfully!")
        print("=" * 60)

    elif args.command == "test":
        if args.data_dir:
            data_dir = args.data_dir
        else:
            if args.format == "parquet":
                data_dir = "./output/8w_rows_parquet"
            elif args.format == "deeplake":
                data_dir = "./output/8w_rows_deeplake"
            elif args.format == "muller":
                data_dir = "./output/8w_rows_muller"

        if args.format == "parquet":
            test_parquet_read(data_dir, args.num_columns)
        elif args.format == "deeplake":
            test_deeplake_read(data_dir, args.num_columns)
        elif args.format == "muller":
            test_muller_read(data_dir, args.num_columns)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()