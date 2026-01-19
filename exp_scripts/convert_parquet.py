import argparse
import pandas as pd
import pyarrow.parquet as pq
import muller
import deeplake
import lance
import pyarrow as pa
from typing import Iterator


def convert_parquet_to_deeplake(parquet_path, output_path, batch_size=100000):
    print(f"Reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"Data shape: {df.shape}")
    print(f"Creating deeplake dataset: {output_path}")
    ds = deeplake.create(output_path)

    for i, column_name in enumerate(df.columns):
        print(f"Adding column {i + 1}/{len(df.columns)}: {column_name}")

        if df[column_name].dtype == 'string' or df[column_name].dtype == 'object':
            ds.add_column(column_name, deeplake.types.Text())
        elif df[column_name].dtype == 'float64' or df[column_name].dtype == 'double':
            ds.add_column(column_name, deeplake.types.Float64())
        elif df[column_name].dtype == 'int64':
            ds.add_column(column_name, deeplake.types.Int64())
        else:
            ds.add_column(column_name, deeplake.types.Float64())

    num_rows = len(df)
    total_batches = (num_rows + batch_size - 1) // batch_size
    print(f"Total batches: {total_batches}")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_rows)

        print(f"Processing batch {batch_idx + 1}/{total_batches}, rows {start_idx} to {end_idx}")

        data_dict = {}
        for column_name in df.columns:
            batch_data = df[column_name].iloc[start_idx:end_idx].tolist()
            data_dict[column_name] = batch_data

        ds.append(data_dict)

    print(f"Conversion complete! Data written to: {output_path}")
    ds.commit()
    ds.summary()
    return ds


def convert_parquet_to_muller(parquet_path, output_path):
    print(f"Reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"Data shape: {df.shape}")
    print(f"Creating muller dataset: {output_path}")
    ds = muller.dataset(output_path, overwrite=True)

    for i, column_name in enumerate(df.columns):
        print(f"Processing column {i + 1}/{len(df.columns)}: {column_name}")

        column_data = df[column_name].values

        if df[column_name].dtype == 'string' or df[column_name].dtype == 'object':
            dtype = 'str'
            htype = 'text'
        elif df[column_name].dtype == 'float64' or df[column_name].dtype == 'double':
            dtype = 'float64'
            htype = 'generic'
        else:
            dtype = df[column_name].dtype
            htype = 'generic'

        ds.create_tensor(column_name, htype=htype, dtype=dtype)
        ds[column_name].extend(column_data.tolist())

        print(f"Column {column_name}: type={df[column_name].dtype}, data size={len(column_data)}")

    print(f"Conversion complete! Data written to: {output_path}")
    ds.summary()
    return ds


def convert_parquet_to_lance(parquet_path, output_path, batch_size=100000, use_streaming=True):
    print(f"Reading parquet file: {parquet_path}")

    if use_streaming:
        print("Using streaming mode for memory efficiency")

        parquet_file = pq.ParquetFile(parquet_path)
        schema = parquet_file.schema_arrow
        total_rows = parquet_file.metadata.num_rows

        print(f"Total rows: {total_rows}")
        print(f"Schema: {schema}")

        def batch_producer() -> Iterator[pa.RecordBatch]:
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                yield batch

        print(f"Creating lance dataset: {output_path}")
        ds = lance.write_dataset(
            batch_producer(),
            output_path,
            schema=schema,
            mode="overwrite"
        )
    else:
        print("Loading entire parquet file into memory")
        table = pq.read_table(parquet_path)

        print(f"Data shape: {table.num_rows} rows, {table.num_columns} columns")
        print(f"Schema: {table.schema}")

        print(f"Creating lance dataset: {output_path}")
        ds = lance.write_dataset(
            table,
            output_path,
            mode="overwrite"
        )

    print(f"Conversion complete! Data written to: {output_path}")
    print(f"Total rows in Lance dataset: {ds.count_rows()}")

    return ds


def main():
    parser = argparse.ArgumentParser(description='Convert parquet files to different formats')
    parser.add_argument('--input', type=str, required=True, help='Input parquet file path')
    parser.add_argument('--format', type=str, required=True,
                        choices=['deeplake', 'muller', 'lance'],
                        help='Output format: deeplake, muller, or lance')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--batch_size', type=int, default=100000,
                        help='Batch size for processing (default: 100000)')

    args = parser.parse_args()

    if args.format == 'deeplake':
        convert_parquet_to_deeplake(args.input, args.output, args.batch_size)
    elif args.format == 'muller':
        convert_parquet_to_muller(args.input, args.output)
    elif args.format == 'lance':
        convert_parquet_to_lance(args.input, args.output, args.batch_size, use_streaming=True)


if __name__ == "__main__":
    main()