# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Modifications Copyright (c) 2026 Xueling Lin

"""Dataset import operations: create datasets from files and dataframes."""

import csv
import json
from typing import Optional


def from_csv(
        csv_path="",
        muller_path="",
        schema=None,
        path_columns=None,
        workers=0,
        scheduler="processed",
        disable_rechunk=True,
        progressbar=True,
        ignore_errors=True,
        split_tensor_meta=True
):
    """Create dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file.
        muller_path: Path where the muller dataset will be created.
        schema: Schema definition for the dataset. Dict mapping column names to
            (htype, dtype, sample_compression) tuples.
        path_columns: Dict mapping column names to handling mode for file path columns.
            - "read": Use muller.read() to load the file (for image/video tensors).
            - "text": Store the path as a plain text string.
            Columns not listed are appended directly as raw CSV values.
        workers: Number of workers for parallel processing.
        scheduler: Scheduler type for compute operations.
        disable_rechunk: Whether to disable rechunking.
        progressbar: Whether to show progress bar.
        ignore_errors: Whether to ignore errors during processing.
        split_tensor_meta: Each tensor has a tensor_meta.json if True.

    Returns:
        Dataset: The created dataset.
    """
    if not csv_path or not muller_path:
        raise ValueError("csv_path and muller_path cannot be empty.")
    org_dicts = get_data_with_dict_from_csv(csv_path)
    return _create_dataset_from_csv(org_dicts, muller_path, schema, path_columns, workers,
                                    scheduler, disable_rechunk, progressbar, ignore_errors,
                                    split_tensor_meta)


def from_file(
        ori_path="",
        muller_path="",
        schema=None,
        workers=0,
        scheduler="processed",
        disable_rechunk=True,
        progressbar=True,
        ignore_errors=True,
        split_tensor_meta=True
):
    """Create dataset from file.

    Args:
        ori_path: Path to the source file (JSON lines format).
        muller_path: Path where the muller dataset will be created.
        schema: Schema definition for the dataset.
        workers: Number of workers for parallel processing.
        scheduler: Scheduler type for compute operations.
        disable_rechunk: Whether to disable rechunking.
        progressbar: Whether to show progress bar.
        ignore_errors: Whether to ignore errors during processing.
        split_tensor_meta: Each tensor has a tensor_meta.json if True.

    Returns:
        Dataset: The created dataset.
    """
    if not ori_path or not muller_path:
        raise ValueError("ori_path and muller_path cannot be empty.")
    org_dicts = get_data_with_dict_from_file(ori_path, schema)
    return _create_dataset(org_dicts, muller_path, schema, workers, scheduler,
                          disable_rechunk, progressbar, ignore_errors, split_tensor_meta)


def from_dataframes(
        dataframes=None,
        muller_path="",
        schema=None,
        workers=0,
        scheduler="processed",
        disable_rechunk=True,
        progressbar=True,
        ignore_errors=True,
        split_tensor_meta=True
):
    """Create dataset from dataframes.

    Args:
        dataframes: List of dataframes (dicts) to import.
        muller_path: Path where the muller dataset will be created.
        schema: Schema definition for the dataset.
        workers: Number of workers for parallel processing.
        scheduler: Scheduler type for compute operations.
        disable_rechunk: Whether to disable rechunking.
        progressbar: Whether to show progress bar.
        ignore_errors: Whether to ignore errors during processing.
        split_tensor_meta: Each tensor has a tensor_meta.json if True.

    Returns:
        Dataset: The created dataset.
    """
    if dataframes is None or not muller_path:
        raise ValueError("dataframes and muller_path cannot be empty.")
    if not isinstance(dataframes, list):
        raise TypeError("Expected a list for dataframes")
    org_dicts = get_data_with_dict_from_dataframes(dataframes, schema)
    return _create_dataset(org_dicts, muller_path, schema, workers, scheduler,
                          disable_rechunk, progressbar, ignore_errors, split_tensor_meta)


def get_data_with_dict_from_file(ori_path, schema):
    """Read data from JSON lines file and convert to dict format."""
    dataframes = []
    try:
        with open(ori_path, 'r') as file:
            for line in file:
                dataframes.append(json.loads(line.strip()))
    except Exception as e:
        raise ValueError(f"Error reading JSON lines from {ori_path}") from e

    return get_data_with_dict_from_dataframes(dataframes, schema)


def get_data_with_dict_from_dataframes(dataframes, schema):
    """Convert dataframes to dict format based on schema."""
    dicts = []
    schema_has_dict_values = any(
        isinstance(value, dict) for value in schema.values()) if schema is not None else False
    for dataframe in dataframes:
        if schema_has_dict_values:
            dataframe = _extract_dataset_dict(dataframe)
        dicts.append(dataframe)
    if not dicts:
        raise ValueError("The input cannot be empty.")

    return dicts


def convert_schema(schema, parent_key=''):
    """Convert nested schema to flat schema with dot notation."""
    converted_schema = {}
    for key, value in schema.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, tuple):
            converted_schema[new_key] = value
        elif isinstance(value, dict):
            converted_schema.update(convert_schema(value, parent_key=new_key))
    return converted_schema


def _extract_dataset_dict(data):
    """Extract and flatten nested dictionary structures."""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            try:
                parsed_value = json.loads(value)
                if isinstance(parsed_value, dict):
                    nested_dict = _extract_dataset_dict(parsed_value)
                    for nested_key, nested_value in nested_dict.items():
                        result[f"{key}.{nested_key}"] = nested_value
                else:
                    result[key] = parsed_value
            except ValueError:
                result[key] = value
        elif isinstance(value, dict):
            nested_dict = _extract_dataset_dict(value)
            for nested_key, nested_value in nested_dict.items():
                result[f"{key}.{nested_key}"] = nested_value
        elif isinstance(value, list):
            result[key] = ', '.join(map(str, value))
        else:
            result[key] = value
    return result


def _create_dataset(
        org_dicts=None,
        muller_path="",
        schema=None,
        workers=0,
        scheduler="processed",
        disable_rechunk=True,
        progressbar=True,
        ignore_errors=True,
        split_tensor_meta=True
):
    """Create dataset and populate with data."""
    import muller

    ds = muller.dataset(path=muller_path, overwrite=True, split_tensor_meta=split_tensor_meta)
    if not schema:
        schema = list(org_dicts[0].keys())
        for col_name in schema:
            ds.create_tensor(col_name)
    else:
        schema = convert_schema(schema)
        for col_name, (htype, dtype, sample_compression) in schema.items():
            ds.create_tensor(col_name, htype=htype, dtype=dtype, exist_ok=True,
                            sample_compression=sample_compression)

    ds = _append_data(workers, ds, org_dicts, schema, scheduler, disable_rechunk,
                     progressbar, ignore_errors)
    return ds


def _append_data(workers, ds, org_dicts, schema, scheduler, disable_rechunk, progressbar, ignore_errors):
    """Append data to dataset using compute or direct append."""
    import muller

    @muller.compute
    def data_to_muller(data, sample_out):
        for col_name in schema:
            sample_out[col_name].append(data[col_name])
        return sample_out

    if workers in [0, 1]:
        with ds:
            for data in org_dicts:
                for col in schema:
                    ds[col].append(data[col])
    else:
        with ds:
            data_to_muller().eval(org_dicts, ds, num_workers=workers,
                                scheduler=scheduler, disable_rechunk=disable_rechunk,
                                progressbar=progressbar, ignore_errors=ignore_errors)
    return ds


def get_data_with_dict_from_csv(csv_path):
    """Read data from a CSV file and convert to list of dicts."""
    dicts = []
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dicts.append(dict(row))
    except Exception as e:
        raise ValueError(f"Error reading CSV from {csv_path}") from e

    if not dicts:
        raise ValueError("The CSV file is empty or has no data rows.")

    return dicts


def _create_dataset_from_csv(
        org_dicts=None,
        muller_path="",
        schema=None,
        path_columns=None,
        workers=0,
        scheduler="processed",
        disable_rechunk=True,
        progressbar=True,
        ignore_errors=True,
        split_tensor_meta=True
):
    """Create dataset from CSV data and populate with data."""
    import muller

    ds = muller.dataset(path=muller_path, overwrite=True, split_tensor_meta=split_tensor_meta)
    if not schema:
        schema = list(org_dicts[0].keys())
        for col_name in schema:
            ds.create_tensor(col_name)
    else:
        schema = convert_schema(schema)
        for col_name, (htype, dtype, sample_compression) in schema.items():
            ds.create_tensor(col_name, htype=htype, dtype=dtype, exist_ok=True,
                            sample_compression=sample_compression)

    ds = _append_csv_data(workers, ds, org_dicts, schema, path_columns, scheduler,
                          disable_rechunk, progressbar, ignore_errors)
    return ds


def _append_csv_data(workers, ds, org_dicts, schema, path_columns, scheduler,
                     disable_rechunk, progressbar, ignore_errors):
    """Append CSV data to dataset, handling path_columns with muller.read() or as text."""
    import muller as m

    def _process_value(col, value):
        if path_columns and col in path_columns:
            mode = path_columns[col]
            if mode == "read":
                return m.read(value)
            # "text" mode: return value as-is
        return value

    if workers in [0, 1]:
        with ds:
            for data in org_dicts:
                for col in schema:
                    ds[col].append(_process_value(col, data[col]))
    else:
        @m.compute
        def data_to_muller(data, sample_out):
            for col_name in schema:
                sample_out[col_name].append(_process_value(col_name, data[col_name]))
            return sample_out

        with ds:
            data_to_muller().eval(org_dicts, ds, num_workers=workers,
                                 scheduler=scheduler, disable_rechunk=disable_rechunk,
                                 progressbar=progressbar, ignore_errors=ignore_errors)
    return ds
