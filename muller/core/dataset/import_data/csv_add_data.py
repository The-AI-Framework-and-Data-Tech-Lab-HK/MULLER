# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import muller


def add_data_from_csv(
        ds,
        org_dicts=None,
        schema=None,
        path_columns=None,
        workers=0,
        scheduler="processed",
        disable_rechunk=True,
        progressbar=True,
        ignore_errors=True
):
    """Add CSV data to an existing dataset, handling path_columns with muller.read() or as text.

    Args:
        ds: The existing dataset to append data to.
        org_dicts: List of dicts parsed from CSV rows.
        schema: Schema definition (dict or None). If None, uses dict keys.
        path_columns: Dict mapping column names to handling mode:
            - "read": Use muller.read() to load the file as a Sample.
            - "text": Store the path as a plain text string.
        workers: Number of workers for parallel processing.
        scheduler: Scheduler type for compute operations.
        disable_rechunk: Whether to disable rechunking.
        progressbar: Whether to show progress bar.
        ignore_errors: Whether to ignore errors during processing.

    Returns:
        Dataset: The updated dataset.
    """
    keys = list(ds.tensors)

    if not schema:
        schema = list(org_dicts[0].keys())
    else:
        from muller.api.dataset.import_data import convert_schema
        schema = convert_schema(schema)

    if not all(col in keys for col in schema):
        raise ValueError("The column names in schema do not match the dataset keys.")

    def _process_value(col, value):
        if path_columns and col in path_columns:
            mode = path_columns[col]
            if mode == "read":
                return muller.read(value)
        # Auto-convert CSV string values to match the tensor's dtype
        tensor_dtype = str(ds.tensors[col].dtype)
        if "int" in tensor_dtype:
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return value
        elif "float" in tensor_dtype:
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        elif "bool" in tensor_dtype:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        return value

    if workers in (0, 1):
        with ds:
            for data in org_dicts:
                for col in schema:
                    ds[col].append(_process_value(col, data[col]))
    else:
        @muller.compute(batch_enable=True)
        def data_to_muller(data_batch, sample_out):
            for col_name in schema:
                col_data = [_process_value(col_name, data[col_name]) for data in data_batch]
                sample_out[col_name].append(col_data)
            return sample_out

        with ds:
            data_to_muller().eval(org_dicts, ds, num_workers=workers,
                                 scheduler=scheduler, disable_rechunk=disable_rechunk,
                                 progressbar=progressbar, ignore_errors=ignore_errors,
                                 cache_size=64)

    return ds
