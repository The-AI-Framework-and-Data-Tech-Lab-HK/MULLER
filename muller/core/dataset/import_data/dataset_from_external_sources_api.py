# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import json
import muller


class DatasetFromExternalSource:
    @staticmethod
    def convert_schema(schema, parent_key=''):
        """Function to convert schema."""
        converted_schema = {}
        for key, value in schema.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, tuple):
                converted_schema[new_key] = value
            elif isinstance(value, dict):
                converted_schema.update(DatasetFromExternalSource.convert_schema(value, parent_key=new_key))
        return converted_schema

    @staticmethod
    def extract_dataset_dict(data):
        """Function to extract dataset dict."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                try:
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, dict):
                        nested_dict = DatasetFromExternalSource.extract_dataset_dict(parsed_value)
                        for nested_key, nested_value in nested_dict.items():
                            result[f"{key}.{nested_key}"] = nested_value
                    else:
                        result[key] = parsed_value
                except ValueError:
                    result[key] = value
            elif isinstance(value, dict):
                nested_dict = DatasetFromExternalSource.extract_dataset_dict(value)
                for nested_key, nested_value in nested_dict.items():
                    result[f"{key}.{nested_key}"] = nested_value
            elif isinstance(value, list):
                result[key] = ', '.join(map(str, value))
            else:
                result[key] = value
        return result

    @staticmethod
    def get_data_with_dict_from_file(ori_path, schema):
        """Function to get data with dict from file."""
        dataframes = []
        try:
            with open(ori_path, 'r') as file:
                for line in file:
                    dataframes.append(json.loads(line.strip()))
        except Exception as e:
            raise ValueError(f"Error reading JSON lines from {ori_path}") from e

        return DatasetFromExternalSource.get_data_with_dict_from_dataframes(dataframes, schema)

    @staticmethod
    def get_data_with_dict_from_dataframes(dataframes, schema):
        """Function to get data with dict from dataframes."""
        dicts = []
        schema_has_dict_values = any(
            isinstance(value, dict) for value in schema.values()) if schema is not None else False
        for dataframe in dataframes:
            if schema_has_dict_values:
                dataframe = DatasetFromExternalSource.extract_dataset_dict(dataframe)
            dicts.append(dataframe)
        if not dicts:
            raise ValueError("The input cannot be empty.")

        return dicts

    @staticmethod
    def create_dataset_from_file(
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
        """Function to create dataset from file."""
        if not ori_path or not muller_path:
            raise ValueError("ori_path and muller_path cannot be empty.")
        org_dicts = DatasetFromExternalSource.get_data_with_dict_from_file(ori_path, schema)
        return DatasetFromExternalSource.create_dataset(org_dicts, muller_path, schema, workers, scheduler,
                                      disable_rechunk, progressbar, ignore_errors, split_tensor_meta)

    @staticmethod
    def create_dataset_from_dataframes(
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
        """Function to create dataset from dataframes."""
        if dataframes is None or not muller_path:
            raise ValueError("dataframes and muller_path cannot be empty.")
        if not isinstance(dataframes, list):
            raise TypeError("Expected a list for dataframes")
        org_dicts = DatasetFromExternalSource.get_data_with_dict_from_dataframes(dataframes, schema)
        return DatasetFromExternalSource.create_dataset(org_dicts, muller_path, schema, workers, scheduler,
                                      disable_rechunk, progressbar, ignore_errors, split_tensor_meta)

    @staticmethod
    def create_dataset(
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
        """Function to create dataset."""
        ds = muller.dataset(path=muller_path, overwrite=True, split_tensor_meta=split_tensor_meta)
        if not schema:
            schema = list(org_dicts[0].keys())
            for col_name in schema:
                ds.create_tensor(col_name)
        else:
            schema = DatasetFromExternalSource.convert_schema(schema)
            for col_name, (htype, dtype, sample_compression) in schema.items():
                ds.create_tensor(col_name, htype=htype, dtype=dtype, exist_ok=True,
                                 sample_compression=sample_compression)

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
