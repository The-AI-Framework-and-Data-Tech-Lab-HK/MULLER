# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/api/dataset.py
#
# Modifications Copyright (c) 2026 Xueling Lin

import json
import logging
import os
import pathlib
import posixpath
from typing import Dict, Optional, Union, List

import muller
from muller.constants import DEFAULT_LOCAL_CACHE_SIZE, DEFAULT_MEMORY_CACHE_SIZE, TENSOR_META_FILENAME
from muller.core.dataset.dataset import Dataset
from muller.util.exceptions import (AgreementError,
                                    CheckoutError,
                                    LockedException,
                                    ReadOnlyModeError,
                                    DatasetCreationError,
                                    DatasetCorruptionError,
                                    DatasetAlreadyExistsError,
                                    DatasetNotExistsError,
                                    DatasetViewDeletionError, DirectoryAtPathException)
from muller.util.keys import dataset_exists
from muller.util.path import process_dataset_path
from muller.util.path import verify_dataset_name, convert_pathlib_to_string_if_needed
from muller.util.permission.invalid_user_op import validate_permissions
from muller.util.storage import get_storage_and_cache_chain
from muller.util.version_control import (
    load_version_info,
    get_parent_and_reset_commit_ids,
    replace_head,
    integrity_check,
)


class DatasetAPI:
    @staticmethod
    def dataset(
            path: Union[str, pathlib.Path],
            read_only: Optional[bool] = None,
            overwrite: bool = False,
            memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
            local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
            creds: Optional[Union[Dict, str]] = None,
            verbose: bool = True,
            reset: bool = False,  # Version Control related
            check_integrity: bool = True,
            lock_enabled: Optional[bool] = True,
            lock_timeout: Optional[int] = 0,
            split_tensor_meta: bool = True
    ):
        """
        Returns a :class:`~muller.core.dataset.Dataset` object referencing either a new or existing dataset.
        Examples:

            >>> ds = muller.dataset("./datasets/my_dataset", overwrite=True)
            >>> ds = muller.dataset("roma://mybucket/my_dataset")

        Args:
            path (str, pathlib.Path): - The full path to the dataset. Can be:
              - an s3 path of the form ``s3://bucketname/path/to/dataset``.
                Credentials are required in either the environment or passed to the creds argument.
              - a local file system path of the form ``./path/to/dataset``,``~/path/to/dataset`` or ``path/to/dataset``.
            read_only (bool, optional): Opens dataset in read only mode if this is passed as ``True``.
                Defaults to ``False``.
            overwrite (bool): If set to ``True`` this overwrites the dataset if it already exists.
                Defaults to ``False``.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            creds (dict, str, optional): credentials for OBS service.
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.
            reset (bool): If the specified dataset cannot be loaded due to a corrupted HEAD state of the branch
                being loaded, setting ``reset=True`` will reset HEAD changes and load the previous version.
            check_integrity (bool, Optional): Performs an integrity check by default (None) if the dataset has 20
                or fewer tensors. Set to ``True`` to force integrity check, ``False`` to skip integrity check.
            lock_timeout (int): Number of seconds to wait before throwing a LockException. If None, wait indefinitely.
            lock_enabled (bool): If true, the dataset manages a write lock.
                NOTE: Only set to False if you are managing concurrent access externally
            split_tensor_meta (bool): Each tensor has a tensor_meta.json if True, all tensors share one tensor_meta.json
                if False. Default to True.

        """

        processed_path, address, _, cache_chain = DatasetAPI._valiadation(path=path,
                                                                       path_verify=True,
                                                                       read_only=read_only,
                                                                       creds=creds,
                                                                       memory_cache_size=memory_cache_size,
                                                                       local_cache_size=local_cache_size)

        ds_exists = dataset_exists(cache_chain)

        if ds_exists and not overwrite:
            create = False
        else:
            create = True

        if overwrite:
            cache_chain.clear()
            create = True

        if create and address:
            raise DatasetCreationError(path)

        dataset_kwargs: Dict[str, Union[None, str, bool, int, Dict]] = {
            "path": processed_path,
            "read_only": read_only,
            "verbose": verbose,
            "lock_timeout": lock_timeout,
            "lock_enabled": lock_enabled,
            "split_tensor_meta": split_tensor_meta,
            "storage": cache_chain,
            "address": address,
            "creds": creds
        }

        try:
            return DatasetAPI._load(dataset_kwargs, check_integrity=check_integrity)
        except (AgreementError, CheckoutError, LockedException) as e:
            raise e from None
        except Exception as e:
            if create:
                raise e
            if not reset:
                raise DatasetCorruptionError from e
            return DatasetAPI._reset_and_load(cache_chain, dataset_kwargs, address, e)

    @validate_permissions
    @staticmethod
    def load(
            path: Union[str, pathlib.Path],
            read_only: Optional[bool] = None,
            memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
            local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
            creds: Optional[Union[Dict, str]] = None,
            verbose: bool = True,
            check_integrity: bool = True,
            lock_enabled: Optional[bool] = True,
            lock_timeout: Optional[int] = 0,
            split_tensor_meta: bool = True,
    ) -> Dataset:
        """
        Load dataset from  given path.
        :param path: The full path to the dataset.
        :param read_only: Opens dataset in read only mode if this is passed as ``True``. Defaults to ``False``.
        :param memory_cache_size: The size of the memory cache to be used in MB.
        :param local_cache_size: The size of the local filesystem cache to be used in MB.
        :param creds: credentials for OBS service.
        :param verbose: If ``True``, logs will be printed. Defaults to ``True``.
        :param check_integrity: Performs an integrity check by default (None) if the dataset has 20 or fewer tensors.
        Set to ``True`` to force integrity check, ``False`` to skip integrity check.
        :param lock_enabled: If true, the dataset manages a write lock.
        NOTE: Only set to False if you are managing concurrent access externally
        :param lock_timeout: Number of seconds to wait before throwing a LockException. If None, wait indefinitely.
        :param split_tensor_meta: Each tensor has a tensor_meta.json if True, all tensors share one tensor_meta.json
        if False. Default to True.
        :return: Dataset
        """

        processed_path, address, _, cache_chain = DatasetAPI._valiadation(path=path,
                                                                       path_verify=False,
                                                                       read_only=read_only,
                                                                       creds=creds,
                                                                       memory_cache_size=memory_cache_size,
                                                                       local_cache_size=local_cache_size)

        if not dataset_exists(cache_chain):
            raise DatasetNotExistsError(processed_path)

        dataset_kwargs: Dict[str, Union[None, str, bool, int, Dict]] = {
            "path": processed_path,
            "read_only": read_only,
            "verbose": verbose,
            "lock_timeout": lock_timeout,
            "lock_enabled": lock_enabled,
            "split_tensor_meta": split_tensor_meta,
            "storage": cache_chain,
            "address": address,
            "creds": creds
        }

        try:
            return DatasetAPI._load(
                dataset_kwargs, check_integrity=check_integrity
            )
        except (AgreementError, CheckoutError, LockedException) as e:
            raise e from None
        except Exception as e:
            raise e

    @staticmethod
    def get_col_info(path: Union[str, pathlib.Path],
                     col_name: str = None):
        processed_path, _ = process_dataset_path(path)
        meta_file = TENSOR_META_FILENAME if col_name == "" else "/".join((col_name, TENSOR_META_FILENAME))
        try:
            fpath = DatasetAPI._check_is_file(processed_path, meta_file)
            with open(fpath, "rb") as file:
                return file.read()
        except DirectoryAtPathException:
            raise
        except FileNotFoundError as e:
            raise KeyError(meta_file) from e

    @staticmethod
    def empty(
            path: Union[str, pathlib.Path],
            overwrite: bool = False,
            memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
            local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
            creds: Optional[Union[Dict, str]] = None,
            lock_enabled: Optional[bool] = True,
            lock_timeout: Optional[int] = 0,
            verbose: bool = True,
            split_tensor_meta: bool = True,
    ) -> Dataset:
        """Creates an empty dataset

        Args:
            path (str, pathlib.Path): - The full path to the dataset. It can be:
                - an roma path of the form ``roma://bucketname/path/to/dataset``.
                  Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or
                  ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in
                  memory instead. Should be used only for testing as it does not persist.
            overwrite (bool): If set to ``True`` we overwrite the dataset if it already exists. Defaults to ``False``.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access
                the dataset at the path.
            verbose (bool): If True, logs will be printed. Defaults to True.
            lock_timeout (int): Number of seconds to wait before throwing a LockException. If None, wait indefinitely
            lock_enabled (bool): If true, the dataset manages a write lock. NOTE: Only set to False if you are managing
                concurrent access externally.
            index_params: Optional[Dict[str, Union[int, str]]]: Index parameters used while creating vector store,
                passed down to dataset.

        Returns:
            Dataset: Dataset created using the arguments provided.

        Raises:
            DatasetHandlerError: If a Dataset already exists at the given path and overwrite is False.
            UserNotLoggedInException: When user is not logged in
            ValueError: If version is specified in the path

        Danger:
            Setting ``overwrite`` to ``True`` will delete all of your data if it exists! Be very careful when setting!
        """

        processed_path, address, storage, cache_chain = DatasetAPI._valiadation(path=path,
                                                                       path_verify=True,
                                                                       read_only=False,
                                                                       creds=creds,
                                                                       memory_cache_size=memory_cache_size,
                                                                       local_cache_size=local_cache_size)
        if address:
            raise DatasetCreationError(path)
        if overwrite and dataset_exists(cache_chain):
            cache_chain.clear()
        elif dataset_exists(cache_chain):
            raise DatasetAlreadyExistsError(processed_path)

        dataset_kwargs = {
            "path": processed_path,
            "read_only": storage.read_only,
            "verbose": verbose,
            "lock_timeout": lock_timeout,
            "lock_enabled": lock_enabled,
            "split_tensor_meta": split_tensor_meta,
            "storage": cache_chain,
            "creds": creds,
        }
        ret = DatasetAPI._load(dataset_kwargs)
        return ret

    @staticmethod
    def delete(
            path: Union[str, pathlib.Path],
            large_ok: bool = False,
            creds: Optional[Union[dict, str]] = None,
    ) -> None:
        """
        Delete a dataset at the given path
        """

        qtokens = ["/.queries/", "\\.queries\\"]
        for qt in qtokens:
            if qt in path:
                raise DatasetViewDeletionError
        try:
            ds = DatasetAPI.load(path, verbose=False, creds=creds)
        except Exception as e:
            raise e
        ds.delete(large_ok)

    @staticmethod
    def like(
            dest: Union[str, Dataset],
            src: Union[str, Dataset],
            tensors: Optional[List[str]] = None,
            overwrite: bool = False,
            verbose: bool = True,
    ) -> Dataset:
        """Copies the `source` dataset's structure to a new location.
        No samples are copied, only the meta/info for the dataset and it's tensors.

        Args:
            dest(Union[str, Dataset]): Empty Dataset or Path where the new dataset will be created.
            src (Union[str, Dataset]): Path or dataset object that will be used as the template for the new dataset.
            tensors (List[str], optional): Names of tensors (and groups) to be replicated.
                If not specified all tensors in source dataset are considered.
            overwrite (bool): If True and a dataset exists at `destination`, it will be overwritten. Defaults to False.
            creds (dict, str, optional): credentials for cloud.
            verbose (bool): If True, logs will be printed. Defaults to ``True``.

        Returns:
            Dataset: New dataset object.
        """

        if isinstance(src, str):
            src = convert_pathlib_to_string_if_needed(src)
            source_ds = DatasetAPI.load(src, verbose=verbose)
            src_path = src
        else:
            source_ds = src
            src_path = src.path

        if tensors:
            tensors = source_ds.resolve_tensor_list(tensors)  # type: ignore
        else:
            tensors = list(source_ds.tensors.keys())  # type: ignore

        if isinstance(dest, Dataset):
            destination_ds = dest
        else:
            dest = convert_pathlib_to_string_if_needed(dest)
            dest_path = dest
            if dest_path == src_path:
                destination_ds = DatasetAPI.load(
                    dest_path, read_only=False
                )
            else:
                destination_ds = DatasetAPI.empty(
                    dest_path,
                    overwrite=overwrite,  # type: ignore
                )

        for tensor_name in tensors:  # type: ignore
            source_tensor = source_ds[tensor_name]
            if overwrite and tensor_name in destination_ds:
                destination_ds.delete_tensor(tensor_name)
            destination_ds.create_tensor_like(tensor_name, source_tensor)  # type: ignore

        destination_ds.info.update(source_ds.info.__getstate__())  # type: ignore

        return destination_ds

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
        org_dicts = DatasetAPI.get_data_with_dict_from_file(ori_path, schema)
        return DatasetAPI._create_dataset(org_dicts, muller_path, schema, workers, scheduler,
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
        org_dicts = DatasetAPI.get_data_with_dict_from_dataframes(dataframes, schema)
        return DatasetAPI._create_dataset(org_dicts, muller_path, schema, workers, scheduler,
                                                        disable_rechunk, progressbar, ignore_errors, split_tensor_meta)

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

        return DatasetAPI.get_data_with_dict_from_dataframes(dataframes, schema)

    @staticmethod
    def get_data_with_dict_from_dataframes(dataframes, schema):
        """Function to get data with dict from dataframes."""
        dicts = []
        schema_has_dict_values = any(
            isinstance(value, dict) for value in schema.values()) if schema is not None else False
        for dataframe in dataframes:
            if schema_has_dict_values:
                dataframe = DatasetAPI._extract_dataset_dict(dataframe)
            dicts.append(dataframe)
        if not dicts:
            raise ValueError("The input cannot be empty.")

        return dicts

    @staticmethod
    def convert_schema(schema, parent_key=''):
        """Function to convert schema."""
        converted_schema = {}
        for key, value in schema.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, tuple):
                converted_schema[new_key] = value
            elif isinstance(value, dict):
                converted_schema.update(DatasetAPI.convert_schema(value, parent_key=new_key))
        return converted_schema

    @staticmethod
    def _load(
            dataset_kwargs,
            check_integrity=True):
        """Load the dataset using the input kwags"""
        ret = Dataset(**dataset_kwargs)

        if check_integrity:
            integrity_check(ret)

        verbose = dataset_kwargs.get("verbose")
        processed_path = dataset_kwargs.get("path")

        if verbose:
            logging.info(f"{processed_path} loaded successfully.")
        return ret

    @staticmethod
    def _reset_and_load(
            storage,
            dataset_kwargs,
            address,
            err):
        """Reset and then load the dataset. Only called when loading dataset errored out with ``err``."""

        try:
            version_info = load_version_info(storage)
        except Exception as e:
            raise err from e

        address = address or "main"
        parent_commit_id, reset_commit_id = get_parent_and_reset_commit_ids(
            version_info, address
        )
        if parent_commit_id is False:
            # non-head node corrupted
            raise err
        if storage.read_only:
            msg = "Cannot reset when loading dataset in read-only mode."
            if parent_commit_id:
                msg += " However, you can try loading the previous commit using "
                msg += f"`muller.load('{dataset_kwargs.get('path')}@{parent_commit_id}')`."
            raise ReadOnlyModeError(msg)
        if parent_commit_id is None:
            # no commits in the dataset
            storage.clear()
            ds = DatasetAPI._load(dataset_kwargs)
            return ds

        # load previous version, replace head and checkout to new head
        dataset_kwargs["address"] = parent_commit_id
        ds = DatasetAPI._load(dataset_kwargs)
        new_commit_id = replace_head(storage, ds.version_state, ds.tensors, reset_commit_id)
        ds.checkout(new_commit_id)

        return ds

    @staticmethod
    def _valiadation(
            path,
            path_verify,
            read_only,
            creds,
            memory_cache_size,
            local_cache_size,
            ):
        processed_path, address = process_dataset_path(path)
        if path_verify:
            verify_dataset_name(processed_path)

        if creds is None:
            creds = {}

        try:
            storage, cache_chain = get_storage_and_cache_chain(  # LocalStorage/MemStorage/RomaStorage here
                path=processed_path,
                read_only=read_only,
                creds=creds,
                memory_cache_size=memory_cache_size,
                local_cache_size=local_cache_size,
            )
        except Exception as e:
            raise e
        return processed_path, address, storage, cache_chain

    @staticmethod
    def _extract_dataset_dict(data):
        """Function to extract dataset dict."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                try:
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, dict):
                        nested_dict = DatasetAPI._extract_dataset_dict(parsed_value)
                        for nested_key, nested_value in nested_dict.items():
                            result[f"{key}.{nested_key}"] = nested_value
                    else:
                        result[key] = parsed_value
                except ValueError:
                    result[key] = value
            elif isinstance(value, dict):
                nested_dict = DatasetAPI._extract_dataset_dict(value)
                for nested_key, nested_value in nested_dict.items():
                    result[f"{key}.{nested_key}"] = nested_value
            elif isinstance(value, list):
                result[key] = ', '.join(map(str, value))
            else:
                result[key] = value
        return result

    @staticmethod
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
        """Function to create dataset."""
        ds = muller.dataset(path=muller_path, overwrite=True, split_tensor_meta=split_tensor_meta)
        if not schema:
            schema = list(org_dicts[0].keys())
            for col_name in schema:
                ds.create_tensor(col_name)
        else:
            schema = DatasetAPI.convert_schema(schema)
            for col_name, (htype, dtype, sample_compression) in schema.items():
                ds.create_tensor(col_name, htype=htype, dtype=dtype, exist_ok=True,
                                 sample_compression=sample_compression)

        ds = DatasetAPI._append_data(workers, ds, org_dicts, schema, scheduler, disable_rechunk,
                                     progressbar, ignore_errors)
        return ds

    @staticmethod
    def _append_data(workers, ds, org_dicts, schema, scheduler, disable_rechunk, progressbar, ignore_errors):
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

    @staticmethod
    def _check_is_file(ds_path, key):
        fpath = posixpath.join(ds_path, key)
        fpath = os.path.expanduser(fpath)
        fpath = str(pathlib.Path(fpath))
        if os.path.isdir(fpath):
            raise DirectoryAtPathException
        return fpath
