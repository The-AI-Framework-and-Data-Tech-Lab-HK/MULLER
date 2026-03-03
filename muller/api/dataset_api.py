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

"""
Legacy DatasetAPI class for backward compatibility.

This module provides the DatasetAPI class which delegates to the new modular API structure.
All methods are now simple wrappers that call the corresponding functions in muller.api.dataset.

For new code, prefer using the functions directly:
    - muller.dataset() instead of muller.api.DatasetAPI.dataset()
    - muller.load() instead of muller.api.DatasetAPI.load()
    etc.
"""

import pathlib
from typing import Dict, List, Optional, Union

from muller.core.dataset.dataset import Dataset

# Import from new modular structure
from muller.api.dataset import (
    dataset as _dataset,
    delete as _delete,
    empty as _empty,
    get_col_info as _get_col_info,
    like as _like,
    load as _load,
)
from muller.api.dataset.import_data import (
    from_dataframes as _from_dataframes,
    from_file as _from_file,
)


class DatasetAPI:
    """
    Legacy API class for backward compatibility.

    All methods delegate to the new modular API structure in muller.api.dataset.
    This class is maintained for backward compatibility only.

    Deprecated: Use the module-level functions directly instead:
        - muller.dataset() instead of DatasetAPI.dataset()
        - muller.load() instead of DatasetAPI.load()
        - etc.
    """

    @staticmethod
    def dataset(
            path: Union[str, pathlib.Path],
            read_only: Optional[bool] = None,
            overwrite: bool = False,
            memory_cache_size: int = 512,
            local_cache_size: int = 0,
            creds: Optional[Union[Dict, str]] = None,
            verbose: bool = True,
            reset: bool = False,
            check_integrity: bool = True,
            lock_enabled: Optional[bool] = True,
            lock_timeout: Optional[int] = 0,
            split_tensor_meta: bool = True
    ) -> Dataset:
        """Delegates to muller.api.dataset.dataset()"""
        return _dataset(
            path=path,
            read_only=read_only,
            overwrite=overwrite,
            memory_cache_size=memory_cache_size,
            local_cache_size=local_cache_size,
            creds=creds,
            verbose=verbose,
            reset=reset,
            check_integrity=check_integrity,
            lock_enabled=lock_enabled,
            lock_timeout=lock_timeout,
            split_tensor_meta=split_tensor_meta
        )

    @staticmethod
    def load(
            path: Union[str, pathlib.Path],
            read_only: Optional[bool] = None,
            memory_cache_size: int = 512,
            local_cache_size: int = 0,
            creds: Optional[Union[Dict, str]] = None,
            verbose: bool = True,
            check_integrity: bool = True,
            lock_enabled: Optional[bool] = True,
            lock_timeout: Optional[int] = 0,
            split_tensor_meta: bool = True,
    ) -> Dataset:
        """Delegates to muller.api.dataset.load()"""
        return _load(
            path=path,
            read_only=read_only,
            memory_cache_size=memory_cache_size,
            local_cache_size=local_cache_size,
            creds=creds,
            verbose=verbose,
            check_integrity=check_integrity,
            lock_enabled=lock_enabled,
            lock_timeout=lock_timeout,
            split_tensor_meta=split_tensor_meta
        )

    @staticmethod
    def get_col_info(path: Union[str, pathlib.Path], col_name: str = None):
        """Delegates to muller.api.dataset.get_col_info()"""
        return _get_col_info(path=path, col_name=col_name)

    @staticmethod
    def empty(
            path: Union[str, pathlib.Path],
            overwrite: bool = False,
            memory_cache_size: int = 512,
            local_cache_size: int = 0,
            creds: Optional[Union[Dict, str]] = None,
            lock_enabled: Optional[bool] = True,
            lock_timeout: Optional[int] = 0,
            verbose: bool = True,
            split_tensor_meta: bool = True,
    ) -> Dataset:
        """Delegates to muller.api.dataset.empty()"""
        return _empty(
            path=path,
            overwrite=overwrite,
            memory_cache_size=memory_cache_size,
            local_cache_size=local_cache_size,
            creds=creds,
            lock_enabled=lock_enabled,
            lock_timeout=lock_timeout,
            verbose=verbose,
            split_tensor_meta=split_tensor_meta
        )

    @staticmethod
    def delete(
            path: Union[str, pathlib.Path],
            large_ok: bool = False,
            creds: Optional[Union[dict, str]] = None,
    ) -> None:
        """Delegates to muller.api.dataset.delete()"""
        return _delete(path=path, large_ok=large_ok, creds=creds)

    @staticmethod
    def like(
            dest: Union[str, Dataset],
            src: Union[str, Dataset],
            tensors: Optional[List[str]] = None,
            overwrite: bool = False,
            verbose: bool = True,
    ) -> Dataset:
        """Delegates to muller.api.dataset.like()"""
        return _like(
            dest=dest,
            src=src,
            tensors=tensors,
            overwrite=overwrite,
            verbose=verbose
        )

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
        """Delegates to muller.api.dataset.from_file()"""
        return _from_file(
            ori_path=ori_path,
            muller_path=muller_path,
            schema=schema,
            workers=workers,
            scheduler=scheduler,
            disable_rechunk=disable_rechunk,
            progressbar=progressbar,
            ignore_errors=ignore_errors,
            split_tensor_meta=split_tensor_meta
        )

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
        """Delegates to muller.api.dataset.from_dataframes()"""
        return _from_dataframes(
            dataframes=dataframes,
            muller_path=muller_path,
            schema=schema,
            workers=workers,
            scheduler=scheduler,
            disable_rechunk=disable_rechunk,
            progressbar=progressbar,
            ignore_errors=ignore_errors,
            split_tensor_meta=split_tensor_meta
        )

    # Legacy helper methods - delegate to import_data module
    @staticmethod
    def get_data_with_dict_from_file(ori_path, schema):
        """Delegates to muller.api.dataset.import_data.get_data_with_dict_from_file()"""
        from muller.api.dataset.import_data import get_data_with_dict_from_file
        return get_data_with_dict_from_file(ori_path, schema)

    @staticmethod
    def get_data_with_dict_from_dataframes(dataframes, schema):
        """Delegates to muller.api.dataset.import_data.get_data_with_dict_from_dataframes()"""
        from muller.api.dataset.import_data import get_data_with_dict_from_dataframes
        return get_data_with_dict_from_dataframes(dataframes, schema)

    @staticmethod
    def convert_schema(schema, parent_key=''):
        """Delegates to muller.api.dataset.import_data.convert_schema()"""
        from muller.api.dataset.import_data import convert_schema
        return convert_schema(schema, parent_key)

    @staticmethod
    def _extract_dataset_dict(data):
        """Delegates to muller.api.dataset.import_data._extract_dataset_dict()"""
        from muller.api.dataset.import_data import _extract_dataset_dict
        return _extract_dataset_dict(data)
