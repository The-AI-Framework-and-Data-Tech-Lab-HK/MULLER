# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Modifications Copyright (c) 2026 Xueling Lin

"""Core dataset operations: create, load, empty, delete, get_col_info."""

import logging
import os
import pathlib
import posixpath
from typing import Dict, Optional, Union

from muller.constants import DEFAULT_LOCAL_CACHE_SIZE, DEFAULT_MEMORY_CACHE_SIZE, TENSOR_META_FILENAME
from muller.core.dataset.dataset import Dataset
from muller.core.version_control.functions import (
    load_version_info,
    get_parent_and_reset_commit_ids,
    replace_head,
    integrity_check,
)
from muller.util.exceptions import (
    AgreementError,
    CheckoutError,
    LockedException,
    ReadOnlyModeError,
    DatasetCreationError,
    DatasetCorruptionError,
    DatasetNotExistsError,
    DatasetViewDeletionError,
    DirectoryAtPathException,
)
from muller.core.storage_keys import dataset_exists
from muller.util.path import process_dataset_path, verify_dataset_name
from muller.core.auth.permission.invalid_user_op import validate_permissions
from muller.core.storage.factory import get_storage_and_cache_chain


def dataset(
        path: Union[str, pathlib.Path],
        read_only: Optional[bool] = None,
        overwrite: bool = False,
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        creds: Optional[Union[Dict, str]] = None,
        verbose: bool = True,
        reset: bool = False,
        check_integrity: bool = True,
        lock_enabled: Optional[bool] = True,
        lock_timeout: Optional[int] = 0,
        split_tensor_meta: bool = True
) -> Dataset:
    """
    Returns a :class:`~muller.core.dataset.Dataset` object referencing either a new or existing dataset.

    Examples:
        >>> ds = muller.dataset("./datasets/my_dataset", overwrite=True)
        >>> ds = muller.dataset("roma://mybucket/my_dataset")

    Args:
        path (str, pathlib.Path): The full path to the dataset. Can be:
          - an s3 path of the form ``s3://bucketname/path/to/dataset``.
          - a local file system path of the form ``./path/to/dataset``,``~/path/to/dataset`` or ``path/to/dataset``.
        read_only (bool, optional): Opens dataset in read only mode if this is passed as ``True``. Defaults to ``False``.
        overwrite (bool): If set to ``True`` this overwrites the dataset if it already exists. Defaults to ``False``.
        memory_cache_size (int): The size of the memory cache to be used in MB.
        local_cache_size (int): The size of the local filesystem cache to be used in MB.
        creds (dict, str, optional): credentials for OBS service.
        verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.
        reset (bool): If the specified dataset cannot be loaded due to a corrupted HEAD state, setting ``reset=True``
            will reset HEAD changes and load the previous version.
        check_integrity (bool, Optional): Performs an integrity check by default if the dataset has 20 or fewer tensors.
        lock_timeout (int): Number of seconds to wait before throwing a LockException.
        lock_enabled (bool): If true, the dataset manages a write lock.
        split_tensor_meta (bool): Each tensor has a tensor_meta.json if True. Default to True.
    """
    processed_path, address, _, cache_chain = _validation(
        path=path,
        path_verify=True,
        read_only=read_only,
        creds=creds,
        memory_cache_size=memory_cache_size,
        local_cache_size=local_cache_size
    )

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
        return _load(dataset_kwargs, check_integrity=check_integrity)
    except (AgreementError, CheckoutError, LockedException) as e:
        raise e from None
    except Exception as e:
        if create:
            raise e
        if not reset:
            raise DatasetCorruptionError from e
        return _reset_and_load(cache_chain, dataset_kwargs, address, e)


@validate_permissions
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
    Load dataset from given path.

    Args:
        path: The full path to the dataset.
        read_only: Opens dataset in read only mode if this is passed as ``True``. Defaults to ``False``.
        memory_cache_size: The size of the memory cache to be used in MB.
        local_cache_size: The size of the local filesystem cache to be used in MB.
        creds: credentials for OBS service.
        verbose: If ``True``, logs will be printed. Defaults to ``True``.
        check_integrity: Performs an integrity check by default if the dataset has 20 or fewer tensors.
        lock_enabled: If true, the dataset manages a write lock.
        lock_timeout: Number of seconds to wait before throwing a LockException.
        split_tensor_meta: Each tensor has a tensor_meta.json if True. Default to True.

    Returns:
        Dataset: The loaded dataset.
    """
    processed_path, address, _, cache_chain = _validation(
        path=path,
        path_verify=False,
        read_only=read_only,
        creds=creds,
        memory_cache_size=memory_cache_size,
        local_cache_size=local_cache_size
    )

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
        return _load(dataset_kwargs, check_integrity=check_integrity)
    except (AgreementError, CheckoutError, LockedException) as e:
        raise e from None
    except Exception as e:
        raise e


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
    """Creates an empty dataset.

    Args:
        path (str, pathlib.Path): The full path to the dataset.
        overwrite (bool): If set to ``True`` we overwrite the dataset if it already exists. Defaults to ``False``.
        memory_cache_size (int): The size of the memory cache to be used in MB.
        local_cache_size (int): The size of the local filesystem cache to be used in MB.
        creds (dict, str, optional): Credentials used to access the dataset.
        verbose (bool): If True, logs will be printed. Defaults to True.
        lock_timeout (int): Number of seconds to wait before throwing a LockException.
        lock_enabled (bool): If true, the dataset manages a write lock.
        split_tensor_meta (bool): Each tensor has a tensor_meta.json if True. Default to True.

    Returns:
        Dataset: Dataset created using the arguments provided.

    Danger:
        Setting ``overwrite`` to ``True`` will delete all of your data if it exists!
    """
    from muller.util.exceptions import DatasetAlreadyExistsError

    processed_path, address, storage, cache_chain = _validation(
        path=path,
        path_verify=True,
        read_only=False,
        creds=creds,
        memory_cache_size=memory_cache_size,
        local_cache_size=local_cache_size
    )

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
    return _load(dataset_kwargs)


def delete(
        path: Union[str, pathlib.Path],
        large_ok: bool = False,
        creds: Optional[Union[dict, str]] = None,
) -> None:
    """Delete a dataset at the given path."""
    qtokens = ["/.queries/", "\\.queries\\"]
    for qt in qtokens:
        if qt in path:
            raise DatasetViewDeletionError
    try:
        ds = load(path, verbose=False, creds=creds)
    except Exception as e:
        raise e
    ds.delete(large_ok)


def get_col_info(path: Union[str, pathlib.Path], col_name: str = None):
    """Get column information from dataset."""
    processed_path, _ = process_dataset_path(path)
    meta_file = TENSOR_META_FILENAME if col_name == "" else "/".join((col_name, TENSOR_META_FILENAME))
    try:
        fpath = _check_is_file(processed_path, meta_file)
        with open(fpath, "rb") as file:
            return file.read()
    except DirectoryAtPathException:
        raise
    except FileNotFoundError as e:
        raise KeyError(meta_file) from e


# Private helper functions

def _load(dataset_kwargs, check_integrity=True):
    """Load the dataset using the input kwargs."""
    ret = Dataset(**dataset_kwargs)

    if check_integrity:
        integrity_check(ret)

    verbose = dataset_kwargs.get("verbose")
    processed_path = dataset_kwargs.get("path")

    if verbose:
        logging.info(f"{processed_path} loaded successfully.")
    return ret


def _reset_and_load(storage, dataset_kwargs, address, err):
    """Reset and then load the dataset. Only called when loading dataset errored out."""
    try:
        version_info = load_version_info(storage)
    except Exception as e:
        raise err from e

    address = address or "main"
    parent_commit_id, reset_commit_id = get_parent_and_reset_commit_ids(version_info, address)

    if parent_commit_id is False:
        raise err

    if storage.read_only:
        msg = "Cannot reset when loading dataset in read-only mode."
        if parent_commit_id:
            msg += f" However, you can try loading the previous commit using "
            msg += f"`muller.load('{dataset_kwargs.get('path')}@{parent_commit_id}')`."
        raise ReadOnlyModeError(msg)

    if parent_commit_id is None:
        storage.clear()
        ds = _load(dataset_kwargs)
        return ds

    dataset_kwargs["address"] = parent_commit_id
    ds = _load(dataset_kwargs)
    new_commit_id = replace_head(storage, ds.version_state, ds.tensors, reset_commit_id)
    ds.checkout(new_commit_id)
    return ds


def _validation(path, path_verify, read_only, creds, memory_cache_size, local_cache_size):
    """Validate and process dataset path and credentials."""
    processed_path, address = process_dataset_path(path)
    if path_verify:
        verify_dataset_name(processed_path)

    if creds is None:
        creds = {}

    try:
        storage, cache_chain = get_storage_and_cache_chain(
            path=processed_path,
            read_only=read_only,
            creds=creds,
            memory_cache_size=memory_cache_size,
            local_cache_size=local_cache_size,
        )
    except Exception as e:
        raise e
    return processed_path, address, storage, cache_chain


def _check_is_file(ds_path, key):
    """Check if the path points to a file (not a directory)."""
    fpath = posixpath.join(ds_path, key)
    fpath = os.path.expanduser(fpath)
    fpath = str(pathlib.Path(fpath))
    if os.path.isdir(fpath):
        raise DirectoryAtPathException
    return fpath
