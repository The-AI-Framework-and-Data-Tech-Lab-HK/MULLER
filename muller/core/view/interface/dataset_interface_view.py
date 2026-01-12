# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/dataset/dataset.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import json
import os
import pathlib
import warnings
from functools import partial
from time import time
from typing import Any, List, Optional, Union, Dict

import numpy as np

import muller
from muller.constants import FIRST_COMMIT_ID, VDS_INDEX
from muller.core.index import Index, IndexEntry
from muller.core.lock import Lock
from muller.core.view.view_entry import ViewEntry
from muller.core.storage import MemoryProvider
from muller.util.authorization import obtain_current_user
from muller.util.exceptions import DatasetViewSavingError
from muller.util.hash import hash_inputs
from muller.util.keys import get_queries_key, get_queries_lock_key
from muller.util.path import convert_pathlib_to_string_if_needed


class LockQueriesJson:
    def __init__(self, ds):
        self.dataset = ds
        self.storage_read_only = False  # New added
        self.lock = None # New added

    def __enter__(self):
        storage = self.dataset.base_storage
        self.storage_read_only = storage.read_only
        if self.dataset.locked_out:
            # Ignore storage level lock since we have file level lock
            storage.read_only = False
        lock = Lock(storage, get_queries_lock_key())
        lock.acquire(timeout=10)
        self.lock = lock

    def __exit__(self, *_, **__):
        self.lock.release()
        self.dataset.base_storage.read_only = self.storage_read_only


def get_views(ds, commit_id: Optional[str] = None) -> List[ViewEntry]:
    """Returns list of views stored in this Dataset.

    Args:
        ds (Dataset): Dataset to get views from.
        commit_id (str, optional): - Commit from which views should be returned.
            - If not specified, views from all commits are returned.

    Returns:
        List[ViewEntry]: List of :class:`ViewEntry` instances.
    """
    queries = _read_queries_json(ds)
    if commit_id is not None:
        queries = filter(
            lambda x: x["source-dataset-version"] == commit_id, queries
        )
    return list(map(partial(ViewEntry, dataset=ds), queries))


def get_view(ds, view_id: str) -> ViewEntry:
    """Returns the dataset view corresponding to ``id``.

    Examples:
        >>> # save view
        >>> ds[:100].save_view(id="first_100")
        >>> # load view
        >>> first_100 = ds.get_view("first_100").load()
        >>> # 100
        >>> print(len(first_100))

        See :func:`Dataset.save_view` to learn more about saving views.

    Args:
        ds (Dataset): Dataset to get views from.
        view_id (str): id of required view.

    Returns:
        ViewEntry

    Raises:
        KeyError: If no such view exists.
    """
    queries = _read_queries_json(ds)
    for q in queries:
        if q["id"] == view_id:
            return ViewEntry(q, ds)
    raise KeyError(f"No view with id {view_id} found in the dataset.")


def save_view(
        ds,
        message: Optional[str] = None,
        path: Optional[Union[str, pathlib.Path]] = None,
        view_id: Optional[str] = None,
        optimize: bool = False,
        tensors: Optional[List[str]] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        ignore_errors: bool = False,
        **ds_args,
) -> str:
    """Saves a dataset view as a virtual dataset (VDS)

    Examples:

        >>> # Save to specified path
        >>> vds_path = ds[:10].save_view(path="views/first_10", id="first_10")
        >>> vds_path
        views/first_10

        >>> # Path unspecified
        >>> vds_path = ds[:100].save_view(id="first_100", message="first 100 samples")
        >>> # vds_path = path/to/dataset

        >>> # Random id
        >>> vds_path = ds[:100].save_view()
        >>> # vds_path = path/to/dataset/.queries/92f41922ed0471ec2d27690b7351fc96bea060e6c5ee22b14f7ffa5f291aa068

        See :func:`Dataset.get_view` to learn how to load views by id.
        These virtual datasets can also be loaded from their path like normal datasets.

    Args:
        ds (Dataset): Dataset of the view to be saved.
        message (Optional, str): Custom user message.
        path (Optional, str, pathlib.Path): - The VDS will be saved as a standalone dataset at the specified path.
            - If not specified, the VDS is saved under ``.queries`` subdirectory of the source dataset's storage.
        view_id (Optional, str): Unique id for this view. Random id will be generated if not specified.
        optimize (bool):
            - If ``True``, the dataset view will be optimized by copying and rechunking the required data.
              This is necessary to achieve fast streaming speeds when training models using the dataset view.
              The optimization process will take some time, depending on the size of the data.
            - You can also choose to optimize the saved view later by calling its :meth:`ViewEntry.optimize` method.
        tensors (List, optional): Names of tensors to be copied. If not specified all tensors are copied.
        num_workers (int): Number of workers to be used for optimization process.
            Applicable only if ``optimize=True``. Defaults to 0.
        scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded',
            'processed' and 'distributed'. Only applicable if ``optimize=True``. Defaults to 'threaded'.
        ignore_errors (bool): Skip samples that cause errors while saving views.
             Only applicable if ``optimize=True``. Defaults to ``False``.
        ds_args (dict): Additional args for creating VDS when path is specified.
             (See documentation for :func:`muller.dataset()`)

    Returns:
        str: Path to the saved VDS.

    Raises:
        ReadOnlyModeError: When attempting to save a view inplace and the user doesn't have write access.
        DatasetViewSavingError: If HEAD node has uncommitted changes.
        TypeError: If ``id`` is not of type ``str``.

    Note:
        Specifying ``path`` makes the view external. External views cannot be accessed using
             the parent dataset's :func:`Dataset.get_view`,  :func:`Dataset.load_view`,
             :func:`Dataset.delete_view` methods. They have to be loaded using :func:`muller.load`.
    """

    if view_id is not None and not isinstance(view_id, str):
        raise TypeError(f"id {view_id} is of type {type(view_id)}, expected `str`.")
    return _save_view(
        ds,
        path,
        view_id,
        message or ds.query_string,
        optimize,
        tensors,
        num_workers,
        scheduler,
        False,
        ignore_errors,
        **ds_args,
    )


def load_view(
        ds,
        view_id: str,
        optimize: Optional[bool] = False,
        tensors: Optional[List[str]] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: Optional[bool] = True,
):
    """Loads the view and returns the :class:`~muller.core.dataset.dataset.Dataset` by id.
       Equivalent to ds.get_view(id).load().

    Args:
        ds (Dataset): dataset of the view to be loaded.
        view_id (str): id of the view to be loaded.
        optimize (bool): If ``True``, the dataset view is optimized by copying and rechunking the required data
            before loading. This is necessary to achieve fast streaming speeds when training models using the
            dataset view. The optimization process will take some time, depending on the size of the data.
        tensors (Optional, List[str]): Tensors to be copied if `optimize=True`. By default all tensors are copied.
        num_workers (int): Number of workers to be used for the optimization process.
            Only applicable if `optimize=True`. Defaults to 0.
        scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded',
            'processed' and 'distributed'. Only applicable if `optimize=True`. Defaults to 'threaded'.
        progressbar (bool): Whether to use progressbar for optimization. Only applicable if `optimize=True`.
            Defaults to True.

    Returns:
        Dataset: The loaded view.

    Raises:
        KeyError: if view with given id does not exist.
    """

    view = get_view(ds, view_id)
    if optimize:
        return view.optimize(   # Sherry: not implemented yet
            tensors=tensors,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
        ).load()
    return view.load()


def delete_view(ds, view_id: str):
    """Deletes the view with given view id.

    Args:
        ds (Dataset): The dataset that the view should be deleted from.
        view_id (str): Id of the view to delete.

    Raises:
        KeyError: if view with given id does not exist.
    """

    try:
        with _lock_queries_json(ds):
            qjson = _read_queries_json(ds)
            for i, q in enumerate(qjson):
                if q["id"] == view_id:
                    qjson.pop(i)
                    ds.base_storage.subdir(".queries/" + (q.get("path") or q["id"])).clear()
                    _write_queries_json(ds, qjson)
                    return
    except KeyError as e:
        raise KeyError(f"No view with id {view_id} found in the dataset.") from e


def get_view_for_vds(ds, inherit_creds=True, creds: Optional[Dict] = None):
    """Returns a view for this VDS. Only works if this Dataset is a virtual dataset.

    Returns:
        A view of the source dataset based on the indices from VDS.

    Args:
        inherit_creds (bool): Whether to inherit creds from the parent dataset in which this vds is stored.
            Default True.
        creds (optional, Dict): Creds for the source dataset. Used only if inherit_creds is False.

    Raises:
        Exception: If this is not a VDS.
    """

    try:
        commit_id = ds.info["source-dataset-version"]
    except KeyError as e:
        raise Exception("Dataset._get_view() works only for virtual datasets.") from e
    final_ds = (
        ds.parent_dataset[Index()]
        if (inherit_creds and ds.parent_dataset)
        else muller.load(
            ds.info["source-dataset"],
            verbose=False,
            creds=creds,
            read_only=True,
        )
    )

    final_ds.index = Index()
    final_ds.version_state = final_ds.version_state.copy()
    final_ds.protect_checkout(commit_id, verbose=False)
    first_index_subscriptable = ds.info.get("first-index-subscriptable", True)
    if first_index_subscriptable:
        index_entries = [IndexEntry(ds.VDS_INDEX.numpy().reshape(-1).tolist())]
    else:
        index_entries = [IndexEntry(int(ds.VDS_INDEX.numpy()))]
    sub_sample_index = ds.info.get("sub-sample-index")
    if sub_sample_index:
        index_entries += Index.from_json(sub_sample_index).values
    ret = final_ds[Index(index_entries)]
    ret.vds = ds
    return ret


def _read_queries_json(ds) -> list:
    try:
        return json.loads(ds.base_storage[get_queries_key()].decode("utf-8"))
    except KeyError:
        return []


def _save_view(
        ds,
        path: Optional[Union[str, pathlib.Path]] = None,
        view_id: Optional[str] = None,
        message: Optional[str] = None,
        optimize: bool = False,
        tensors: Optional[List[str]] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        _ret_ds: bool = False,
        ignore_errors: bool = False,
        **ds_args,
) -> Union[str, Any]:
    """Saves a dataset view as a virtual dataset (VDS)

    Args:
        ds (Dataset): Dataset to be saved.
        path (Optional, str, pathlib.Path): If specified, the VDS will saved as a standalone dataset at the
            specified path. If not, the VDS is saved under `.queries` subdirectory of the source dataset's storage.
        view_id (Optional, str): Unique id for this view.
        message (Optional, message): Custom user message.
        optimize (bool): Whether the view should be optimized by copying the required data. Default False.
        tensors (Optional, List[str]): Tensors to be copied if `optimize=True`. By default all tensors are copied.
        num_workers (int): Number of workers to be used if `optimize` is True.
        scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded',
            'processed' and 'distributed'. Only applicable if ``optimize=True``. Defaults to 'threaded'.
        _ret_ds (bool): If ``True``, the VDS is returned as such without converting it to a view. If ``False``,
            the VDS path is returned. Default False.
        ignore_errors (bool): Skip samples that cause errors while saving views.
            Only applicable if ``optimize=True``. Defaults to ``False``.
        ds_args (dict): Additional args for creating VDS when path is specified.
            (See documentation for `muller.dataset()`)

    Returns:
        If ``_ret_ds`` is ``True``, the VDS is returned, else path to the VDS is returned.

    Raises:
        ReadOnlyModeError: When attempting to save a view inplace and the user doesn't have write access.
        NotImplementedError: When attempting to save in-memory datasets.
    """

    path = convert_pathlib_to_string_if_needed(path)
    ds_args["verbose"] = False
    vds = None
    if path is None and ds.vds is not None:
        vds = ds.vds
        vds_id = vds.info["id"]
        if view_id is not None and vds_id != view_id:
            vds = None
            warnings.warn(
                f"This view is already saved with id '{vds_id}'. A copy of this view will be created with "
                f"the provided id '{vds_id}'"
            )
    base = ds.view_base or ds
    if not base.read_only:
        base.flush()
    if vds is None:
        if path is None:
            if isinstance(ds, MemoryProvider):
                raise NotImplementedError(
                    "Saving views inplace is not supported for in-memory datasets."
                )
            if ds.read_only and not base.locked_out and not ds.is_optimized:
                raise Exception("do not supported save view in remote yet")  # Sherry: add roma here

            vds = _save_view_in_subdir(
                ds,
                view_id,
                message,
                optimize,
                tensors,
                num_workers,
                scheduler,
                ignore_errors,
            )
        else:
            vds = _save_view_in_path(
                ds,
                path,
                view_id,
                message,
                optimize,
                tensors,
                num_workers,
                scheduler,
                ignore_errors,
                **ds_args,
            )
    if _ret_ds:
        return vds
    return vds.path


def _save_view_in_subdir(
        ds,
        view_id: Optional[str],
        message: Optional[str],
        copy: bool,
        tensors: Optional[List[str]],
        num_workers: int,
        scheduler: str,
        ignore_errors: bool,
):
    """Saves this view under ".queries" sub directory of same storage."""
    info = _get_view_info(ds, view_id, message, copy)
    ds.meta.info = info
    # creating sub-view of optimized view
    if ds.is_optimized:
        final_ds = ds.view_entry.src_ds.no_view_dataset

        if copy:
            view_info = ds.view_entry.info
            info["source-dataset"] = final_ds.path
            info["source-dataset-version"] = view_info["source-dataset-version"]
            if "source-dataset-index" in view_info:
                original_idx = Index.from_json(view_info["source-dataset-index"])
                combined_idx = original_idx[ds.index]
                info["source-dataset-index"] = combined_idx.to_json()
        else:
            info["source-dataset-version"] = (
                    info["source-dataset-version"] or FIRST_COMMIT_ID
            )
    else:
        final_ds = ds
    path = f".queries/{info['id']}"
    vds = final_ds.sub_ds(path, empty=True, verbose=False)
    _write_vds(ds, vds, info, copy, tensors, num_workers, scheduler, ignore_errors)
    _append_to_queries_json(ds, info)
    return vds


def _append_to_queries_json(ds, info: dict):
    """Append to queries json. """
    with _lock_queries_json(ds):
        qjson = _read_queries_json(ds)
        idx = None

        for i, temp_qjson in enumerate(qjson):
            if temp_qjson["id"] == info["id"]:
                idx = i
                break
        if idx is None:
            qjson.append(info)
        else:
            qjson[idx] = info
        _write_queries_json(ds, qjson)


def _save_view_in_path(
        ds,
        path: str,
        view_id: Optional[str],
        message: Optional[str],
        copy: bool,
        tensors: Optional[List[str]],
        num_workers: int,
        scheduler: str,
        ignore_errors: bool,
        **ds_args,
):
    """Saves this view at a given dataset path"""
    if os.path.abspath(path) == os.path.abspath(ds.path):
        raise DatasetViewSavingError("Rewriting parent dataset is not allowed.")
    try:
        vds = muller.empty(path, **ds_args)
    except Exception as e:
        raise DatasetViewSavingError from e
    info = _get_view_info(ds, view_id, message, copy)
    ds.meta.info = info
    _write_vds(ds, vds, info, copy, tensors, num_workers, scheduler, ignore_errors)
    return vds


def _get_view_info(
        ds,
        view_id: Optional[str] = None,
        message: Optional[str] = None,
        copy: bool = False,
):
    if ds.has_head_changes and not ds.is_optimized:
        raise DatasetViewSavingError(
            "The dataset's HEAD node has uncommitted changes. Please create a commit on"
            " the dataset object [ds.commit(<insert optional message>)] prior to saving the view."
        )
    commit_id = ds.commit_id
    tm = getattr(ds, "_created_at", time())
    view_id = _view_hash(ds) if view_id is None else view_id
    info = {"id": view_id, "virtual-datasource": not copy, "source-dataset": ds.path,
            "source-dataset-version": commit_id, "created_at": tm, "uid": obtain_current_user()}

    if message is not None:
        info["message"] = message
    query = getattr(ds, "_query", None)
    if query:
        info["query"] = query
        info["source-dataset-index"] = getattr(ds, "_source_ds_idx", None)
    tql_query = getattr(ds, "_tql_query", None)
    if tql_query:
        info["tql_query"] = tql_query
        info["source-dataset-index"] = getattr(ds, "_source_ds_idx", None)
    return info


def _write_vds(
        ds,
        vds,
        info: dict,
        copy: Optional[bool] = False,
        tensors: Optional[List[str]] = None,
        num_workers: Optional[int] = 0,
        scheduler: str = "threaded",
        ignore_errors: bool = False,
):
    """Writes the indices of this view to a vds."""
    vds.allow_view_updates = True
    try:
        with vds:
            if copy:
                _ = _copy(
                    ds,
                    vds,
                    tensors=tensors,
                    num_workers=num_workers,
                    scheduler=scheduler,
                    create_vds_index_tensor=True,
                    ignore_errors=ignore_errors,
                )
            else:
                vds.create_tensor(
                    VDS_INDEX,
                    dtype="uint64",
                    create_shape_tensor=False,
                    create_id_tensor=False,
                    create_sample_info_tensor=False,
                ).extend(
                    np.array(
                        tuple(ds.index.values[0].indices(ds.num_samples)),
                        dtype="uint64",
                    ),
                    progressbar=True,
                )
                info["first-index-subscriptable"] = ds.index.subscriptable_at(0)
                if len(ds.index) > 1:
                    info["sub-sample-index"] = Index(
                        ds.index.values[1:]
                    ).to_json()
            vds.info.update(info)
            ds.meta.info = info
    finally:
        try:
            delattr(vds, "_allow_view_updates")
        except AttributeError:  # Attribute already deleted by _copy()
            pass


def _copy(
        ds,
        dest: Union[str, pathlib.Path],
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        num_workers: int = 0,
        scheduler="threaded",
        progressbar=True,
        create_vds_index_tensor: bool = False,
        ignore_errors: bool = False,
        verbose: bool = True,
):
    dest_ds = muller.like(
        dest,
        ds,
        tensors=tensors,
        overwrite=overwrite,
        verbose=verbose,
    )

    return _process_copy(ds, dest_ds, num_workers, scheduler, progressbar, ignore_errors, create_vds_index_tensor)


def _process_copy(ds, dest_ds, num_workers, scheduler, progressbar, ignore_errors, create_vds_index_tensor):
    extend_only = True

    def _copy_tensor_extend(sample_in, sample_out):
        for tensor_name in dest_ds.tensors:
            sample_out[tensor_name].extend(sample_in[tensor_name])

    if not ds.index.subscriptable_at(0):
        old_first_index = ds.index.values[0]
        new_first_index = IndexEntry(
            slice(ds.index.values[0].value, ds.index.values[0].value + 1)
        )
        ds.index.values[0] = new_first_index
        reset_index = True
    else:
        reset_index = False
    try:
        muller.compute(
            _copy_tensor_extend,
            name="copy transform",
        )().eval(
            ds,
            dest_ds,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
            skip_ok=True,
            check_lengths=False,
            ignore_errors=ignore_errors,
            disable_label_sync=True,
            extend_only=extend_only,
        )

        dest_ds.flush()
        if create_vds_index_tensor:
            with dest_ds:
                dest_ds.allow_view_updates = True
                try:
                    dest_ds.create_tensor(
                        "VDS_INDEX",
                        dtype=np.uint64,
                        hidden=True,
                        create_shape_tensor=False,
                        create_id_tensor=False,
                        create_sample_info_tensor=False,
                    )
                    dest_ds.VDS_INDEX.extend(list(ds.sample_indices))
                finally:
                    delattr(dest_ds, "_allow_view_updates")
    finally:
        if reset_index:
            dest_ds.meta.default_index = Index([IndexEntry(0)]).to_json()
            dest_ds.meta.is_dirty = True
            dest_ds.flush()
            dest_ds = dest_ds[0]
            ds.index.values[0] = old_first_index
    return dest_ds


def _view_hash(ds) -> str:
    """Generates a unique hash for a filtered dataset view."""
    return hash_inputs(
        ds.path,
        *[e.value for e in ds.index.values],
        ds.pending_commit_id,
        getattr(ds, "_query", None),
        getattr(ds, "_tql_query", None),
    )


def _lock_queries_json(ds):
    return LockQueriesJson(ds)


def _write_queries_json(ds, data: list):
    read_only = ds.base_storage.read_only
    ds.base_storage.disable_readonly()
    try:
        ds.base_storage[get_queries_key()] = json.dumps(data).encode("utf-8")
    finally:
        if read_only:
            ds.base_storage.enable_readonly()
