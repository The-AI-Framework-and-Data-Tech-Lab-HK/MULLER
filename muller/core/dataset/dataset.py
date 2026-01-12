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
# Modifications Copyright (c) 2026 Xueling Lin

import heapq
import json
import os
import pathlib
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os import urandom
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set

import numpy as np
from numpy.typing import NDArray

import muller
from muller.api.info import Info
from muller.constants import (FIRST_COMMIT_ID, VDS_INDEX, CREATE_TENSOR_HIDDEN_UUID, DATASET_UUID_NAME,
                             INVERTED_INDEX_BATCH_SIZE, VIEW_SUMMARY_SAFE_LIMIT, TO_DATAFRAME_SAFE_LIMIT,
                             DEFAULT_MEMORY_CACHE_SIZE, DEFAULT_LOCAL_CACHE_SIZE, MB)
from muller.core.dataset.uuid.shard_hash import divide_to_shard, load_all_shards
from muller.core.index import Index
from muller.core.lock import lock_dataset, unlock_dataset
from muller.core.meta.dataset_meta import DatasetMeta
from muller.core.storage.local import LocalProvider
from muller.core.storage.lru_cache import LRUCache
from muller.core.tensor import Tensor
from muller.core.version_control.commit_node import CommitNode
from muller.core.version_control.dataset_diff import DatasetDiff
from muller.core.version_control.interface.diff_interface import get_changes_and_messages
from muller.core.view.view_entry import ViewEntry
from muller.htype import (UNSPECIFIED)
from muller.util.authorization import obtain_current_user
from muller.util.cache_chain import generate_chain
from muller.util.exceptions import (LockedException,
                                   ReadOnlyModeError,
                                   CheckoutError,
                                   PathNotEmptyException,
                                   CouldNotCreateNewDatasetException,
                                   TensorDoesNotExistError,
                                   EmptyCommitError,
                                   InvalidKeyTypeError,
                                   MemoryDatasetCanNotBePickledError,
                                   VersionControlError, InvalidJsonFileName, InvalidNumWorkers,
                                   SummaryLimit,
                                   ToDataFrameLimit, InvalidTensorList)
from muller.util.iteration_warning import (suppress_iteration_warning,
                                          check_if_iteration)
from muller.util.keys import dataset_exists, get_dataset_diff_key
from muller.util.keys import (get_dataset_meta_key)
from muller.util.path import get_path_from_storage, convert_pathlib_to_string_if_needed
from muller.util.permission.index_permission_check import index_permission_check
from muller.util.permission.invalid_view_op import invalid_view_op
from muller.util.permission.user_permission_check import user_permission_check
from muller.util.remove_cache import get_base_storage
from muller.util.spinner import spinner
from muller.util.version_control import load_meta, load_statistics, save_statistics
from muller.util.version_control import load_version_info
from muller.util.version_control import (rebuild_version_info,
                                        current_commit_has_change)
from muller.util.version_control import (save_version_info,
                                        save_commit_info,
                                        get_dataset_diff_at_commit)

_LOCKABLE_STORAGES = {LocalProvider}


class Dataset:
    def __init__(
            self,
            storage: LRUCache,
            index: Optional[Index] = None,
            read_only: Optional[bool] = None,
            verbose: bool = True,
            version_state: Optional[Dict[str, Any]] = None,
            path: Optional[Union[str, pathlib.Path]] = None,
            address: Optional[str] = None,
            is_iteration: bool = False,
            enabled_tensors: Optional[List[str]] = None,
            creds: Union[Dict, None] = None,
            **kwargs
    ):
        """
        Args:
            path (str, pathlib.Path): the path to initializes a new or existing dataset.
        """
        d: Dict[str, Any] = {}  # initialize d as a dictionary
        d["path"] = convert_pathlib_to_string_if_needed(path) or get_path_from_storage(storage)
        d["storage"] = storage
        d["_read_only_error"] = read_only is False
        d["base_storage"] = get_base_storage(storage)
        d["_read_only"] = d["base_storage"].read_only
        d["_locked_out"] = False  # User requested write access but was denied
        d["is_iteration"] = is_iteration
        d["is_first_load"] = version_state is None
        d["version_state"] = version_state or {}
        d["enabled_tensors"] = enabled_tensors

        dct = self.__dict__
        dct.update(d)

        self.path = convert_pathlib_to_string_if_needed(path) or get_path_from_storage(storage)
        self._is_filtered_view = False
        self.index = index or Index()
        self.version_state = version_state or {}
        self._locking_enabled = kwargs.get("lock_enabled", True)
        self._lock_timeout = kwargs.get("lock_timeout", 0)
        self.temp_tensors = []
        self.verbose = verbose
        self._vc_info_updated = True
        self._info = None
        self._ds_diff = None
        self._view_id = str(uuid.uuid4())
        self._view_base = kwargs.get("view_base", None)
        self._view_use_parent_commit = False
        self._parent_dataset = None
        self._query_string = None
        self.filtered_index = None
        self.split_tensor_meta = kwargs.get("split_tensor_meta", True)
        self.creds = creds
        self._vector_index = None

        self.append_only = True
        self.version_state = version_state or {}
        self._locked_out = False
        self._allow_view_updates = False

        # Filter
        self._query = None
        self._source_ds_idx = None
        self._create_at = None
        self._vds = None
        self._query_string = None

        # View Entry
        self._view_entry = None

        # Preprocessing steps
        self._preprocessing_meta(address=address)
        if enabled_tensors:
            self.enabled_tensors = set(self.resolve_tensor_list(enabled_tensors))
        else:
            self.enabled_tensors = None

        self.initial_autoflush: List[bool] = []  # This is a stack to support nested with contexts
        self._indexing_history: List[int] = []

        self.temp_tensors = self._initial_temp_tensors(self.temp_tensors)

        self.use_dataset_uuid = self._load_uuids()


    def __setstate__(self, state: Dict[str, Any]):
        state["is_first_load"] = True
        state["_info"] = None
        state["_read_only_error"] = False
        state["initial_autoflush"] = []
        state["_ds_diff"] = None
        state["_view_base"] = None
        state["temp_tensors"] = []
        state["_vc_info_updated"] = False
        state["_locked_out"] = False
        state["_vector_index"] = None
        self.__dict__.update(state)
        self.__dict__["base_storage"] = get_base_storage(self.storage)
        # clear cache while restoring
        self.storage.clear_cache_without_flush()
        self._set_derived_attributes(verbose=False)
        self._indexing_history = []

    def __setitem__(self, item: str, value: Any):
        if not isinstance(item, str):
            raise TypeError("Datasets do not support item assignment")
        tensor = self[item]
        tensor.index = Index()
        tensor._update(self.index, value)

    def __setattr__(self, name: str, value):
        try:
            # Dataset is not fully loaded if meta is not in version_state
            if "meta" in self.version_state:
                return self.__setitem__(name, value)
            raise TensorDoesNotExistError(name)
        except TensorDoesNotExistError as e:
            if isinstance(value, (np.ndarray, np.generic)):
                raise TypeError(
                    "Setting tensor attributes directly is not supported. To add a tensor, "
                    "use the `create_tensor` method."
                    + "To add data to a tensor, use the `append` and `extend` methods."
                ) from e
        return super().__setattr__(name, value)

    def __getstate__(self) -> Dict[str, Any]:
        """Returns a dict that can be pickled and used to restore this dataset.

        Note:
            Pickling a dataset cannot copy the dataset,
            it only saves attributes that can be used to restore the dataset.
            If you pickle a local dataset and try to access it on a machine that does not have the data present,
            the dataset will not work.
        """
        if self.path.startswith("mem://"):
            raise MemoryDatasetCanNotBePickledError
        keys = [
            "path",
            "_read_only",
            "index",
            "storage",
            "verbose",
            "version_state",
            "_is_filtered_view",
            "_view_id",
            "_view_use_parent_commit",
            "_locking_enabled",
            "_lock_timeout",
            "enabled_tensors",
            "is_iteration",
            "filtered_index",
            "split_tensor_meta",
            "use_dataset_uuid"
        ]
        state = {k: getattr(self, k) for k in keys}
        return state

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except TensorDoesNotExistError as ke:
            raise AttributeError(
                f"'{self.__class__}' object has no attribute '{key}'"
            ) from ke

    def __len__(self, warn: bool = True):
        """Returns the length of the smallest tensor."""
        tensor_names = []
        tensor_lengths = []
        for tensor in self.tensors.values():
            tensor_names.append(tensor)
            tensor_lengths.append(len(tensor))
        min_len = min(tensor_lengths, default=0)
        max_len = max(tensor_lengths, default=0)
        if warn and min_len != max_len:
            warnings.warn(f"The length of tensors in the dataset is different. "
                          f"The min length is {min_len} from {tensor_names[np.argmin(tensor_lengths)]}, "
                          f"while the max length is {max_len} from {tensor_names[np.argmin(tensor_lengths)]}."
                          f"The len(ds) returns the length of the smallest tensor in the dataset. "
                          f"Please use ds.max_len ig the length of the longest tensor is needed.")
        return min_len

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(
                i, is_iteration=not isinstance(self.index.values[0], list)
            )

    def __getitem__(
            self,
            item: Union[
                str, int, slice, List[int], Tuple[Union[int, slice, Tuple[int]]], Index
            ],
            is_iteration: bool = False,
    ):
        is_iteration = is_iteration or self.is_iteration
        if isinstance(item, str):
            return self._return_tensor(item, is_iteration)

        if isinstance(item, (int, slice, list, tuple, Index, type(Ellipsis))):

            def _is_list_of_str_or_list_of_list_of_str(tmp_item):
                if not isinstance(tmp_item, list):
                    return False
                if not tmp_item:
                    return False
                first = tmp_item[0]
                if isinstance(first, str):
                    return True
                if isinstance(first, (list, tuple)) and first:
                    return isinstance(first[0], str)
                return False

            if _is_list_of_str_or_list_of_list_of_str(item):
                enabled_tensors = list(item)
                ret = self.__class__(
                    storage=self.storage,
                    index=self.index,
                    read_only=self._read_only,
                    verbose=False,
                    version_state=self.version_state,
                    path=self.path,
                    is_iteration=is_iteration,
                    enabled_tensors=enabled_tensors,
                    view_base=self._view_base or self,
                )

            elif isinstance(item, tuple) and len(item) and isinstance(item[0], str):
                ret = self
                for x in item:
                    ret = self[x]
                return ret
            else:
                if not is_iteration and isinstance(item, int):
                    is_iteration = check_if_iteration(self._indexing_history, item)
                    if is_iteration and muller.constants.SHOW_ITERATION_WARNING:
                        warnings.warn(
                            "Indexing by integer in a for loop, like `for i in range(len(ds)): ... ds[i]` "
                            "can be quite slow. Use `for i, sample in enumerate(ds)` instead."
                        )

                ret = self.__class__(
                    storage=self.storage,
                    index=self.index[item],
                    read_only=self._read_only,
                    verbose=False,
                    version_state=self.version_state,
                    path=self.path,
                    is_iteration=is_iteration,
                    enabled_tensors=self.enabled_tensors,
                    view_base=self._view_base or self,
                    split_tensor_meta=self.split_tensor_meta,
                )
        else:
            raise InvalidKeyTypeError(item)
        if hasattr(self, "_view_entry"):
            ret.view_entry = self.view_entry
        return ret

    def __del__(self):
        try:
            unlock_dataset(self)
        except Exception:  # python shutting down
            pass

    def __enter__(self):
        self.initial_autoflush.append(self.storage.autoflush)
        self.storage.autoflush = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        autoflush = self.initial_autoflush.pop()
        if not self._read_only and autoflush:
            if self.vc_info_updated:
                self._flush_vc_info()
            spinner(self.storage.flush)()
        self.storage.autoflush = autoflush

    @property
    def meta(self) -> DatasetMeta:
        """Returns the metadata of the dataset."""
        return self.version_state.get("meta", None)

    @property
    def is_head_node(self):
        """Returns True if the current commit is the head node of the branch and False otherwise."""
        commit_node = self.version_state["commit_node"]
        return not commit_node.children

    @property
    def read_only(self):
        """Returns True if dataset is in read-only mode and False otherwise."""
        return self._read_only

    @property
    def branch(self) -> str:
        """The current branch of the dataset"""
        return self.version_state["branch"]

    @property
    def is_view(self) -> bool:
        """Returns ``True`` if this dataset is a view and ``False`` otherwise."""
        return (
                not self.index.is_trivial()
                or hasattr(self, "_vds")
                or hasattr(self, "_view_entry")
                or self.view_entry is not None
        )

    @property
    def is_optimized(self) -> bool:
        """Return true if the dataset is in optimized mode and False otherwise."""
        return not getattr(getattr(self, "_view_entry", None), "virtual", True)

    @property
    def no_view_dataset(self):
        """Returns the same dataset without slicing."""
        if self.index is None or self.index.is_trivial():
            return self
        return self.__class__(
            storage=self.storage,
            index=None,
            read_only=self.read_only,
            verbose=False,
            version_state=self.version_state,
            path=self.path,
            enabled_tensors=self.enabled_tensors,
        )

    @property
    def min_view(self):
        """Returns a view of the dataset in which all tensors are sliced to have the same length as
        the shortest tensor.

        Example:

            Creating a dataset with 5 images and 4 labels. ``ds.min_view`` will return a view in which tensors are
            sliced to have 4 samples.

            >>> import muller
            >>> ds = muller.dataset('../test/test_ds', overwrite=True)
            >>> ds.create_tensor('images', htype='image', sample_compression="jpg")
            >>> ds.create_tensor('labels', htype='class_label')
            >>> ds.images.extend([muller.read('../images') for _ in range(5)])
            >>> ds.labels.extend([0, 1, 2, 1])
            >>> len(ds.images)
            5
            >>> len(ds.labels)
            4
            >>> for i, sample in enumerate(ds.max_view):
            ...     print(sample['images'].shape, sample['labels'].numpy())
            ...
            (20, 20, 3) [0]
            (20, 20, 3) [1]
            (20, 20, 3) [2]
            (20, 20, 3) [1]

        """
        min_length = min(map(len, self.tensors.values()))
        return self[:min_length]

    @property
    def max_view(self):
        """Returns a view of the dataset in which shorter tensors are padded with ``None`` s to have the same length as
        the longest tensor.

        Example:

            Creating a dataset with 5 images and 4 labels. ``ds.max_view`` will return a view with ``labels`` tensor
            padded to have 5 samples.

            >>> import muller
            >>> ds = muller.dataset("../test/test_ds", overwrite=True)
            >>> ds.create_tensor("images", htype="link[image]", sample_compression="jpg")
            >>> ds.create_tensor("labels", htype="class_label")
            >>> ds.images.extend([muller.read("../images") for _ in range(5)])
            >>> ds.labels.extend([0, 1, 2, 1])
            >>> len(ds.images)
            5
            >>> len(ds.labels)
            4
            >>> for i, sample in enumerate(ds.max_view):
            ...     print(sample["images"].shape, sample["labels"].numpy())
            ...
            (20, 20, 3) [0]
            (20, 20, 3) [1]
            (20, 20, 3) [2]
            (20, 20, 3) [1]
            (20, 20, 3) [None]
        """
        return self.__class__(
            storage=self.storage,
            index=self.index,
            read_only=self.read_only,
            verbose=False,
            version_state=self.version_state,
            path=self.path,
            enabled_tensors=self.enabled_tensors,
        )

    @property
    def sample_indices(self):
        """Returns all the indices pointed to by this dataset view."""
        return self.get_sample_indices(min(t.num_samples for t in self.tensors.values()))

    @property
    def num_samples(self) -> int:
        """Returns the length of the smallest tensor.
        Ignores any applied indexing and returns the total length.
        """
        return min(
            map(
                len,
                filter(
                    lambda t: t.key not in self.meta.hidden_tensors,
                    self.version_state["full_tensors"].values(),
                ),
            ),
            default=0,
        )

    @property
    def max_len(self):
        """Return the maximum length of the tensor."""
        return 0 if len(self.tensors) == 0 else max([len(tensor) for tensor in self.tensors.values()])

    @property
    def min_len(self):
        """Return the minimum length of the tensor."""
        return 0 if len(self.tensors) == 0 else min([len(tensor) for tensor in self.tensors.values()])

    @property
    def has_head_changes(self):
        """Returns True if currently at head node and uncommitted changes are present."""
        return self.is_head_node and current_commit_has_change(
            self.version_state, self.storage
        )

    @property
    def info(self):
        """Returns the information about the dataset."""
        if self._info is None:
            self.__dict__["_info"] = Info(self)
        return self._info

    @info.setter
    def info(self, value):
        if isinstance(value, dict):
            info = self.info
            info.replace_with(value)
        elif value is None:
            self._info = value
        else:
            raise TypeError("Info must be set with type Dict")

    @property
    def _dataset_diff(self):
        if self._ds_diff is None:
            self.__dict__["_ds_diff"] = get_dataset_diff_at_commit(
                self.version_state["commit_node"].commit_id, self.storage)
        return self._ds_diff

    @property
    def get_dataset_diff(self):
        """Returns the dataset diff. """
        return self._dataset_diff

    @property
    def tensors(self) -> Dict[str, Tensor]:
        """All tensors belonging to this dataset. Always returns the sliced tensors."""
        return self.get_tensors(include_hidden=False, include_disabled=False)

    @property
    def branches(self):
        """Lists all the branches of the dataset. """
        return self.version_state.get("branch_info", "Not Supported")

    @property
    def commit_id(self) -> Optional[str]:
        """The lasted committed commit id of the dataset. If there are no commits, this returns ``None``."""
        commit_node = self.version_state["commit_node"]
        if not commit_node.is_head_node:
            return commit_node.commit_id

        parent = commit_node.parent
        if parent is None:
            return None

        return parent.commit_id

    @property
    def indexed_tensors(self) -> List[str]:
        """Returns the tensor column with inverted index built as a list."""
        indexed = []
        branch = self.version_state["branch"]
        meta_path = os.path.join("inverted_index_dir", branch, "meta.json")
        try:
            meta_json = json.loads(self.storage[meta_path].decode('utf-8'))
            for key in meta_json:
                indexed.append(key)
        except KeyError:
            pass

        return indexed

    @property
    def indexed_tensors_vec(self) -> Set[str]:
        """Returns the tensor column with inverted index (vectorized version) built as a list."""
        indexed = set()
        branch = self.version_state.get("branch", None)
        meta_path = os.path.join("inverted_index_dir_vec", branch, "meta.json")
        try:
            meta_json = json.loads(self.storage[meta_path].decode('utf-8'))
            for key in meta_json:
                if meta_json[key].get("commit_id", None) == self.commit_id:
                    indexed.add(key)
        except KeyError:
            pass
        return indexed

    @property
    def pending_commit_id(self) -> str:
        """The commit_id of the next commit that will be made to the dataset.
        If you're not at the head of the current branch, this will be the same as the commit_id.
        """
        return self.version_state["commit_id"]

    @property
    def is_filtered_view(self):
        """Returns whether this is a filtered view."""
        return self._is_filtered_view

    @is_filtered_view.setter
    def is_filtered_view(self, value):
        """Sets the filter_view value."""
        self._is_filtered_view = value

    @property
    def source_ds_idx(self):
        """Returns the source dataset index. """
        return self._source_ds_idx

    @source_ds_idx.setter
    def source_ds_idx(self, value):
        """Sets the source dataset index. """
        self._source_ds_idx = value

    @property
    def create_at(self):
        """Returns the create at value. """
        return self._create_at

    @create_at.setter
    def create_at(self, value):
        """Sets the create at value. """
        self._create_at = value

    @property
    def vds(self):
        """Returns the vds value. """
        return self._vds

    @vds.setter
    def vds(self, value):
        """Sets the vds value. """
        self._vds = value

    @property
    def view_entry(self):
        """Returns the view entry. """
        return self._view_entry

    @view_entry.setter
    def view_entry(self, value):
        """Sets the view entry. """
        self._view_entry = value

    @property
    def ds_diff(self):
        """Returns the dataset diff value. """
        return self._ds_diff

    @ds_diff.setter
    def ds_diff(self, value):
        """Sets the dataset diff value. """
        self._ds_diff = value

    @property
    def parent_dataset(self):
        """Returns the parent dataset. """
        return self._parent_dataset

    @parent_dataset.setter
    def parent_dataset(self, value):
        """Sets the parent dataset. """
        self._parent_dataset = value

    @property
    def locked_out(self):
        """Returns whether this dataset is locked out. """
        return self._locked_out

    @locked_out.setter
    def locked_out(self, value):
        """Sets whether this dataset is locked out. """
        self._locked_out = value

    @property
    def allow_view_updates(self):
        """Returns whether this dataset is allowed to view update. """
        return self._allow_view_updates

    @allow_view_updates.setter
    def allow_view_updates(self, value):
        """Set whether this dataset is allowed to view update. """
        self._allow_view_updates = value

    @property
    def vc_info_updated(self):
        """Returns whether the version control info is updated in this dataset. """
        return self._vc_info_updated

    @vc_info_updated.setter
    def vc_info_updated(self, value):
        """Sets the version control info update status of this dataset. """
        self._vc_info_updated = value

    @property
    def view_base(self):
        """Returns the base view. """
        return self._view_base

    @view_base.setter
    def view_base(self, value):
        """Sets the base view. """
        self._view_base = value

    @property
    def query_string(self):
        """Returns the query string. """
        return self._query_string

    @query_string.setter
    def query_string(self, value):
        """Sets the query string. """
        self._query_string = value

    @property
    def vector_index(self):
        """Returns the vector index. """
        return self._vector_index

    @vector_index.setter
    def vector_index(self, value):
        """Sets the vector index. """
        self._vector_index = value

    @staticmethod
    def _get_commit_id_for_address(address, version_state):
        if address in version_state["branch_commit_map"]:
            branch = address
            commit_id = version_state["branch_commit_map"][branch]
        elif address in version_state["commit_node_map"]:
            commit_id = address
        else:
            raise CheckoutError(
                f"Address {address} not found. Ensure the commit id / branch name is correct."
            )
        return commit_id


    @spinner
    def flush(self):
        """Necessary operation after writes if caches are being used.
        Writes all the dirty data from the cache layers (if any) to the underlying storage.
        Here dirty data corresponds to data that has been changed/assigned and but hasn't yet been sent to the
        underlying storage.
        """
        self._flush_vc_info()
        self.storage.flush()

    def maybe_flush(self):
        """Flush if necessary."""
        if not self._read_only:
            if self.storage.autoflush:
                if self.vc_info_updated:
                    self._flush_vc_info()
                self.storage.flush()

    def resolve_tensor_list(self, keys: List[str]) -> List[str]:
        """Resolve the tensor list."""
        ret = []
        for k in keys:
            fullpath = k
            if (
                    self.version_state["tensor_names"].get(fullpath)
                    in self.version_state["full_tensors"]
            ):
                ret.append(k)
            else:
                enabled_tensors = self.enabled_tensors
                if fullpath[-1] != "/":
                    fullpath = fullpath + "/"
                hidden = self.meta.hidden_tensors
                for temp_tensor in self.version_state["tensor_names"]:
                    temp_tensor_valid = temp_tensor.startswith(fullpath) and temp_tensor not in hidden
                    if temp_tensor_valid and (enabled_tensors is None or temp_tensor in enabled_tensors):
                        ret.append(temp_tensor)
        return ret

    @user_permission_check
    def create_tensor(
            self,
            name: str,
            htype: str = UNSPECIFIED,
            dtype: Union[str, np.dtype] = UNSPECIFIED,
            sample_compression: Union[str, None] = UNSPECIFIED,
            chunk_compression: str = UNSPECIFIED,
            hidden: bool = False,
            **kwargs,
    ):
        """ Create tensors. """
        return muller.core.dataset.create_tensor(self, name, htype, dtype, sample_compression,
                                                                  chunk_compression, hidden,
                                                                  **kwargs)

    @invalid_view_op
    @user_permission_check
    def create_tensor_like(
            self, name: str, source: "Tensor",
    ) -> "Tensor":
        """
        Copies the ``source`` tensor's meta information and creates a new tensor with it. No samples are copied,
        only the meta/info for the tensor is.
        """
        return muller.core.dataset.create_tensor_like(self, name, source)

    @invalid_view_op
    @user_permission_check
    def delete_tensor(self, name: str, large_ok: bool = False):
        """Delete a tensor."""
        return muller.core.dataset.delete_tensor(self, name, large_ok)

    @user_permission_check
    def extend(
            self,
            samples: Dict[str, Any],
            skip_ok: bool = False,
            append_empty: bool = False,
            ignore_errors: bool = False,
            progressbar: bool = False,
    ):
        """Extend samples to the dataset."""
        muller.core.dataset.extend(self, samples, skip_ok, append_empty, ignore_errors, progressbar)

    @invalid_view_op
    @user_permission_check
    def append(
            self,
            sample: Dict[str, Any],
            skip_ok: bool = False,
            append_empty: bool = False,
    ):
        """Append samples to the dataset."""
        muller.core.dataset.append(self, sample, skip_ok, append_empty)

    @user_permission_check
    def update(self, sample: Dict[str, Any]):
        """Update samples in the dataset."""
        muller.core.dataset.update(self, sample)

    @invalid_view_op
    @user_permission_check
    def pop(self, index: Optional[Union[List, int]] = None, rechunk: bool = False):
        """Pop samples in the dataset."""
        muller.core.dataset.pop(self, index, rechunk)

    @invalid_view_op
    @user_permission_check
    def delete(self, large_ok=False):
        """Delete the dataset."""
        muller.core.dataset.delete(self, large_ok)

    @invalid_view_op
    @user_permission_check
    def rename(self, path: Union[str, pathlib.Path]):
        """Renames the dataset to `path`. """
        # Note: currently we only accept the rename operation in LocalProvider and MemProvider
        muller.core.dataset.rename(self, path)

    def handle_rename_tensor(self, name, new_name):
        """Function to handle rename tensor"""
        muller.core.dataset.handle_rename_tensor(self, name, new_name)

    @invalid_view_op
    @user_permission_check
    def rename_tensor(self, name: str, new_name: str):
        """Renames tensor with name ``name`` to ``new_name``"""
        return muller.core.dataset.rename_tensor(self, name, new_name)

    @user_permission_check
    def add_data_from_file(self, ori_path="", schema=None, workers=0, scheduler="processed", disable_rechunk=True,
                           progressbar=True, ignore_errors=True):
        """Add samples from external files to the dataset."""
        if not ori_path:
            raise ValueError("ori_path cannot be empty.")

        org_dicts = muller.api.dataset_api.DatasetAPI.get_data_with_dict_from_file(ori_path, schema)
        return muller.core.dataset.add_data(self, org_dicts, schema, workers, scheduler, disable_rechunk, progressbar,
                                           ignore_errors)

    @user_permission_check
    def add_data_from_dataframes(self, dataframes=None, schema=None, workers=0, scheduler="processed",
                                 disable_rechunk=True, progressbar=True, ignore_errors=True):
        """Add samples from external dataframes to the dataset."""
        if not dataframes:
            raise ValueError("dataframes cannot be empty.")
        if not isinstance(dataframes, list):
            raise TypeError("Expected a list for dataframes")

        org_dicts = muller.api.dataset_api.DatasetAPI.get_data_with_dict_from_dataframes(dataframes, schema)
        return muller.core.dataset.add_data(self, org_dicts, schema, workers, scheduler, disable_rechunk, progressbar,
                                           ignore_errors)


    def numpy(self, aslist=False, fetch_chunks=False, asrow=False) -> Union[dict, list]:
        """Computes the contents of the dataset slices in numpy format."""
        return muller.core.dataset.to_numpy(self, aslist, fetch_chunks, asrow)

    @user_permission_check
    def rechunk(
            self,
            tensors: Optional[Union[str, List[str]]] = None,
            num_workers: int = 0,
            scheduler: str = "threaded",
            progressbar: bool = True,
    ):
        """Rechunk the dataset."""
        return muller.core.chunk.dataset_rechunk(self, tensors, num_workers, scheduler, progressbar)

    def rechunk_if_necessary(
            self,
            tensor_spec: Optional[Union[List[str], Dict[str, Optional[int]]]] = None,
            num_workers: int = 1
    ) -> None:
        """ Rechunk the data chunks on several tensors. """
        return muller.core.chunk.dataset_rechunk_if_necessary(self, tensor_spec, num_workers)

    def check_uuid(self):
        """Check uuids"""
        tensor_names = self.version_state.get("tensor_names", [])
        # The historical dataset is not affected by new designed dataset uuid
        if DATASET_UUID_NAME not in tensor_names and len(tensor_names) > 0:
            return
        if not CREATE_TENSOR_HIDDEN_UUID and DATASET_UUID_NAME not in tensor_names:
            muller.core.dataset.create_uuid_tensor(self)
            if len(self.tensors.keys()) > 0 and self[DATASET_UUID_NAME].size != self.max_len:
                warnings.warn(
                    f"uuid's length should be equal to the max length of dataset tensors, but uuid length is "
                    f"{self[DATASET_UUID_NAME].size} and dataset max length is {self.max_len}")

    def resize_uuid(self):
        """Manage dataset uuid. """
        if not self.use_dataset_uuid:
            return
        extend_size = self.max_len - self[DATASET_UUID_NAME].size
        if extend_size > 0:
            uuids = np.frombuffer(urandom(8 * extend_size), dtype=np.uint64).reshape(-1)
            self[DATASET_UUID_NAME].extend(uuids)
        elif extend_size == 0:
            return
        else:
            pop_index = []
            for i in range(self[DATASET_UUID_NAME].size - abs(extend_size), self[DATASET_UUID_NAME].size):
                pop_index.append(i)
            self[DATASET_UUID_NAME].pop(pop_index)

    def commits(self, ordered_by_date=False) -> List[Dict]:
        """Lists all the commits leading to the current dataset state."""
        return muller.core.version_control.commits(self, ordered_by_date)

    def get_commit_details(self, commit_id) -> Dict:
        """Get details of a particular commit."""
        return muller.core.version_control.get_commit_details(self, commit_id)

    @spinner
    @invalid_view_op
    @user_permission_check
    def commit(self, message: Optional[str] = None, allow_empty=False) -> str:
        """Stores a snapshot of the current state of the dataset."""

        if not allow_empty and not self.has_head_changes:
            raise EmptyCommitError(
                "There are no changes, commit is not done. Try again with allow_empty=True."
            )

        return muller.core.version_control.protected_commit(self, message, None, False)

    @invalid_view_op
    def checkout(
            self, address: str, create: bool = False, reset: bool = False
    ) -> Optional[str]:
        """
        Checks out to a specific commit_id or branch.
        If ``create = True``, creates a new branch named ``address``.
        """
        return muller.core.version_control.checkout(self, address, create, reset)

    @invalid_view_op
    def detect_merge_conflict(self, target_id: str, show_value: bool = False):
        """Detect the conflict between current stage and target stage of given commit id. """
        return muller.core.version_control.detect_merge_conflict(self, target_id, show_value)

    @spinner
    @invalid_view_op
    @suppress_iteration_warning
    @user_permission_check
    def merge(
            self,
            target_id: str,
            append_resolution: Optional[str] = None,
            update_resolution: Optional[str] = None,
            pop_resolution: Optional[str] = None,
            delete_removed_tensors: bool = False,
            force: bool = False,
    ):
        """Merges the target_id into the current dataset."""
        return muller.core.version_control.merge(self, target_id, append_resolution, update_resolution,
                                                pop_resolution, delete_removed_tensors, force)

    def protect_checkout(
            self,
            address: str,
            create: bool = False,
            commit_hash: Optional[str] = None,
            verbose: bool = True,
            flush_version_control_info: bool = False,
    ) -> Optional[str]:
        """Protected checkout."""
        return muller.core.version_control.protect_checkout(self, address, create, commit_hash, verbose,
                                                                   flush_version_control_info)

    def generate_add_update_value(self, commit_changes, offset, limit, asrow, tensors=None):
        """Obtain the details of the add/update/delete samples."""
        return muller.core.version_control.generate_add_update_value(self, commit_changes, offset,
                                                                            limit, asrow, tensors)

    def direct_diff(self, id_1: str = None, id_2: str = None,
                    as_dataframe: Optional[bool] = False, force: Optional[bool] = False):
        """Detect the direct difference of id_2 compared with id_1."""
        return muller.core.version_control.direct_diff(self, id_1, id_2, as_dataframe, force)

    def diff(
            self,
            id_1: Optional[str] = None,
            id_2: Optional[str] = None,
            as_dict: bool = False,
            show_value: bool = False,
            offset: int = 0,
            limit: Optional[int] = None,
            asrow: bool = False
    ) -> Optional[Dict]:
        """
        Returns/displays the differences between commits/branches.
        For each tensor this contains information about the sample indexes that were added/modified
        as well as whether the tensor was created.
        """

        return muller.core.version_control.diff(self, id_1, id_2, as_dict, show_value, offset, limit, asrow)

    def diff_to_prev(
            self,
            commit_id: str = None,
            as_dict=False,
            show_value=False,
            offset: int = 0,
            limit: Optional[int] = None,
            asrow: bool = False
    ) -> Optional[Dict]:
        """ Returns/displays the differences between the given commit/current commit and its previous commit. """
        return muller.core.version_control.diff_to_prev(self, commit_id, as_dict, show_value, offset, limit, asrow)

    def commits_under(
            self,
            branch: str = None,
            ordered_by_date: bool = False
    ) -> List[CommitNode]:
        """Return the list of commits under the given branch. """
        return muller.core.version_control.commits_under(self, branch, ordered_by_date)

    def commits_between(self, id_1: Optional[str] = None, id_2: Optional[str] = None, as_dict=False):
        """ Show the commits history between given ids or branch names. """
        return muller.core.version_control.commits_between(self, id_1, id_2, as_dict=as_dict)

    def get_children_nodes(self, target_commit_id: str = ""):
        """ Obtain the sub-node tree of the target commit ID. """
        return muller.core.version_control.get_children_nodes(self, target_commit_id)

    def log(self, ordered_by_date=False):
        """Displays the details of all the past commits."""
        return muller.core.version_control.log(self, ordered_by_date)

    @invalid_view_op
    @user_permission_check
    def delete_branch(self, name: str) -> None:
        """Delete a branch of the dataset."""
        return muller.core.version_control.delete_branch(self, name)

    @spinner
    @user_permission_check
    def reset(self, force: bool = False):
        """Resets the uncommitted changes present in the branch.
        Note:The uncommitted data is deleted from underlying storage, this is not a reversible operation.
        """
        return muller.core.version_control.reset(self, force)

    def get_tensor_uuids(self, tensor_name, target_commit_id) -> List[int]:
        """获取版本target_commit_id中tensor_name的所有uuid, 按照顺序排列.
            注意，该函数是为了做到能够获取当前版本以外的其它版本中的tensor uuid，而无需check out
        到那个版本而存在，这里直接读取所需版本的uuid数据并返回。
            该函数参考tensor的_sample_id_tensor属性的numpy()的实现逻辑，正确性不能保证，如果出现问题，
        还是以原来的实现为参考，先checkout到目标版本，然后获取相应结果, 可以获得原来的输出结果，最后为了
        不改变dataset原来的状态，再checkout回去.
            该函数理论上应该放在ChunkEngine里面，由于历史原因，先将就一下。
        """
        return muller.core.version_control.get_tensor_uuids(self, tensor_name, target_commit_id)

    def parse_changes(self, diff, tensor_name, last_indexed_commit):
        """Parse the changes of target tensor."""
        return muller.core.version_control.parse_changes(self, diff, tensor_name, last_indexed_commit)

    def get_views(self, commit_id: Optional[str] = None) -> List[ViewEntry]:
        """Returns list of views stored in this Dataset."""
        return muller.core.view.get_views(self, commit_id)

    def get_view(self, view_id: str) -> ViewEntry:
        """Returns the dataset view corresponding to ``id``. """
        return muller.core.view.get_view(self, view_id)

    def save_view(
            self,
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
        """Saves a dataset view as a virtual dataset (VDS)"""

        return muller.core.view.save_view(self, message, path, view_id, optimize, tensors, num_workers,
                                                    scheduler, ignore_errors, **ds_args)

    def load_view(
            self,
            view_id: str,
            optimize: Optional[bool] = False,
            tensors: Optional[List[str]] = None,
            num_workers: int = 0,
            scheduler: str = "threaded",
            progressbar: Optional[bool] = True,
    ):
        """
        Loads the view and returns the :class:`~muller.core.dataset.dataset.Dataset` by id.
        Equivalent to ds.get_view(id).load().
        """
        return muller.core.view.load_view(self, view_id, optimize, tensors, num_workers,
                                                    scheduler, progressbar)

    @user_permission_check
    def delete_view(self, view_id: str):
        """Deletes the view with given view id. """
        return muller.core.view.delete_view(self, view_id)

    def get_view_for_vds(self, inherit_creds=True, creds: Optional[Dict] = None):
        """Returns a view for this VDS. Only works if this Dataset is a virtual dataset. """
        # Sherry: no need to keep this function?
        return muller.core.view.get_view_for_vds(self, inherit_creds, creds)

    def create_index(self, columns, use_uuid: bool = False, batch_size: int = INVERTED_INDEX_BATCH_SIZE):
        """Creates inverted index for target columns. """
        from muller.core.query import create_index
        create_index(self, columns, use_uuid, batch_size)

    def create_index_vectorized(self,
                                tensor_column: str,
                                index_type: str = "fuzzy_match",
                                use_uuid: bool = False,
                                force_create: bool = False,
                                delete_old_index: bool = True,
                                use_cpp: bool = False,
                                **kwargs):
        """Creates inverted index (vectorized) for the target tensor column of the dataset."""
        from muller.core.query import create_index_vectorized
        create_index_vectorized(self, tensor_column, index_type, use_uuid, force_create,
                                                           delete_old_index, use_cpp, **kwargs)

    def optimize_index(self, tensor, use_uuid=None,
                       optimize_mode="create", max_workers=16, delete_old_index=True, use_cpp=False):
        """Optimize inverted index (vectorized) for the target tensor column of the dataset."""
        inverted_index = self.get_inverted_index(tensor, use_uuid, vectorized=True)
        inverted_index.optimize_index(optimize_mode=optimize_mode,
                                      max_workers=max_workers,
                                      delete_old_index=delete_old_index,
                                      use_cpp=use_cpp)

    def reshard_index(self, tensor, old_shard_num: int, new_shard_num: int, max_workers: int = 16, use_uuid=None):
        """Reshard the inverted index (vectorized) for the target tensor column of the dataset."""
        inverted_index = self.get_inverted_index(tensor, use_uuid, vectorized=True)
        inverted_index.reshard_index(old_shard_num=old_shard_num,
                                     new_shard_num=new_shard_num,
                                     max_workers=max_workers)

    def create_hot_shard_index(self, tensor, use_uuid=None, max_workers: int = 16, n=100000):
        """Create a hot shard for the inverted index (vectorized) for the target tensor column of the dataset."""
        inverted_index = self.get_inverted_index(tensor, use_uuid, vectorized=True)
        inverted_index.add_hot_shard(max_workers=max_workers, n=n)

    @index_permission_check
    def get_inverted_index(self, tensor, use_uuid: bool = False, vectorized: bool = False):
        """
        Load inverted index from storage
        注意：如果要使用老版的倒排索引，需要把vectorized改为False
        """
        branch = self.version_state.get("branch", "main")
        optimize = self[tensor].htype == "generic"
        if vectorized:
            from muller.core.query.inverted_index_vectorized import InvertedIndexVectorized
            return InvertedIndexVectorized(self, self.storage, branch, column_name=tensor, use_uuid=use_uuid)

        from muller.core.query.inverted_index_muller import InvertedIndex
        return InvertedIndex(self.storage, tensor, branch, use_uuid, optimize)

    def query(self, tensor_name, query):
        """Query the target tensor column based on inverted index."""
        inverted_index = self.get_inverted_index(tensor_name)
        ids = inverted_index.search(query)
        if inverted_index.use_uuid:
            # map uuids to global idx
            commit_node = self.version_state.get("commit_node")
            commit_id = commit_node.parent.commit_id
            uuids = self.get_tensor_uuids(tensor_name, commit_id)
            index = set()
            for idx, tmp_uuid in enumerate(uuids):
                if str(tmp_uuid) in ids:
                    index.add(idx)
            ids = index
        return ids

    def filter(
            self,
            function: Optional[Union[Callable, str]] = None,
            index_query: Optional[str] = None,
            connector: Optional[str] = "AND",
            offset: Optional[int] = 0,
            limit: Optional[int] = None,
            **kwargs
    ):
        """Filter the dataset with specified function."""
        compute_future = kwargs.get("compute_future", True)

        def function_contents(func):
            try:
                func.__closure__
            except Exception as e:
                func = func.func
                raise Exception from e
            finally:
                closure = tuple(cell.cell_contents for cell in func.__closure__) if func.__closure__ else ()
                tmp_list = [func.__name__, func.__defaults__, func.__kwdefaults__, closure, func.__code__.co_code,
                        func.__code__.co_consts]
                return tmp_list

        function_key = hash(function) if function is None or isinstance(function, str) else hash(
            tuple(function_contents(function)))
        key = str(index_query).strip() + str(function_key) + connector

        if "filter" not in self.storage.upper_cache:
            self.storage.upper_cache["filter"] = {}
        cache_key = key + str(offset)
        if compute_future and cache_key in self.storage.upper_cache["filter"]:
            result = self.storage.upper_cache["filter"].pop(cache_key).result(timeout=None)  # index
            if (len(result.filtered_index) > 0 and
                    result.filtered_index[-1] != len(self) - 1):  # not last index, have next
                self.filter_next(cache_key, function, index_query, connector, offset=result.filtered_index[-1] + 1,
                                 limit=limit)
            return result
        if bool(self.storage.upper_cache["filter"]) and cache_key not in self.storage.upper_cache[
            "filter"]:  # key doesn't match
            self.storage.upper_cache["filter"] = {}  # clear cache

        ids = []
        if index_query is not None:
            def dynamic_function():
                return eval(index_query, {'__builtins__': None}, {'query': self.query_string})
            ids_fuzzy_matching = dynamic_function()
            ids = list(ids_fuzzy_matching)
            ids.sort()
        if offset > 0:
            ids = [i for i in ids if i >= offset]
        if function is None:  # fuzzy matching only
            ids = ids[:limit]
            ret = self[ids]
            ret.filtered_index = ids
            return ret

        ds, ids = self._get_filter_res_from_conditions(function, connector, offset, limit, ids, index_query)

        kwargs["index_query"] = index_query
        kwargs["connector"] = connector
        kwargs["ids"] = ids
        kwargs["key"] = key
        from muller.core.query import filter_dataset, query_dataset
        fn = query_dataset if isinstance(function, str) else filter_dataset
        ret = self._process_filter(fn, ds, function, offset, limit, kwargs)
        return ret


    def filter_next(self, key, function, index_query, connector, offset, limit):
        """Filter the dataset with specified function (in advance and save the results to upper cache)"""
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.filter, function=function, index_query=index_query, connector=connector,
                                     offset=offset, limit=limit)
            self.storage.upper_cache["filter"].update({key: future})

    def aggregate(
            self,
            group_by_tensors: List[str],
            selected_tensors: List[str],
            order_by_tensors: Optional[list] = None,
            aggregate_tensors: Optional[list] = None,
            function: Optional[Callable] = None,
            order_direction: str = 'DESC',
            num_workers: int = 0,
            scheduler: str = "processed",
            progressbar: bool = True,
            method: str = "count",
    ):
        """Conduct aggregation query on the dataset."""
        from muller.core.query import aggregate_dataset
        return aggregate_dataset(
            self,
            function,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
            group_by_tensors=group_by_tensors,
            selected_tensors=selected_tensors,
            order_by_tensors=order_by_tensors,
            aggregate_tensors=aggregate_tensors,
            order_direction=order_direction,
            method=method
        )

    def aggregate_vectorized(
            self,
            group_by_tensors: List[str],
            selected_tensors: List[str],
            order_by_tensors: Optional[list] = None,
            aggregate_tensors: Optional[list] = None,
            order_direction: Optional[str] = 'DESC',
            method: str = 'count'
    ):
        """ A vectorized aggregate function accelerated by the parallel computing supported by numpy. """

        from muller.core.query import aggregate_vectorized_dataset
        return aggregate_vectorized_dataset(
            dataset=self,
            group_by_tensors=group_by_tensors,
            selected_tensors=selected_tensors,
            order_by_tensors=order_by_tensors,
            aggregate_tensors=aggregate_tensors,
            order_direction=order_direction,
            method=method
        )

    def filter_vectorized(
            self,
            condition_list,
            connector_list: Optional[List] = None,
            offset: Optional[int] = 0,
            limit: Optional[int] = None,
            compute_future: Optional[bool] = True,
            use_local_index: bool = True,
            max_workers: Optional[int] = 16,
            show_progress: bool = False,
    ):
        """ A vectorized filtering function accelerated by the parallel computing supported by numpy. """
        from muller.core.query import filter_vectorized_dataset
        return filter_vectorized_dataset(
            self,
            condition_list=condition_list,
            connector_list=connector_list,
            offset=offset,
            limit=limit,
            compute_future=compute_future,
            use_local_index=use_local_index,
            max_workers=max_workers,
            show_progress=show_progress,
        )

    def create_uuid_index(self):
        """Create uuid and index pair and stored in the disk. """
        try:
            current_id = self.version_state['commit_id']
        except KeyError as e:
            raise KeyError from e
        if current_id != FIRST_COMMIT_ID:
            raise ValueError
        uuids = self.get_tensor_uuids(DATASET_UUID_NAME, current_id)
        divide_to_shard(path=os.path.join(self.path, DATASET_UUID_NAME), uuids=uuids)

    def load_uuid_index(self):
        """Load all uuid indexes from shards. """
        try:
            current_id = self.version_state['commit_id']
        except KeyError as e:
            raise KeyError from e
        if current_id != FIRST_COMMIT_ID:
            raise ValueError
        return load_all_shards(path=os.path.join(self.path, DATASET_UUID_NAME))

    @invalid_view_op
    @user_permission_check
    def create_vector_index(self, tensor_name: str, index_name: str, index_type: str = 'FLAT', metric: str = 'l2',
                            **kwargs: Union[int, float, str]):
        """Create index for tensor in vector type. """
        from muller.core.query import create_vector_index
        create_vector_index(self, tensor_name, index_name, index_type, metric, **kwargs)

    def update_vector_index(self, tensor_name: str, index_name: str) -> None:
        """Update vector index after add some samples into dataset and commit. """
        from muller.core.query import update_vector_index
        update_vector_index(self, tensor_name, index_name)

    def vector_search(self, query_vector: Union[NDArray, Tensor], tensor_name: str, index_name: str, **kwargs):
        """KNN Search on vector index. """
        from muller.core.query import vector_search
        return vector_search(self, query_vector, tensor_name, index_name, **kwargs)

    def load_vector_index(self, tensor_name: str, index_name: str, **kwargs):
        """Load vector index into memory when index is unloaded. """
        from muller.core.query import load_vector_index
        load_vector_index(self, tensor_name, index_name, **kwargs)

    def unload_vector_index(self, tensor_name: str, index_name: str):
        """ Unload vector index from memory when index is loaded. """
        from muller.core.query import unload_vector_index
        unload_vector_index(self, tensor_name, index_name)

    def drop_vector_index(self, tensor_name: str, index_name: str):
        """ Drop vector index permanently. """
        from muller.core.query import drop_vector_index
        drop_vector_index(self, tensor_name, index_name)

    def summary(self, force: bool = False):
        """Print out a summarization of the schema and statistic information of the dataset."""
        from muller.core.dataset.statistics.summary import summary_dataset
        if (
                not self.index.is_trivial()
                and self.max_len > VIEW_SUMMARY_SAFE_LIMIT
                and not force
        ):
            raise SummaryLimit(self.max_len, VIEW_SUMMARY_SAFE_LIMIT)
        pretty_print = summary_dataset(self)
        print(pretty_print)

    def to_dataframe(self, tensor_list: Optional[List[str]] = None,
                     index_list: Optional[List] = None,
                     force: bool = False):
        """ Returns a pandas dataframe of the dataset.
        Example:

            >>> ds.to_dataframe()
            >>> ds.to_dataframe(index_list=[-1, -2])
            >>> ds.to_dataframe(tensor_list=["categories"], index_list=[1, 2, 4])

        Args:
            tensor_list (List of str, Optional) - The tensor columns selected to be exported as pandas dataframe.
                        If not provided, we will export all the tensor columns.
            index_list (List of int, Optional) - The indices of the rows selected to be exported as pandas dataframe.
                        If not provided, we will export all the row.
            force (bool, Optional) - Dataset with more than TO_DATAFRAME_SAFE_LIMIT samples might take a long time to
                        export. If force = True, the dataset will be exported regardless.
                        An error will be raised otherwise.

        Raises:
            InvalidTensorList: If ``tensor_list`` contains tensors that are not in the current columns.
            ToDataFrameLimit: If the length of ``index_list`` exceeds the TO_DATAFRAME_SAFE_LIMIT.
        """
        if tensor_list and (len(tensor_list) > len(self.tensors) or
                            not all(isinstance(x, str) and x in self.tensors for x in tensor_list)): # 确认目标列打对了
            raise InvalidTensorList(tensor_list)
        max_num = -1
        if index_list and len(index_list) > TO_DATAFRAME_SAFE_LIMIT and not force:
            max_num = len(index_list)
        elif self.max_len > TO_DATAFRAME_SAFE_LIMIT and not force:
            max_num = self.max_len
        if max_num != -1:
            raise ToDataFrameLimit(max_num, TO_DATAFRAME_SAFE_LIMIT)

        return muller.core.dataset.to_dataframe(self, tensor_list, index_list)

    def to_arrow(self):
        """Returns an arrow object of the dataset."""
        return muller.core.dataset.MULLERArrowDataset(self)

    def write_to_parquet(self, path, columns=None):
        """Returns an arrow of the dataset."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        arrow_dataset = self.to_arrow()
        arrow_table = arrow_dataset.to_table(columns)
        writer = pa.BufferOutputStream()
        pq.write_table(arrow_table, writer)
        self.storage[path] = bytes(writer.getvalue())

    def statistics(self):
        """Get statistics info of dataset. Load from dataset_meta.json first, if empty, calculate and then save it.

        Example:
            >>> ds = muller.load("path_to_dataset")
            >>> ds.statistics()
        """
        from muller.core.dataset.statistics.statistics import get_statistics
        if self.has_head_changes:
            warnings.warn(
                "There are uncommitted changes, showing statistics from last committed version, try again after commit."
            )

        stats = load_statistics(self)
        if not stats:
            stats = get_statistics(self)
            save_statistics(self, stats)
        print(json.dumps(stats))

    def to_json(
            self,
            path,
            tensors: Optional[List[str]] = None,
            num_workers: Optional[int] = 1,
    ):
        """Write MULLER data by row to a jsonl or json file.

        Example:
            >>> ds.to_json("test.jsonl")

        Args:
            path (str): The jsonl or json file name to save MULLER data.
            tensors (List of str, Optional): The tensor columns selected to be exported to the jsonl or json file.
            num_workers (int, Optional): The number of workers that can be used to dump to the path.
        """
        if not (path.endswith("json") or path.endswith("jsonl")):
            raise InvalidJsonFileName(path)
        if num_workers <= 0:
            raise InvalidNumWorkers(num_workers)
        muller.core.dataset.to_json(self, path, tensors, num_workers)

    def to_mindrecord(
            self,
            file_name,
            shard_num: int = 1,
            batch_size: int = 100000,
            overwrite: bool = False,
            scheduler: str = "threaded",
    ):
        """Create MindRecord and save as file_name.

        Example:
            >>> ds.to_mindrecord("ds.mindrecord", shard_num=1, batch_size=1000, overwrite=True)

        Args:
            file_name (str): The filename to save MindRecord.
            shard_num (int, Optional): The number of MindRecord files to generate.
            batch_size (int, Optional): The batch size to get numpy data from MULLER.
            overwrite (bool, Optional): Overwrite if same file name exists.
            scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded',
                'processed' and 'distributed'. Defaults to 'threaded'.
        """
        muller.core.dataset.export_data.to_mindrecord.create_mindrecord(self, file_name, shard_num, batch_size,
                                                                       overwrite, scheduler)

    def size_approx(self):
        """Estimates the size in bytes of the dataset. """
        tensors = self.version_state["full_tensors"].values()
        chunk_engines = [tensor.chunk_engine for tensor in tensors]
        size = sum(c.num_chunks * c.min_chunk_size for c in chunk_engines)
        return size

    def get_tensors(self, include_hidden: bool = True, include_disabled=True) -> Dict[str, Tensor]:
        """All tensors belonging to this group, including those within sub groups. Always returns the sliced tensors."""
        version_state = self.version_state
        index = self.index
        all_tensors = self.all_tensors_filtered(include_hidden, include_disabled)
        return {
            t: version_state["full_tensors"][
                version_state["tensor_names"][t]
            ][index]
            for t in all_tensors
        }

    def populate_meta(self, address: Optional[str] = None, verbose=True):
        """Populates the meta information for the dataset."""
        if address is None:
            commit_id = self._get_commit_id_for_address("main", self.version_state)
        else:
            commit_id = self._get_commit_id_for_address(address, self.version_state)

        if dataset_exists(self.storage, commit_id):
            load_meta(self)
        elif not self.storage.empty():
            # dataset does not exist, but the path was not empty
            raise PathNotEmptyException
        else:
            if self.read_only:
                # cannot create a new dataset when in read_only mode.
                raise CouldNotCreateNewDatasetException(self.path)
            try:
                commit_id = self.version_state["commit_id"]
            except KeyError as e:
                raise VersionControlError from e

            meta = DatasetMeta()
            meta.set_dataset_creator(obtain_current_user())
            key = get_dataset_meta_key(commit_id)
            self.version_state["meta"] = meta
            self.storage.register_muller_object(key, meta)

            dataset_diff = DatasetDiff()
            key = get_dataset_diff_key(commit_id)
            self.storage.register_muller_object(key, dataset_diff)

            self.flush()

    def all_tensors_filtered(self, include_hidden: bool = True, include_disabled=True) -> List[str]:
        """Names of all tensors belonging to this group, including those within sub groups"""
        hidden_tensors = self.meta.hidden_tensors
        tensor_names = self.version_state["tensor_names"]
        enabled_tensors = self.enabled_tensors
        final_results = []
        for t in tensor_names:
            if include_hidden or tensor_names[t] not in hidden_tensors:
                if include_disabled or enabled_tensors is None or t in enabled_tensors:
                    final_results.append(t)
        return final_results

    def get_sample_indices(self, maxlen: int):
        """Get sample indices"""
        vds_index = self.get_tensors(include_hidden=True).get(VDS_INDEX)
        if vds_index:
            return vds_index.numpy().reshape(-1).tolist()
        return self.index.values[0].indices(maxlen)

    def set_read_only(self, value: bool, err: bool):
        """Set the read only variable. """
        storage = self.storage
        self.__dict__["_read_only"] = value

        if value:
            storage.enable_readonly()
            if isinstance(storage, LRUCache) and storage.next_storage is not None:
                storage.next_storage.enable_readonly()
            unlock_dataset(self)  # Sherry: not support yet
        else:
            try:
                locked = self.lock(err=err)
                if locked:
                    self.storage.disable_readonly()
                    if (
                            isinstance(storage, LRUCache)
                            and storage.next_storage is not None
                    ):
                        storage.next_storage.disable_readonly()
                else:
                    self.__dict__["_read_only"] = True
            except LockedException as e:
                self.__dict__["_read_only"] = True
                if err:
                    raise e

    def lock(self, err=False, verbose=True):
        """Lock the dataset. """
        if not self.is_head_node or not self._locking_enabled:
            return True
        storage = self.base_storage
        if storage.read_only and not self._locked_out:
            if err:
                raise ReadOnlyModeError()
            return False

        if isinstance(storage, tuple(_LOCKABLE_STORAGES)) and (
                not self.read_only or self._locked_out
        ):
            if not muller.constants.LOCK_LOCAL_DATASETS and isinstance(
                    storage, LocalProvider
            ):
                return True
            try:
                # temporarily disable read only on base storage, to try to acquire lock,
                # if exception, it will be again made readonly
                storage.disable_readonly()
                lock_dataset(
                    self,
                    lock_lost_callback=self._lock_lost_handler,
                )
            except LockedException as e:
                self.set_read_only(True, False)
                self.__dict__["_locked_out"] = True
                if err:
                    raise e
                return False
        return True

    def tensor_diff(self, id_1, id_2, tensors: List[str] = None):
        """
        displays the differences between commits (in the same branch) for certain tensor
        """
        version_state, storage = self.version_state, self.storage
        res = get_changes_and_messages(version_state, storage, id_1, id_2)
        tensor_changes = res[3]  # The changes between id_2 and common ancestor on tensor (res[2] is always empty)
        if tensor_changes is not None and len(tensor_changes) > 0:
            for tensor_change_ver in tensor_changes:
                _ = self.generate_add_update_value(tensor_change_ver, 0, None, False, tensors)

        changes = {"tensor": (tensor_changes,)}
        return changes

    def sub_ds(
            self,
            path,
            empty=False,
            memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
            local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
            read_only=None,
            verbose=True,
    ):
        """Loads a nested dataset. Internal.

        Args:
            path (str): Path to sub directory.
            empty (bool): If ``True``, all contents of the sub directory is cleared before initializing the sub dataset.
            memory_cache_size (int): Memory cache size for the sub dataset.
            local_cache_size (int): Local storage cache size for the sub dataset.
            read_only (bool): Loads the sub dataset in read only mode if ``True``. Default ``False``.
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.

        Returns:
            Sub dataset

        Note:
            Virtual datasets are returned as such, they are not converted to views.
        """
        sub_storage = self.base_storage.subdir(path, read_only=read_only)

        if empty:
            sub_storage.clear()

        path = sub_storage
        cls = muller.core.dataset.Dataset

        ret = cls(
            generate_chain(
                sub_storage,
                memory_cache_size * MB,
                local_cache_size * MB,
            ),
            path=path,
            read_only=read_only,
            verbose=verbose,
        )
        ret.parent_dataset = self
        return ret

    def _reload_version_state(self):
        version_state = self.version_state
        # share version state if at HEAD
        if (
                not self._view_use_parent_commit
                and self._view_base
                and version_state["commit_node"].is_head_node
        ):
            return version_state  # Sherry: don't need currently
        vs_copy = {}
        vs_copy["branch"] = version_state.get("branch", None)
        # share branch_commit_map and commit_node_map
        vs_copy["branch_commit_map"] = version_state.get("branch_commit_map", None)
        vs_copy["commit_node_map"] = version_state.get("commit_node_map", None)
        vs_copy["branch_info"] = version_state.get("branch_info", None)
        commit_node = version_state.get("commit_node", None)
        if self._view_use_parent_commit:
            vs_copy["commit_node"] = commit_node.parent
        else:
            vs_copy["commit_node"] = commit_node
        vs_copy["commit_id"] = vs_copy.get("commit_node", None).commit_id
        vs_copy["tensor_names"] = version_state.get("tensor_names", None).copy()
        vs_copy["meta"] = DatasetMeta()
        vs_copy["meta"].__setstate__(version_state.get("meta", None).__getstate__())
        self.version_state = vs_copy
        vs_copy["full_tensors"] = {
            key: Tensor(key, self)
            for key in version_state.get("full_tensors", None)
        }
        self._view_base = None
        return version_state

    def _lock_lost_handler(self):
        """This is called when lock is acquired but lost later on due to slow update."""
        self.read_only = True
        self._locked_out = True

    def _set_derived_attributes(self, verbose: bool = True, address: Optional[str] = None):
        """Sets derived attributes during init and unpickling."""
        if self.is_first_load:
            self.storage.autoflush = True
            self._load_version_info(address)
            self.set_read_only(
                self._read_only, err=self._read_only_error
            )
            self.populate_meta(  # Sherry: OBS output here
                address, verbose
            )
            if self.index.is_trivial():
                self.index = Index.from_json(self.meta.default_index)
        elif not self._read_only:
            _ = self.lock(verbose=verbose)  # for ref counting

        if not self.is_first_load:
            _ = self._reload_version_state()

    def _preprocessing_meta(self, address: str):
        try:
            self._set_derived_attributes(address=address)
        except ReadOnlyModeError as e:
            raise ReadOnlyModeError(
                "This dataset cannot be open for writing as you don't have permissions. "
                "Try loading the dataset with `read_only=True."
            ) from e
        except LockedException as e:
            raise LockedException(
                "This dataset cannot be open for writing as it is locked by another machine. "
                "Try loading the dataset with `read_only=True`."
            ) from e

    def _initial_temp_tensors(self, temp_tensors):
        final_tensors = temp_tensors
        if not self.read_only:
            for _tensor in temp_tensors:
                self._delete_tensor(_tensor, large_ok=True)
            final_tensors = []
        return final_tensors

    def _load_version_info(self, address=None):
        """Loads data from version_control_file otherwise assume it doesn't exist and load all empty"""
        if self.version_state:
            return

        if address is None:
            address = "main"

        version_state = {}
        try:
            try:
                version_info = load_version_info(self.storage)
            except Exception as e:
                version_info = rebuild_version_info(self.storage)
                if version_info is None:
                    raise e
            version_state["branch_commit_map"] = version_info["branch_commit_map"]
            version_state["commit_node_map"] = version_info["commit_node_map"]
            version_state["branch_info"] = version_info["branch_info"]
            commit_id = self._get_commit_id_for_address(address, version_state)

            version_state["commit_id"] = commit_id
            version_state["commit_node"] = version_state["commit_node_map"][commit_id]
            version_state["branch"] = version_state["commit_node"].branch
        except Exception as e:
            if isinstance(e, CheckoutError):
                raise e from None
            if address != "main":
                raise CheckoutError(
                    f"Address {address} not found. Ensure the commit id / branch name is correct."
                ) from e
            branch = "main"
            version_state["branch"] = branch
            version_state["branch_commit_map"] = {}
            version_state["commit_node_map"] = {}
            version_state["branch_info"] = {}
            commit_id = FIRST_COMMIT_ID
            commit_node = CommitNode(branch, commit_id)
            commit_node.commit_time = datetime.utcnow()
            version_state["commit_id"] = commit_id
            version_state["commit_node"] = commit_node
            version_state["branch_commit_map"][branch] = commit_id
            version_state["commit_node_map"][commit_id] = commit_node
            version_state["branch_info"][branch] = {"based_on": None, "create_time": commit_node.commit_time}
        version_state["full_tensors"] = {}
        version_state["tensor_names"] = {}
        self.__dict__["version_state"] = version_state

    def _return_tensor(self, item, is_iteration):
        fullpath = item
        enabled_tensors = self.enabled_tensors
        if enabled_tensors is None or fullpath in enabled_tensors:
            tensor = self._get_tensor_from_dataset(fullpath)
            if tensor is not None:
                index = self.index
                if index.is_trivial() and is_iteration == tensor.is_iteration:
                    return tensor
                return tensor.__getitem__(index, is_iteration=is_iteration)
        raise TensorDoesNotExistError(item)

    def _get_tensor_from_dataset(self, name: str) -> Optional[Tensor]:
        """Gets a tensor from the dataset. Accesses storage only for the first call. """
        key = self.version_state["tensor_names"].get(name)
        return self.version_state["full_tensors"].get(key)

    def _flush_vc_info(self):
        if self.vc_info_updated:
            save_version_info(self.version_state, self.storage)
            for node in self.version_state["commit_node_map"].values():
                if node.info_updated:
                    save_commit_info(node, self.storage)
            self.vc_info_updated = False

    def _load_uuids(self):
        # load uuids
        self.check_uuid()
        tensor_names = self.version_state.get("tensor_names", [])
        if DATASET_UUID_NAME in tensor_names:
            return True
        return False

    def _get_filter_res_from_conditions(self, function, connector, offset, limit, ids, index_query):

        if connector == "AND":
            ids = [i - offset for i in ids]
            ds = self[offset:]
            ds = ds[ids] if index_query is not None else ds
        elif connector == "OR":
            ids = ids[:limit]
            ds = self[offset:]
        else:
            raise Exception(f"Unsupported connector {connector}")
        return ds, ids

    def _process_filter(self, fn, ds, function, offset, limit, kwargs):

        index_query = kwargs.get("index_query", None)
        connector = kwargs.get("connector", None)
        ids = kwargs.get("ids", None)
        key = kwargs.get("key", None)

        ret = fn(
            ds,
            function,
            num_workers=kwargs.get("num_workers", 0),
            scheduler=kwargs.get("scheduler", "threaded"),
            progressbar=kwargs.get("progressbar", False),
            save_result=kwargs.get("save_result", False),
            result_path=kwargs.get("result_path", None),
            result_ds_args=kwargs.get("result_ds_args", None),
            offset=offset,
            limit=limit,
        )

        if index_query is None and offset > 0:
            ret.filtered_index = [i + offset for i in ret.filtered_index]
        if connector == "AND" and index_query is not None:
            index_map = [ids[local_index] for local_index in ret.filtered_index][:limit]
            ret.filtered_index = index_map
            if offset > 0:
                ret.filtered_index = [i + offset for i in ret.filtered_index]
        if connector == "OR":
            if offset > 0:
                ret.filtered_index = [i + offset for i in ret.filtered_index]
            merged_ids = list(heapq.merge(ret.filtered_index, ids))[:limit]
            ret = self[merged_ids]
            ret.filtered_index = merged_ids

        if limit and len(ret.filtered_index) > 0 and ret.filtered_index[-1] != len(self) - 1:
            if kwargs.get("compute_future", True):
                key = key + str(ret.filtered_index[-1] + 1)  # the start index of next filter computation
                self.filter_next(key, function, index_query, connector, offset=ret.filtered_index[-1] + 1, limit=limit)
        return ret
    