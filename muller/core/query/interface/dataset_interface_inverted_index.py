# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import json
import os
import warnings

from muller.constants import INVERTED_INDEX_BATCH_SIZE
from muller.util.exceptions import TensorDoesNotExistError, UnsupportedInvertedIndexError, \
    MultiProcessUnsupportedError


def create_index(ds, columns, use_uuid: bool = False, batch_size: int = INVERTED_INDEX_BATCH_SIZE):
    """Creates inverted index for target columns

    Example:

        >>> ds = muller.load("path_to_dataset")
        >>> ds.create_index(["tensor1", "tensor2"])

    Args:
        ds (Dataset): The dataset to be created index on.
        columns (List): List of tensors to create index.
        use_uuid (bool, Optional): use uuid to create index if true, otherwise use global index.
        batch_size (int, Optional): Batch size to split index files, default to be the INVERTED_INDEX_BATCH_SIZE
            provided in constants.

    Raises:
        TensorDoesNotExistError: If ``columns`` contains tensor that doesn't exist.
        UnsupportedInvertedIndexError: It tensor htype is not "text" or "list" of string
    """
    _validate_tensor_before_creating_index(ds, columns)

    if ds.has_head_changes:
        warnings.warn(
            "There are uncommitted changes, try again after commit."
        )
    else:
        branch = ds.version_state["branch"]
        meta_path = os.path.join("inverted_index_dir", branch, "meta.json")
        try:
            meta_json = json.loads(ds.storage[meta_path].decode('utf-8'))
        except KeyError:
            meta_json = {}

        for tensor in columns:
            if tensor in meta_json and meta_json[tensor].get("commit_id", None) != ds.commit_id:
                # update index
                tensor_changes = ds.tensor_diff(meta_json[tensor].get("commit_id", None),
                                                ds.commit_id,
                                                [tensor])
                tensor_diff = ds.parse_changes(tensor_changes, tensor, meta_json[tensor].get("commit_id", None))
                # 只要有update/delete，都重建，无论use uuid or not
                if len(tensor_diff["deleted"]) > 0 or len(tensor_diff["updated"]) > 0:
                    _create_tensor_index(ds, tensor, use_uuid, batch_size)
                else:
                    # 只有append的情况，update index
                    inverted_index = ds.get_inverted_index(tensor, use_uuid)
                    if not use_uuid:
                        # convert uuid -> global index
                        uuid_list = ds.get_tensor_uuids(tensor, ds.commit_id)
                        added = {}
                        for add_id, add_doc in tensor_diff['added'].items():
                            added[uuid_list.index(int(add_id))] = add_doc
                        tensor_diff['added'] = added
                    inverted_index.update_index(tensor_diff, batch_size)
            else:
                # create index
                _create_tensor_index(ds, tensor, use_uuid, batch_size)
            meta_json[tensor] = {"commit_id": ds.commit_id}

        ds.storage[meta_path] = json.dumps(meta_json).encode('utf-8')


def _validate_tensor_before_creating_index(ds, columns):
    for tensor_name in columns:
        if tensor_name not in ds.tensors:
            raise TensorDoesNotExistError(tensor_name)
        htype = ds.tensors[tensor_name].htype
        dtype = ds.tensors[tensor_name].dtype

        if htype not in ("text", "class_label", "list") and dtype not in ("<U0", "int64", "float64"):
            raise UnsupportedInvertedIndexError(ds.tensors[tensor_name].base_htype,
                                                ds.tensors[tensor_name].dtype)


def _create_tensor_index(ds, tensor, use_uuid, batch_size):
    inverted_index = ds.get_inverted_index(tensor, use_uuid)
    tensor_object = ds.tensors[tensor]
    try:
        shm, data = tensor_object.numpy_multi_process(fetch_chunks=True, aslist=True)
    except MultiProcessUnsupportedError:
        shm = None
        data = tensor_object.numpy_full(fetch_chunks=True, aslist=True)
    doc_dicts = {}
    uuids = []
    if inverted_index.use_uuid:
        uuids = ds.get_tensor_uuids(tensor, ds.commit_id)

    for i, line in enumerate(data):
        if tensor_object.htype == 'list':
            temp = ""
            for element in line:
                temp = temp + element
            line = temp
        else:
            line = line.item()
        if inverted_index.use_uuid:
            doc_dicts.update({str(uuids[i]): line})
        else:
            doc_dicts.update({i: line})
    tensor_object.chunk_engine.cached_data = None
    inverted_index.create_index(doc_dicts, batch_size, save_to_next_storage=True)
    # free shared memory to avoid memory leak
    if shm is not None:
        shm.close()
        shm.unlink()
