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

from muller.util.exceptions import (UnsupportedInvertedIndexError,
                                   TensorDoesNotExistError, UnsupportedMethod, UpdateIndexFailError)


def create_index_vectorized(ds,
                            tensor_column: str,
                            index_type: str = "fuzzy_match",
                            use_uuid: bool = False,
                            force_create: bool = False,
                            delete_old_index: bool = True,
                            use_cpp: bool = False,
                            **kwargs):
    """Creates inverted index (vectorized) for the target tensor column of the dataset."""

    _validate_columns_before_creating_index_vec(ds, tensor_column)

    if ds.has_head_changes:
        warnings.warn(
            "There are uncommitted changes, try again after commit."
        )
    else:
        branch = ds.version_state["branch"]
        meta_path = os.path.join("inverted_index_dir_vec", branch, "meta.json")
        try:
            meta_json = json.loads(ds.storage[meta_path].decode('utf-8'))
        except KeyError:
            meta_json = {}
        # 如果该列索引已存在，则更新索引（append-only）
        create, added_index_list = _decide_update_or_create_index(ds, meta_json, tensor_column, force_create)

        if index_type == "exact_match" and use_cpp:
            raise UnsupportedMethod("Not support create index type of 'exact_match' in cpp now, "
                                    "please set use_cpp = False.")

        # 开始进入索引创建流程，根据create的值来判断是重建全部索引还是更新索引（append-only）
        inverted_index = ds.get_inverted_index(tensor_column, use_uuid, vectorized=True)
        if inverted_index.use_uuid:
            uuids = ds.get_tensor_uuids(tensor_column, ds.commit_id)
        else:
            uuids = None

        if create:
            # 直接创建即可！
            _create_new_index(ds,
                              inverted_index,
                              index_type,
                              uuids,
                              force_create,
                              use_cpp,
                              tensor_column,
                              delete_old_index,
                              kwargs,
                              )
        else:
            # 更新索引
            _update_old_index(ds,
                             inverted_index,
                             index_type,
                             uuids,
                             use_cpp,
                             tensor_column,
                             delete_old_index,
                             added_index_list,
                             kwargs,
                             )


def _create_new_index(ds,
                      inverted_index,
                      index_type,
                      uuids,
                      force_create,
                      use_cpp,
                      tensor_column,
                      delete_old_index,
                      kwargs,
                      ):
    tokenizer = kwargs.get("tokenizer", "jieba")
    cut_all = kwargs.get("cut_all", False)
    stop_words_list = kwargs.get("stop_words_list", None)
    compulsory_words = kwargs.get("compulsory_words", None)
    case_sensitive = kwargs.get("case_sensitive", False)

    success = inverted_index.create_index(index_type=index_type,
                                          num_of_shards=kwargs.get("num_of_shards", 1),
                                          uuids=uuids,
                                          max_workers=kwargs.get("max_workers", 16),
                                          num_of_batches=kwargs.get("num_of_batches", 1),
                                          tokenizer=tokenizer,
                                          cut_all=cut_all,
                                          stop_words_list=stop_words_list,
                                          compulsory_words=compulsory_words,
                                          case_sensitive=case_sensitive,
                                          force_create=force_create,
                                          use_cpp=use_cpp
                                          )
    if success:
        branch = ds.version_state["branch"]
        meta_path = os.path.join("inverted_index_dir_vec", branch, "meta.json")
        try:
            meta_json = json.loads(ds.storage[meta_path].decode('utf-8'))
        except KeyError:
            meta_json = {}

        meta_json[tensor_column] = {"commit_id": ds.commit_id,
                                    "index_type": index_type,
                                    "num_of_shards": kwargs.get("num_of_shards", 1),
                                    "use_uuid": bool(uuids),
                                    "tokenizer": kwargs.get("tokenizer", "jieba"),
                                    "cut_all": kwargs.get("cut_all", False),
                                    "stop_words_list": kwargs.get("stop_words_list", None),
                                    "compulsory_words": kwargs.get("compulsory_words", None),
                                    "case_sensitive": kwargs.get("case_sensitive", False),
                                    "optimized": False,
                                    "use_cpp": use_cpp
                                    }
        ds.storage[meta_path] = json.dumps(meta_json).encode('utf-8')
        ds.storage.flush()

        inverted_index.optimize_index(optimize_mode="create",
                                      max_workers=kwargs.get("max_workers", 16),
                                      delete_old_index=delete_old_index,
                                      use_cpp=use_cpp)
        meta_json[tensor_column]["optimized"] = True
        ds.storage[meta_path] = json.dumps(meta_json).encode('utf-8')
        ds.storage.flush()
    else:
        warnings.warn("Create index fails.")


def _update_old_index(ds,
                     inverted_index,
                     index_type,
                     uuids,
                     use_cpp,
                     tensor_column,
                     delete_old_index,
                     added_index_list,
                     kwargs, ):
    if not added_index_list:
        success = False
    else:
        tokenizer_params = {
            'tokenizer': kwargs.get("tokenizer", "jieba"),
            'cut_all': kwargs.get("cut_all", False),
            'stop_words_list': kwargs.get("stop_words_list", None),
            'compulsory_words': kwargs.get("compulsory_words", None),
            'case_sensitive': kwargs.get("case_sensitive", False)
        }
        success = inverted_index.update_index(start_index=added_index_list[0],
                                              end_index=added_index_list[1],
                                              index_type=index_type,
                                              num_of_shards=kwargs.get("num_of_shards", 1),
                                              uuids=uuids,
                                              max_workers=kwargs.get("max_workers", 16),
                                              num_of_batches=kwargs.get("num_of_batches", 1),
                                              tokenizer_params=tokenizer_params,
                                              use_cpp=use_cpp
                                              )
    if success:
        branch = ds.version_state["branch"]
        meta_path = os.path.join("inverted_index_dir_vec", branch, "meta.json")
        try:
            meta_json = json.loads(ds.storage[meta_path].decode('utf-8'))
        except KeyError:
            meta_json = {}
        meta_json[tensor_column] = {"commit_id": ds.commit_id,
                                    "index_type": index_type,
                                    "num_of_shards": kwargs.get("num_of_shards", 1),
                                    "use_uuid": bool(uuids),
                                    "tokenizer": kwargs.get("tokenizer", "jieba"),
                                    "cut_all": kwargs.get("cut_all", False),
                                    "stop_words_list": kwargs.get("stop_words_list", None),
                                    "compulsory_words": kwargs.get("compulsory_words", None),
                                    "case_sensitive": kwargs.get("case_sensitive", False),
                                    "optimized": False,
                                    "use_cpp": use_cpp
                                    }
        ds.storage[meta_path] = json.dumps(meta_json).encode('utf-8')
        ds.storage.flush()

        inverted_index.optimize_index(optimize_mode="update",
                                      max_workers=kwargs.get("max_workers", 16),
                                      delete_old_index=delete_old_index,
                                      use_cpp=use_cpp)
        meta_json[tensor_column]["optimized"] = True
        ds.storage[meta_path] = json.dumps(meta_json).encode('utf-8')
        ds.storage.flush()
    else:
        raise UpdateIndexFailError("Failed to create/update index "
                                   "(attempted to recreate the same column index without changes)")


def _validate_columns_before_creating_index_vec(ds, tensor_column):
    if tensor_column not in ds.tensors:
        raise TensorDoesNotExistError(tensor_column)
    htype = ds.tensors[tensor_column].htype
    dtype = ds.tensors[tensor_column].dtype
    is_valid_type = (htype == "text" or htype == "class_label" or dtype == "int64" or dtype == "float64")
    if not is_valid_type:
        raise UnsupportedInvertedIndexError(ds.tensors[tensor_column].base_htype,
                                            ds.tensors[tensor_column].dtype)


def _decide_update_or_create_index(ds, meta_json, tensor_column, force_create):
    create = False
    added_index_list = None
    if not meta_json:
        create = True
    elif tensor_column not in meta_json:
        create = True
    elif tensor_column in meta_json and meta_json[tensor_column].get("commit_id", None) != ds.commit_id:
        tensor_changes = ds.tensor_diff(meta_json[tensor_column].get("commit_id", None),
                                        ds.commit_id,
                                        [tensor_column])

        def _parse_changes_with_global_indexes(diff, tensor_name):
            temp_tensor_changes = diff["tensor"]
            temp_create = False
            id_list = []
            for tensor_change in temp_tensor_changes:
                change_stack = []
                for change in tensor_change:  # every commit
                    change_stack.append(change)
                while change_stack:
                    change = change_stack.pop()
                    if change[tensor_name]["data_deleted"]:
                        return True, []
                    if change[tensor_name]["data_updated"]:
                        return True, []
                    id_list = change[tensor_name]["data_added"]
            return temp_create, id_list

        create, added_index_list = _parse_changes_with_global_indexes(tensor_changes, tensor_column)

    elif force_create:
        create = True
    return create, added_index_list
    