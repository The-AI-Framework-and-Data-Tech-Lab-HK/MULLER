# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/version_control.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import hashlib
import json
import logging
import random
import time
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, List

from muller.client.log import logger
from muller.constants import FIRST_COMMIT_ID, WRITE_TILES_INDEX
from muller.core import lock
from muller.core.lock import Lock, PersistentLock
from muller.core.meta.dataset_meta import DatasetMeta
from muller.core.storage.muller_memory_object import MULLERMemoryObject
from muller.core.storage.lru_cache import LRUCache
from muller.core.version_control.commit_chunk_map import CommitChunkMap
from muller.core.version_control.commit_node import CommitNode
from muller.core.version_control.dataset_diff import DatasetDiff
from muller.util.authorization import obtain_current_user
from muller.util.exceptions import (CheckoutError,
                                   CommitError,
                                   DatasetCorruptError, VersionControlError)
from muller.util.keys import (get_version_control_info_key,
                             get_version_control_info_lock_key,
                             get_chunk_key,
                             get_commit_info_key,
                             get_dataset_meta_key,
                             get_tensor_meta_key,
                             get_chunk_id_encoder_key,
                             get_tensor_tile_encoder_key,
                             get_tensor_commit_chunk_map_key,
                             get_dataset_diff_key,
                             get_tensor_commit_diff_key)
from muller.util.remove_cache import get_base_storage

branch_commit_map_rebuild: Dict[str, str] = {}
commits_rebuild: Dict[str, Dict] = {}
branch_info_rebuild: Dict[str, Dict] = {}
all_src_keys = []


def load_version_info(storage: LRUCache) -> Dict:
    """Load the version info."""
    try:
        return _version_info_from_json(
            json.loads(storage["version_control_info.json"].decode("utf-8"))
        )
    except Exception as e:
        raise e


def _version_info_from_json(info):
    commits, branch_commit_map, branch_info = info["commits"], info["branches"], info["branches_info"]
    commit_node_map = {}
    stack = ["firstdbf9474d461a19e9333c2fd19b46115348f"]
    while stack:
        commit_id = stack.pop()
        commit_data = commits[commit_id]
        branch_commit_map[commit_data["branch"]]
        node = CommitNode(branch=commit_data["branch"], commit_id=commit_id)
        node.commit_message = commit_data["commit_message"]
        commit_time = commit_data["commit_time"]
        node.commit_time = (
            None if commit_time is None else datetime.fromtimestamp(commit_time)
        )
        node.commit_user_name = commit_data["commit_user_name"]
        node.is_checkpoint = commit_data.get("is_checkpoint", False)
        node.total_samples_processed = commit_data.get("total_samples_processed", 0)
        node.merge_parent = commit_data.get("merge_parent", "")
        node.checkout_node = commit_data.get("checkout_node", False)
        if node.checkout_node or node.commit_id == FIRST_COMMIT_ID:
            create_time = branch_info[commit_data["branch"]]["create_time"]
            branch_info[commit_data["branch"]]["create_time"] = (
                datetime.fromtimestamp(create_time) if create_time else None)
        parent = commit_data["parent"]
        if parent:
            commit_node_map.get(parent, None).add_child(node)
        commit_node_map[commit_id] = node
        stack += commit_data["children"]
    return {
        "commit_node_map": commit_node_map,
        "branch_commit_map": branch_commit_map,
        "branch_info": branch_info
    }


def load_statistics(dataset):
    """Loads the statistics info in meta for the version state."""
    commit_id = dataset.version_state["commit_id"]
    meta_key = get_dataset_meta_key(commit_id)
    stats = dataset.storage.get_muller_object(meta_key, DatasetMeta).statistics
    return stats


def save_statistics(dataset, stats):
    """Saves the statistics info for the version state."""
    initial_autoflush = dataset.storage.autoflush
    dataset.storage.autoflush = False

    commit_id = dataset.version_state["commit_id"]
    meta_key = get_dataset_meta_key(commit_id)
    meta = dataset.version_state["meta"]
    meta.statistics = stats

    dataset.storage[meta_key] = meta
    dataset.storage[meta_key].is_dirty = True
    dataset.storage.autoflush = initial_autoflush
    dataset.storage.flush()  # write to dataset_meta.json


def load_meta(dataset):
    """Loads the meta info for the version state."""
    from ..core.tensor import Tensor

    version_state = dataset.version_state
    storage: LRUCache = dataset.storage
    storage.clear_muller_objects()
    dataset.info = None
    dataset.ds_diff = None
    meta = _get_dataset_meta_at_commit(storage, version_state["commit_id"])  # Sherry: OBS check problem here
    get_dataset_diff_at_commit(version_state["commit_id"], storage)

    version_state["meta"] = meta
    _tensors = version_state["full_tensors"]
    _tensors.clear()
    _tensor_names = version_state["tensor_names"]
    _tensor_names.clear()
    _tensor_names.update(meta.tensor_names)

    # Sherry: To improve the loading efficiency, we check all the meta data at one time.
    if dataset.split_tensor_meta:
        meta_file_list = []
        for tensor_key in _tensor_names.values():
            file_name = get_tensor_meta_key(tensor_key, dataset.version_state["commit_id"])
            meta_file_list.append(file_name)
    else:
        meta_file_list = [get_tensor_meta_key("", dataset.version_state["commit_id"])]
    flag = True
    try:
        # Sherry: Python does not have official overwrite functions. So we create a new method for all the
        # storage providers, including memory, local and roma.
        # Origin: get_filelist
        dataset.storage.get_items(set(meta_file_list))  # just checking, so we do not need the return value
    except KeyError:
        flag = False

    for tensor_key in _tensor_names.values():
        if tensor_key.startswith("__temp"):
            dataset.temp_tensors.append(tensor_key)
        _tensors[tensor_key] = Tensor(key=tensor_key, dataset=dataset, check_flag=flag)


def _get_dataset_meta_at_commit(storage, commit_id):
    """Get dataset meta at commit."""
    meta_key = get_dataset_meta_key(commit_id)

    meta = storage.get_muller_object(meta_key, DatasetMeta)  # Sherry: OBS check problem here. why no exception?
    if not meta.tensor_names:  # backward compatibility
        meta.tensor_names = {key: key for key in meta.tensors}
    storage.register_muller_object(meta_key, meta)
    return meta


def get_dataset_diff_at_commit(commit_id, storage):
    """Get dataset diff at commit."""
    path = get_dataset_diff_key(commit_id)
    try:
        diff = storage.get_muller_object(path, DatasetDiff)
    except KeyError:
        diff = DatasetDiff() # do we need rebuild method
    storage.register_muller_object(path, diff)
    return diff


def integrity_check(dataset):
    """Check the integrity of the dataset."""
    try:
        for k, t in dataset.get_tensors(include_disabled=False).items():
            n1 = t.meta.length
            engine = t.chunk_engine
            n2 = engine.chunk_id_encoder.num_samples
            if n1 != n2:
                raise ValueError(
                    f"Tensor meta and chunk id encoder have different number of samples ({n1} and {n2} "
                    f"respectively) for tensor {k}."
                )
            if engine.enable_tile_encoder:
                _ = engine.tile_encoder  # Sherry: Add tile_encoder here
    except Exception as e:
        raise DatasetCorruptError(
            f"The HEAD node of the branch {dataset.branch} of this dataset is in a corrupted state "
            f"and is likely not recoverable.Please run `ds.reset()` to revert the uncommitted changes "
            f"in order to continue making updates on this branch.",
        ) from e


def load_commit_info(commit_id: str, storage: LRUCache) -> Dict:
    """Loads the commit info from the storage."""
    lru_storage = storage
    storage = get_base_storage(storage)
    key = get_commit_info_key(commit_id)
    commit_info = json.loads(storage[key].decode("utf-8"))
    lru_storage.commit_info = commit_info
    return commit_info


def _generate_hash() -> str:
    hsh = hashlib.sha1()
    hsh.update(str(time.time()).encode("utf-8"))
    hsh.update(random.randrange(0, 1000000).to_bytes(4, "big"))
    return hsh.hexdigest()


def _replace_missing_with_head(missing_id: str, commits: Dict, branch_commit_map: Dict):
    new_commit_id = _generate_hash()
    branch = None
    parent_commit_id = None
    for commit_id, commit_info in commits.items():
        if missing_id in commit_info["children"]:
            commit_info["children"].remove(missing_id)
            commit_info["children"].append(new_commit_id)
            branch = commit_info["branch"]
            parent_commit_id = commit_id
            break

    commit_info = {
        "branch": branch,
        "children": [],
        "parent": parent_commit_id,
        "commit_message": None,
        "commit_time": None,
        "commit_user_name": None,
    }
    commits[new_commit_id] = commit_info
    branch_commit_map[branch] = new_commit_id

    return branch, parent_commit_id, new_commit_id


def _copy_metas(
    src_commit_id: str,
    dest_commit_id: str,
    storage: LRUCache,
) -> None:
    initial_autoflush = storage.autoflush
    storage.autoflush = False

    # Preprocess
    tensor_list, split_tensor_meta = (
        _copy_metas_prepocess(storage, src_commit_id, dest_commit_id))

    # For each tensor, record and process the meta key.
    src_key_list = []
    dest_key_list = []
    dest_dict = {}

    # For each tensor, record and process the meta key.
    for tensor in tensor_list:
        dest_dict, src_key_list, dest_key_list = _get_tensor_meta_key(split_tensor_meta, tensor, src_commit_id,
                                                                      dest_commit_id, storage, dest_dict, src_key_list,
                                                                      dest_key_list)
        dest_dict, src_key_list, dest_key_list = _get_tensor_chunk_id_key(tensor, src_commit_id, dest_commit_id,
                                                                          storage, dest_dict, src_key_list,
                                                                          dest_key_list)
        dest_dict, src_key_list, dest_key_list = _get_tensor_tile_id_key(tensor, src_commit_id, dest_commit_id,
                                                                          storage, dest_dict, src_key_list,
                                                                          dest_key_list)

    # Postprocess
    _copy_metas_postprocess(storage, src_key_list, dest_key_list, dest_dict, initial_autoflush)


def _copy_metas_prepocess(storage, src_commit_id, dest_commit_id):
    # dataset_meta must be fetched first, as it will be used later and should not be modified.
    src_dataset_meta = _get_dataset_meta_at_commit(storage, src_commit_id)
    src_dataset_meta.statistics = {}  # clear stat info
    dest_dataset_meta_key = get_dataset_meta_key(dest_commit_id)
    dest_dataset_meta = _convert_to_bytes(src_dataset_meta)
    storage[dest_dataset_meta_key] = dest_dataset_meta  # Changed something
    tensor_list = src_dataset_meta.tensors
    try:
        storage[get_tensor_meta_key("", src_commit_id)]
    except KeyError:
        split_tensor_meta = True
    else:
        split_tensor_meta = False

    if not split_tensor_meta:
        src_tensor_meta_key = get_tensor_meta_key("", src_commit_id)
        dest_tensor_meta_key = get_tensor_meta_key("", dest_commit_id)  # -> versions/commit_id/tensor_meta.json
        src_tensor_meta_dict = storage.get_muller_object(src_tensor_meta_key, dict)
        dest_tensor_meta_dict = {}
        for tensor in tensor_list:
            src_tensor_meta = src_tensor_meta_dict[tensor]
            dest_tensor_meta = src_tensor_meta
            dest_tensor_meta_dict[tensor] = dest_tensor_meta
        storage[dest_tensor_meta_key] = dest_tensor_meta_dict

    return tensor_list, split_tensor_meta


def _get_tensor_meta_key(split_tensor_meta, tensor, src_commit_id, dest_commit_id, storage,
                         dest_dict, src_key_list, dest_key_list):
    if split_tensor_meta:
        src_tensor_meta_key = get_tensor_meta_key(tensor, src_commit_id)
        dest_tensor_meta_key = get_tensor_meta_key(tensor, dest_commit_id)
        src_tensor_meta_cache = storage.get_item_from_cache(src_tensor_meta_key)
        if src_tensor_meta_cache:
            dest_dict[dest_tensor_meta_key] = src_tensor_meta_cache
        else:
            src_key_list.append(src_tensor_meta_key)
            dest_key_list.append(dest_tensor_meta_key)
    return dest_dict, src_key_list, dest_key_list


def _get_tensor_chunk_id_key(tensor, src_commit_id, dest_commit_id, storage,
                         dest_dict, src_key_list, dest_key_list):
    try:
        src_chunk_id_encoder_key = get_chunk_id_encoder_key(tensor, src_commit_id)
        dest_chunk_id_encoder_key = get_chunk_id_encoder_key(tensor, dest_commit_id)
        src_chunk_id_encoder_cache = storage.get_item_from_cache(src_chunk_id_encoder_key)
        if src_chunk_id_encoder_cache:
            dest_dict[dest_chunk_id_encoder_key] = src_chunk_id_encoder_cache
        else:
            src_key_list.append(src_chunk_id_encoder_key)
            dest_key_list.append(dest_chunk_id_encoder_key)
    except KeyError:
        pass
    return dest_dict, src_key_list, dest_key_list


def _get_tensor_tile_id_key(tensor, src_commit_id, dest_commit_id, storage,
                         dest_dict, src_key_list, dest_key_list):
    if WRITE_TILES_INDEX:
        try:
            src_tile_encoder_key = get_tensor_tile_encoder_key(tensor, src_commit_id)
            dest_tile_encoder_key = get_tensor_tile_encoder_key(tensor, dest_commit_id)
            src_tile_encoder_cache = storage.get_item_from_cache(src_tile_encoder_key)
            if src_tile_encoder_cache:
                dest_dict[dest_tile_encoder_key] = src_tile_encoder_cache
            else:
                src_key_list.append(src_tile_encoder_key)
                dest_key_list.append(dest_tile_encoder_key)
        except KeyError:
            pass
    return dest_dict, src_key_list, dest_key_list




def _copy_metas_postprocess(storage, src_key_list, dest_key_list, dest_dict, initial_autoflush):
    result = storage.next_storage.get_items(set(src_key_list), ignore_key_error=True)
    dest_from_next_storage = {}
    for src_key, dest_key in zip(src_key_list, dest_key_list):
        if result and src_key in result.keys():
            dest_from_next_storage[dest_key] = result[src_key]
    dest_dict.update(dest_from_next_storage)
    for path, value in dest_dict.items():
        storage[path] = _convert_to_bytes(value)

    storage.autoflush = initial_autoflush
    storage.flush()


def _convert_to_bytes(inp):
    return inp.tobytes() if isinstance(inp, MULLERMemoryObject) else inp


def create_commit_chunk_maps(
    src_commit_id: str,
    dest_commit_id: str,
    storage: LRUCache,
) -> None:
    """Creates commit chunk sets for all tensors in new commit."""
    tensor_list = _get_dataset_meta_at_commit(storage, src_commit_id).tensors
    for tensor in tensor_list:
        key = get_tensor_commit_chunk_map_key(tensor, dest_commit_id)
        storage[key] = CommitChunkMap()


def _create_new_head(
    storage: LRUCache, version_state, branch, parent_commit_id, new_commit_id
):
    # populate new commit folder
    _copy_metas(parent_commit_id, new_commit_id, storage)
    create_commit_chunk_maps(parent_commit_id, new_commit_id, storage)

    # create new node
    parent_node: CommitNode = version_state["commit_node_map"][parent_commit_id]
    new_node = CommitNode(branch=branch, commit_id=new_commit_id)
    new_node.parent = parent_node
    version_state["branch_commit_map"][branch] = new_commit_id
    version_state["commit_node_map"][new_commit_id] = new_node

    return new_node


def rebuild_version_info(storage: LRUCache):
    """Rebuilds version info from commit info."""
    # don't do anything if first commit info is missing
    try:
        load_commit_info(FIRST_COMMIT_ID, storage)
    except Exception:
        return None

    stack = [FIRST_COMMIT_ID]
    new_heads = []

    while stack:
        commit_id = stack.pop()

        try:
            commit_info = load_commit_info(commit_id, storage)
        except KeyError:
            if commit_id != FIRST_COMMIT_ID:
                new_head = _replace_missing_with_head(
                    commit_id, commits_rebuild, branch_commit_map_rebuild
                )
                new_heads.append(new_head)
                continue
            raise
        commits_rebuild[commit_id] = commit_info
        if commit_info["commit_time"] is None or not commit_info["children"]:  # it is head node
            branch_commit_map_rebuild[commit_info["branch"]] = commit_id
        if commit_info["checkout_node"]:
            if not commit_info["parent"]:
                branch_info_rebuild[commit_info["branch"]] = {"based_on": None,
                                                              "create_time": commit_info["commit_time"]}
            else:
                try:
                    based_branch = commits_rebuild[commit_info["parent"]]["branch"]
                except KeyError as e:
                    parent_node = commit_info["parent"]
                    raise VersionControlError(f"Can not obtain the branch of {parent_node}, please recheck.") from e
                branch_info_rebuild[commit_info["branch"]] = {"based_on": based_branch,
                                                              "create_time": commit_info["commit_time"]}
        if commit_id == FIRST_COMMIT_ID:
            branch_info_rebuild[commit_info["branch"]] = {"based_on": None,
                                                          "create_time": commit_info["commit_time"]}
        stack += commit_info["children"]

    if not commits_rebuild:
        return {}

    lru_storage = storage
    base_storage = get_base_storage(storage)
    tmp_lock = Lock(storage, get_version_control_info_lock_key(), duration=10)
    tmp_lock.acquire()  # Blocking
    try:
        del storage[get_version_control_info_key()]
    except KeyError:
        pass
    key = get_version_control_info_key()

    version_info = {"commits": commits_rebuild, "branches": branch_commit_map_rebuild,
                    "branches_info": branch_info_rebuild}
    base_storage[key] = json.dumps(version_info).encode("utf-8")
    lru_storage.version_info = version_info
    tmp_lock.release()

    version_info = _version_info_from_json(version_info)

    for new_head in new_heads:
        _ = _create_new_head(storage, version_info, *new_head)

    return version_info


def commit_diff_exists(
    version_state: Dict[str, Any], storage: LRUCache, tensor: str
) -> bool:
    """Checks if the commit chunk set exists for the given tensor in the current commit."""
    try:
        commit_id = version_state["commit_id"]
        key = get_tensor_commit_diff_key(tensor, commit_id)
        storage[key]  # check if commit diff file exists in base storage
        return True
    except KeyError:
        return False


def _current_commit_has_data(version_state: Dict[str, Any], storage: LRUCache) -> bool:
    """Checks if the current commit has any data present in it or not."""
    commit_id = version_state["commit_id"]
    try:
        dataset_diff_key = get_dataset_diff_key(commit_id)
        dataset_diff = storage.get_muller_object(dataset_diff_key, DatasetDiff)
        if dataset_diff.deleted or dataset_diff.renamed or dataset_diff.commit_diff_exist:
            return True
    except KeyError:
        pass
    return False


def _current_commit_has_info_modified(
    version_state: Dict[str, Any], storage: LRUCache
) -> bool:
    commit_id = version_state["commit_id"]
    try:
        dataset_diff_key = get_dataset_diff_key(commit_id)
        dataset_diff = storage.get_muller_object(dataset_diff_key, DatasetDiff)
        if dataset_diff.info_updated or dataset_diff.tensor_info_updated:
            return True
    except KeyError:
        pass

    return False


def current_commit_has_change(version_state: Dict[str, Any], storage: LRUCache) -> bool:
    """Return whether current commit has change. """
    return (
        version_state["commit_id"] == FIRST_COMMIT_ID
        or _current_commit_has_data(version_state, storage)
        or _current_commit_has_info_modified(version_state, storage)
    )


def discard_old_metas(
    src_commit_id: str,
    storage: LRUCache,
    tensors: Dict,
):
    """Discards the metas of the previous commit from cache, during checkout or when a new commit is made."""
    src_dataset_meta_key = get_dataset_meta_key(src_commit_id)
    all_src_keys.append(src_dataset_meta_key)

    src_dataset_diff_key = get_dataset_diff_key(src_commit_id)
    all_src_keys.append(src_dataset_diff_key)

    tensor_list = list(tensors.keys())

    try:
        storage[get_tensor_meta_key("", src_commit_id)]
    except KeyError:
        split_tensor_meta = True
    else:
        split_tensor_meta = False

    if not split_tensor_meta:
        src_tensor_meta_key = get_tensor_meta_key("", src_commit_id)
        all_src_keys.append(src_tensor_meta_key)

    for tensor in tensor_list:
        if split_tensor_meta:
            src_tensor_meta_key = get_tensor_meta_key(tensor, src_commit_id)
            all_src_keys.append(src_tensor_meta_key)

        src_chunk_id_encoder_key = get_chunk_id_encoder_key(tensor, src_commit_id)
        all_src_keys.append(src_chunk_id_encoder_key)

        src_tile_encoder_key = get_tensor_tile_encoder_key(tensor, src_commit_id)
        all_src_keys.append(src_tile_encoder_key)

    for key in all_src_keys:
        storage.dirty_keys.pop(key, None)
        if key in storage.lru_sizes:
            size = storage.lru_sizes.pop(key)
            storage.cache_used -= size
        try:
            del storage.cache_storage[key]
        except KeyError:
            pass


def _merge_commit_node_maps(map1, map2):
    merged_map = {}

    def _merge_node(commit_id):
        if commit_id in map1 and commit_id in map2:
            node1 = map1[commit_id]
            node2 = map2[commit_id]
            merged_node = CommitNode(node1.branch, node2.commit_id)

            for attr in (
                "commit_message",
                "commit_user_name",
                "commit_time",
                "is_checkpoint",
                "merge_parent",
                "total_samples_processed",
                "checkout_node",
            ):
                setattr(merged_node, attr, getattr(node1, attr) or getattr(node2, attr))
            for child in set(
                [node.commit_id for node in node1.children]
                + [node.commit_id for node in node2.children]
            ):
                merged_node.add_child(_merge_node(child))
        else:
            if commit_id in map1:
                orig_node = map1[commit_id]
            else:
                orig_node = map2[commit_id]
            merged_node = orig_node.copy()
            for child in [node.commit_id for node in orig_node.children]:
                merged_node.add_child(_merge_node(child))
        merged_map[commit_id] = merged_node
        return merged_node

    _ = _merge_node(FIRST_COMMIT_ID)
    return merged_map


def _is_one_fast_forward(old_branch_map, old_commit_map, new_branch_map, new_commit_map, branch):
    cur_node_id = new_branch_map[branch]  # the latest node to append
    parent_node_id = new_commit_map[cur_node_id].parent.commit_id
    try:
        latest_old_id = old_branch_map[branch]
    except KeyError:
        if len(new_branch_map) - 1 == len(old_branch_map):
            par_children = set(old_commit_map[parent_node_id].children)
            new_children = set(new_commit_map[parent_node_id].children)
            diff_children = new_children.difference(par_children)  # parent node in new exceed one node of old.
            return len(diff_children) == 1 and diff_children.pop().commit_id == cur_node_id
        return False
    return (len(new_commit_map) == len(old_commit_map) + 1) and parent_node_id == latest_old_id


def _merge_version_info(old_info, new_info, cur_branch):
    try:
        if _is_one_fast_forward(old_info['branch_commit_map'], old_info['commit_node_map'],
                                new_info['branch_commit_map'], new_info['commit_node_map'], cur_branch):
            commit_node_map = new_info["commit_node_map"]
        else:
            commit_node_map = _merge_commit_node_maps(
                old_info["commit_node_map"], new_info["commit_node_map"]
            )
    except Exception:
        commit_node_map = _merge_commit_node_maps(
            old_info["commit_node_map"], new_info["commit_node_map"]
        )
    branch_commit_map = {}
    branch_commit_map.update(old_info["branch_commit_map"])
    branch_commit_map.update(new_info["branch_commit_map"])
    branches_info = {}
    branches_info.update(old_info["branch_info"])
    branches_info.update(new_info["branch_info"])
    return {
        "commit_node_map": commit_node_map,
        "branch_commit_map": branch_commit_map,
        "branch_info": branches_info
    }


def _version_info_to_json(info):
    commit_node_map, branch_commit_map, branch_info = (
        info["commit_node_map"],
        info["branch_commit_map"],
        info["branch_info"]
    )
    commits = {}
    for return_commit, node in commit_node_map.items():
        commits[return_commit] = node.to_json()
    branches_info = {}
    for branch_name, target_info in branch_info.items():
        branches_info[branch_name] = {
            "based_on": target_info["based_on"],
            "create_time": target_info["create_time"].timestamp() if target_info["create_time"] else None,
        }
    return {
        "commits": commits,
        "branches": branch_commit_map,
        "branches_info": branches_info
    }


def save_version_info(version_state: Dict[str, Any], storage: LRUCache) -> None:
    """Saves the current version info to the storage."""
    lru_storage = storage
    storage = get_base_storage(storage)
    temp_lock = Lock(storage, get_version_control_info_lock_key(), duration=10)
    temp_lock.acquire()  # Blocking
    key = get_version_control_info_key()
    new_version_info = {
        "commit_node_map": version_state["commit_node_map"],
        "branch_commit_map": version_state["branch_commit_map"],
        "branch_info": version_state["branch_info"],
    }
    try:
        old_version_info = _version_info_from_json(
            json.loads(storage[key].decode("utf-8"))
        )
        version_info = _merge_version_info(old_version_info, new_version_info, version_state['branch'])
    except KeyError:
        version_info = new_version_info
    # Modification: Instead of directly flushing to next_storage, we flush to the LRUCache first.
    lru_storage.version_state = version_state
    storage[key] = json.dumps(_version_info_to_json(version_info)).encode("utf-8")
    temp_lock.release()


def save_commit_info(commit_node: CommitNode, storage: LRUCache) -> None:
    """Saves the commit info to the storage."""
    lru_storage = storage
    storage = get_base_storage(storage)
    key = get_commit_info_key(commit_node.commit_id)
    storage[key] = json.dumps(commit_node.to_json()).encode("utf-8")
    lru_storage.commit_info = commit_node.to_json()
    commit_node.info_updated = False


def commit(
    dataset,
    message: Optional[str] = None,
    temp_hash: Optional[str] = None,
    flush_version_control_info: bool = True,
    reload_meta: bool = True,
    is_checkpoint: bool = False,
    total_samples_processed: int = 0,
    author_name: Optional[str] = None,
) -> None:
    """Modifies the version state to reflect the commit and also copies required data to the new commit directory."""

    storage = dataset.storage
    version_state = dataset.version_state
    storage.check_readonly()
    integrity_check(dataset)

    # if not the head node, checkout to an auto branch that is newly created
    auto_checkout(dataset, flush_version_control_info=False)
    stored_commit_node: CommitNode = version_state["commit_node"]
    stored_commit_id = version_state["commit_id"]
    if temp_hash:
        if temp_hash in version_state["commit_node_map"]:
            raise CommitError(f"Commit {temp_hash} already exists")
    else:
        temp_hash = _generate_hash()
    version_state["commit_id"] = temp_hash
    new_node = CommitNode(branch=version_state["branch"], commit_id=temp_hash, commit_user_name=author_name)

    # Sherry: Previously it uses dataset.username here to obtain the username (dataset token related).
    stored_commit_node.add_successor(node=new_node, author=author_name if author_name else obtain_current_user(),
                                     message=message)
    stored_commit_node.is_checkpoint = is_checkpoint
    stored_commit_node.total_samples_processed = total_samples_processed
    version_state["commit_node"] = new_node
    version_state["branch_commit_map"][version_state["branch"]] = version_state[
        "commit_id"
    ]
    version_state["commit_node_map"][temp_hash] = new_node
    _copy_metas(stored_commit_id, temp_hash, storage)
    create_commit_chunk_maps(stored_commit_id, temp_hash, storage)
    discard_old_metas(stored_commit_id, storage, version_state["full_tensors"])
    if reload_meta:
        load_meta(dataset)
    if flush_version_control_info:
        save_version_info(version_state, storage)
        save_commit_info(stored_commit_node, storage)
        save_commit_info(new_node, storage)
    else:
        stored_commit_node.info_updated = True
        new_node.info_updated = True


def auto_commit(dataset, message: str, is_raise: bool, flush_version_control_info: bool = True) -> None:
    """Automatically commits to the current branch before a checkout to a newly created branch
        if the current node is the head node and has changes.
    """
    version_state = dataset.version_state
    commit_node = version_state["commit_node"]
    head = commit_node.is_head_node
    if not head:
        return

    if not current_commit_has_change(version_state, dataset.storage):
        parent_id = commit_node.parent.commit_id  # type: ignore
        checkout(dataset, parent_id, False)
        return
    if is_raise:
        raise DatasetCorruptError(f"The given target state:{commit_node.commit_id} has uncommitted changes. "
                                              f"Please checkout to the target state and commit first.")
    original_commit_id = version_state["commit_id"]
    branch = version_state["branch"]
    logger.info(
        f"Auto committing to branch '{branch}' before checkout as currently at head node."
    )
    commit(
        dataset,
        message,
        flush_version_control_info=flush_version_control_info,
        reload_meta=False,
        author_name=version_state["commit_node_map"][original_commit_id].commit_user_name,
    )
    checkout(dataset, original_commit_id)


def checkout(
    dataset,
    address: str,
    create: bool = False,
    temp_hash: Optional[str] = None,
    flush_version_control_info=True,
) -> None:
    """Modifies the version state to reflect the checkout and also copies required data to the new branch directory
        if a new one is being created.
    """
    storage = dataset.storage
    version_state = dataset.version_state
    original_commit_id = version_state["commit_id"]

    if address in version_state["branch_commit_map"].keys():
        if create:
            raise CheckoutError(f"Can't create new branch, '{address}' already exists.")
        new_commit_id = version_state["branch_commit_map"][address]
        if original_commit_id == new_commit_id:
            return
        if not storage.read_only:
            storage.flush()
        version_state["commit_id"] = new_commit_id
        version_state["commit_node"] = version_state["commit_node_map"][new_commit_id]
        version_state["branch"] = address

    elif address in version_state["commit_node_map"].keys():
        if create:
            raise CheckoutError(
                f"Can't create new branch, commit '{address}' already exists."
            )
        if address == original_commit_id:
            return
        if not storage.read_only:
            storage.flush()
        version_state["commit_id"] = address
        version_state["commit_node"] = version_state["commit_node_map"][address]
        version_state["branch"] = version_state["commit_node"].branch
    elif create:
        storage.check_readonly()
        # if the original commit is head of the branch, auto commit and checkout to original commit
        # before creating new branch

        auto_commit(
            dataset,
            f"auto commit before checkout to {address}",
            False,
            flush_version_control_info=False
        )

        if temp_hash:
            if temp_hash in version_state["commit_node_map"]:
                raise CommitError(f"Commit {temp_hash} already exists")
            new_commit_id = temp_hash
        else:
            new_commit_id = _generate_hash()

        new_node = CommitNode(branch=address, commit_id=new_commit_id)
        new_node.add_checkout_node()
        stored_commit_node = version_state["commit_node"]
        stored_commit_node.add_child(new_node)
        version_state["commit_id"] = new_commit_id
        version_state["commit_node"] = new_node
        version_state["branch"] = address
        version_state["commit_node_map"][new_commit_id] = new_node
        version_state["branch_commit_map"][address] = new_commit_id
        version_state["branch_info"][address] = {"based_on": stored_commit_node.branch,
                                                 "create_time": new_node.commit_time}
        if flush_version_control_info:
            save_version_info(version_state, storage)
            save_commit_info(new_node, storage)
            save_commit_info(stored_commit_node, storage)
        else:
            stored_commit_node.info_updated = True
            new_node.info_updated = True
        _copy_metas(original_commit_id, new_commit_id, storage)
        create_commit_chunk_maps(original_commit_id, new_commit_id, storage)
    else:
        raise CheckoutError(
            f"Address {address} not found. If you want to create a new branch, use checkout with create=True"
        )

    discard_old_metas(
        original_commit_id,
        storage,
        version_state["full_tensors"],
    )

    try:
        load_meta(dataset)
    except Exception as e:
        checkout(dataset, original_commit_id)
        raise CheckoutError(
            f"Unable to checkout to '{address}', failed to load meta data."
        ) from e


def auto_checkout(dataset, flush_version_control_info: bool = True) -> bool:
    """Automatically checks out if current node is not the head node of the branch.
        This may happen either during commit/setitem/append/extend/create_tensor/delete_tensor/info updates.
    """
    version_state = dataset.version_state
    if not version_state["commit_node"].is_head_node:
        current_branch = version_state["branch"]
        auto_branch = f"auto_{_generate_hash()}"
        logger.info(
            f"Automatically checking out to branch '{auto_branch}' as not currently at the head node "
            f"of branch '{current_branch}'."
        )
        checkout(
            dataset,
            auto_branch,
            True,
            flush_version_control_info=flush_version_control_info,
        )
        return True
    return False


def _replace_head(storage, version_state, commit_id, new_head):
    parent_node = new_head.parent
    del version_state["commit_node_map"][commit_id]
    for i, child in enumerate(parent_node.children):
        if child.commit_id == commit_id:
            parent_node.children[i] = new_head
            break

    save_version_info(version_state, storage)


def _delete_version_from_storage(storage, tensors, commit_id):
    for tensor_name in tensors:
        delete_target_commit_chunk(storage, tensor_name, commit_id)
    deletion_folder = "/".join(("versions", commit_id))
    storage.clear(prefix=deletion_folder)
    storage.flush()


def _find_branch_commits(branch_name: str, version_state: dict) -> List[str]:
    """
    Returns a list of all commits used by the given branch
    """
    all_branch_commits = []
    branch_commit = version_state["branch_commit_map"][branch_name]
    branch_commit_node = version_state["commit_node_map"][branch_commit]
    while branch_commit_node.branch == branch_name:
        all_branch_commits.append(branch_commit_node.commit_id)
        if (
            len(
                [
                    child
                    for child in branch_commit_node.children
                    if child.commit_id not in all_branch_commits
                ]
            )
            > 0
        ):
            raise VersionControlError(
                f"Cannot delete branch {branch_name} because it has sub-branches"
            )
        branch_commit_node = branch_commit_node.parent
    return all_branch_commits


def delete_branch(
    dataset,
    branch_name: str,
) -> None:
    """
    Deletes the branch and cleans up any unneeded data.
    Branches can only be deleted if there are no sub-branches and if it has never been merged into another branch.
    """

    storage = dataset.storage
    storage.check_readonly()

    version_state = dataset.version_state
    if version_state["branch"] == branch_name:
        raise VersionControlError(
            f"Cannot delete the currently checked out branch: {branch_name}"
        )

    if branch_name == "main":
        raise VersionControlError("Cannot delete the main branch")

    if branch_name not in version_state["branch_commit_map"].keys():
        raise VersionControlError(f"Branch {branch_name} does not exist")

    storage = get_base_storage(storage)
    versioncontrol_lock = PersistentLock(storage, get_version_control_info_lock_key())
    versioncontrol_lock.acquire()  # Blocking

    lock.lock_dataset(
        dataset, version=dataset.version_state["branch_commit_map"][branch_name]
    )

    try:
        # Check that nothing branches with sub-branch to delete
        all_branch_commits = _find_branch_commits(branch_name, version_state)

        # Check that nothing points to any of the commits to delete
        for commit_id, commit_node in version_state["commit_node_map"].items():
            if commit_id in all_branch_commits:
                continue

            if commit_node.merge_parent in all_branch_commits:
                raise VersionControlError(
                    f"Cannot delete branch {branch_name} because it has "
                    f"been previously merged into {commit_node.branch}"
                )

            if commit_node.parent in all_branch_commits:
                raise VersionControlError(
                    f"Cannot delete branch {branch_name} because it has been previously merged"
                )

        _delete_branch_and_commits(branch_name, all_branch_commits, dataset, storage)

    finally:
        versioncontrol_lock.release()


def _delete_branch_and_commits(
    branch_name: str, all_branch_commits: List[str], dataset, storage
) -> None:
    """
    Physically deletes the given branch and list of commits from the version_control_info.json and versions directories.
    Any validation on the information should have been performed before this method is called
    """
    version_state = dataset.version_state

    version_state["branch_commit_map"].pop(branch_name)
    version_state["branch_info"].pop(branch_name)
    for commit_id, commit_node in list(version_state["commit_node_map"].items()):
        if commit_id in all_branch_commits:
            version_state["commit_node_map"].pop(commit_id)
            continue

        commit_node.children = [
            child
            for child in commit_node.children
            if child.commit_id not in all_branch_commits
        ]
    for commit_id in all_branch_commits:
        _delete_version_from_storage(dataset.storage, dataset.tensors, commit_id)

    storage[get_version_control_info_key()] = json.dumps(
        _version_info_to_json(
            {
                "commit_node_map": version_state["commit_node_map"],
                "branch_commit_map": version_state["branch_commit_map"],
                "branch_info": version_state["branch_info"]
            }
        )
    ).encode("utf-8")


def replace_head(storage, version_state, tensor_names, reset_commit_id):
    """Replace HEAD of current branch with new HEAD"""
    branch = version_state["commit_node_map"][reset_commit_id].branch
    parent_commit_id = version_state["commit_id"]
    new_commit_id = _generate_hash()

    new_node = _create_new_head(
        storage, version_state, branch, parent_commit_id, new_commit_id
    )

    _replace_head(storage, version_state, reset_commit_id, new_node)

    _delete_version_from_storage(storage, tensor_names, reset_commit_id)

    return new_node.commit_id


def warn_node_checkout(commit_node: CommitNode, create: bool):
    """Throws a warning if there are no commits in a branch after checkout.
    This warning isn't thrown if the branch was newly created.
    """
    if not create and commit_node.is_head_node:
        branch = commit_node.branch
        parent = commit_node.parent
        if parent is None or parent.branch != branch:
            warnings.warn(
                f"The branch ({branch}) that you have checked out to, has no commits."
            )


def get_parent_and_reset_commit_ids(version_info, address):
    """Returns parent commit id and commit id which will be reset. Returns (False, False)
        if address is a non-HEAD commit id
    """
    if address in version_info["branch_commit_map"]:
        commit_id = version_info["branch_commit_map"][address]
    elif address in version_info["commit_node_map"]:
        commit_id = address
    commit_node = version_info["commit_node_map"][commit_id]
    if not commit_node.is_head_node:
        return False, False
    parent_node = commit_node.parent
    if parent_node is None:
        previous_commit_id = None
    else:
        previous_commit_id = parent_node.commit_id
    return previous_commit_id, commit_id


def reset_and_checkout(ds, address, err, verbose=True):
    """Reset the dataset and checkout."""
    storage = ds.storage
    version_state = ds.version_state

    parent_commit_id, reset_commit_id = get_parent_and_reset_commit_ids(
        version_state, address
    )
    if parent_commit_id is False:
        # non-head node corrupted
        raise err
    if parent_commit_id is None:
        # no commits in the dataset
        storage.clear()
        ds.populate_meta()
        load_meta(ds)
        return []

    ds.checkout(parent_commit_id)
    new_commit_id = replace_head(storage, version_state, ds.tensors, reset_commit_id)
    ds.checkout(new_commit_id)

    current_node = version_state["commit_node_map"][ds.commit_id]
    if verbose:
        logging.info(f"HEAD reset. Current version: %s\n", current_node)
    return ds.commit_id


def delete_target_commit_chunk(storage, tensor_name, commit_id):
    """delete chunks belong to the commit."""
    chunk_map_key = get_tensor_commit_chunk_map_key(tensor_name, commit_id)
    try:
        chunk_map = storage.get_muller_object(chunk_map_key, CommitChunkMap).chunks
    except Exception as e:
        raise KeyError(f"commit {commit_id} has no chunk_set") from e

    key_set = set()
    for chunk_name in chunk_map:
        chunk_key = get_chunk_key(tensor_name, chunk_name)
        key_set.add(chunk_key)
    storage.del_items(key_set)


def save_branch_metadata(storage, branch_name: str, owner: str, parent_branch: Optional[str] = None):
    """Save branch metadata including ownership information.
    
    Args:
        storage: Storage provider instance
        branch_name: Name of the branch
        owner: Username of the branch owner (creator)
        parent_branch: Name of the parent branch (optional)
    """
    branch_meta_path = f"branches/{branch_name}/branch_meta.json"
    
    metadata = {
        "branch_name": branch_name,
        "owner": owner,
        "created_at": datetime.utcnow().timestamp(),
        "parent_branch": parent_branch,
    }
    
    storage[branch_meta_path] = json.dumps(metadata).encode("utf-8")


def load_branch_metadata(storage, branch_name: str) -> Dict:
    """Load branch metadata.
    
    Args:
        storage: Storage provider instance
        branch_name: Name of the branch
        
    Returns:
        Dict containing branch metadata, or default dict if not found
    """
    branch_meta_path = f"branches/{branch_name}/branch_meta.json"
    
    try:
        return json.loads(storage[branch_meta_path].decode("utf-8"))
    except KeyError:
        # Fallback for existing branches without metadata
        # Try to infer owner from the first commit on the branch
        return {"branch_name": branch_name, "owner": None, "created_at": None, "parent_branch": None}


def get_branch_owner(dataset, branch_name: str) -> Optional[str]:
    """Get the owner of a branch (the user who created it).
    
    Args:
        dataset: Dataset instance
        branch_name: Name of the branch
        
    Returns:
        Username of the branch owner, or None if not found
    """
    # First try to load from branch metadata
    try:
        metadata = load_branch_metadata(dataset.storage, branch_name)
        if metadata.get("owner"):
            return metadata["owner"]
    except Exception:
        pass
    
    # Fallback: infer from version state
    version_state = dataset.version_state
    
    # Get the commit that created this branch
    branch_commit_id = version_state.get("branch_commit_map", {}).get(branch_name)
    if not branch_commit_id:
        return None
    
    # Find the commit node that marks the branch creation
    commit_node = version_state.get("commit_node_map", {}).get(branch_commit_id)
    if not commit_node:
        return None
    
    # Traverse back to find the first commit on this branch
    while commit_node.parent and commit_node.parent.branch == branch_name:
        commit_node = commit_node.parent
    
    return commit_node.commit_user_name


def auto_commit_before_checkout(dataset, target_address: str) -> bool:
    """Auto-commit uncommitted changes before checkout.
    
    This function automatically commits any pending changes in the current
    branch before checking out to a different branch or commit, ensuring
    that users always work on HEAD nodes.
    
    Args:
        dataset: Dataset instance
        target_address: The target branch or commit ID to checkout to
        
    Returns:
        bool: True if auto-commit was performed, False otherwise
    """
    if not dataset.has_head_changes:
        return False
    
    try:
        from muller.client.log import logger
        commit_message = f"Auto-commit before checkout to {target_address} at {datetime.utcnow().isoformat()}"
        
        # Use the commit function from this module
        commit(dataset, message=commit_message)
        logger.info(f"Auto-committed changes before checkout to {target_address}")
        return True
    except Exception as e:
        from muller.client.log import logger
        logger.warning(f"Auto-commit failed: {e}")
        return False
