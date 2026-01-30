# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/merge.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import collections
import logging
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from muller.constants import DATASET_UUID_NAME, CREATE_TENSOR_HIDDEN_UUID, FIRST_COMMIT_ID
from muller.core.meta.dataset_meta import DatasetMeta
from muller.core.meta.encode.chunk_id import ChunkIdEncoder
from muller.core.version_control.commit_chunk_map import CommitChunkMap
from muller.core.version_control.commit_diff import CommitDiff
from muller.core.version_control.commit_node import CommitNode
from muller.core.version_control.interface.diff_interface import sanitize_commit, get_lowest_common_ancestor, \
    merge_renamed_deleted, has_change
from muller.util.class_label import convert_to_text
from muller.util.exceptions import (MergeConflictError, MergeMismatchError, TensorDoesNotExistError, MergeLostUUid,
                                   ReadOnlyModeError)
from muller.util.keys import get_tensor_commit_diff_key, get_tensor_meta_key, \
    get_chunk_id_encoder_key, get_tensor_tile_encoder_key, get_creds_encoder_key, \
    get_dataset_meta_key, get_tensor_commit_chunk_map_key
from muller.util.remove_cache import create_read_copy_dataset
from muller.util.version_control import auto_checkout, auto_commit, commit, checkout

Records = collections.namedtuple("Records", ["del_ori_idx", "del_ori_values", "del_tar_idx",
                                             "del_tar_values", "app_ori_idx", "app_ori_values", "app_tar_idx",
                                             "app_tar_values", "update"])
RecordsCache = collections.namedtuple("Records_Cache", ["target_commit_id", "original_commit_id",
                                                         "tensor_name", "app_ori_idx", "app_tar_idx", "delete_ori",
                                                         "delete_tar", "original_id_to_index_map",
                                                         "target_id_to_index_map", "updated_indexes",
                                                         "detect_conflicts"])


def merge_detect(
        dataset,
        original_node,
        target_node,
        show_value: bool = False
):
    """Function to detect merge."""
    nodes: Dict[str, CommitNode] = {}
    nodes["original"] = original_node
    nodes["target"] = target_node

    lca_id = get_lowest_common_ancestor(original_node, target_node)
    try:
        nodes["lca"] = dataset.version_state["commit_node_map"][lca_id]
    except KeyError as e:
        raise ValueError(f"Can not find the commit id {lca_id} from commit_node_map.") from e

    target_ds = create_read_copy_dataset(dataset, target_node.commit_id)

    if lca_id == target_node.commit_id:
        raise ValueError("No merge needed, target id is an ancestor of the current commit")
    (
        _,
        common_tensors,
        _,
        conflict_tensors,
    ) = get_new_common_deleted_tensors(dataset, target_ds, lca_id, False, True)

    records = {
        tensor_name: common_tensor_detect(
            tensor_name, dataset, target_ds, nodes, show_value
        )
        for tensor_name in common_tensors
    }
    return conflict_tensors, records


def get_indexes_values(
        dataset,
        tensor_name,
        uuids,
        commit_id,
        id_to_index_map,
        show_value:bool = False,
        is_del:bool = False
):
    """Function to generate indexes and values of append and delete conflicts."""
    indexes = []
    values = []
    if tensor_name == "_uuid" and is_del:
        return indexes, values

    for idx in uuids:
        try:
            target_index: int = id_to_index_map[idx]
            indexes.append(target_index)
        except KeyError as e:
            raise MergeLostUUid from e
    indexes = sorted(indexes)
    if show_value:
        original_id = dataset.pending_commit_id
        dataset.checkout(commit_id, False)
        for i in indexes:
            value = dataset[tensor_name][i].numpy(aslist=True)
            values.append(value)
        dataset.checkout(original_id, False)
    return indexes, values


def get_uuids_values(
        dataset,
        tensor_name,
        uuids,
        commit_id,
        id_to_index_map,
):
    """Function to obtain values according to given uuids."""
    uuids_values = {}
    original_id = dataset.pending_commit_id
    dataset.checkout(commit_id, False)
    for idx in uuids:
        try:
            target_index: int = id_to_index_map[idx]
        except KeyError as e:
            raise MergeLostUUid from e
        value = dataset[tensor_name][target_index].numpy(aslist=True)
        uuids_values[idx] = value
    dataset.checkout(original_id, False)
    return uuids_values


def get_update_values(
        dataset,
        tensor_name,
        ori_commit_id,
        tar_commit_id,
        conflict_indexes,
        show_value: bool = False
):
    """Function to get values and indexes of conflict update."""
    update_values = {
        'update_tar': [],
        'update_ori': []
    }
    original_id = dataset.pending_commit_id
    for item in conflict_indexes:
        if item[0] is not None:
            if show_value:
                dataset.checkout(ori_commit_id, False)
                value = dataset[tensor_name][item[0]].numpy(aslist=True)
                update_values['update_ori'].append({item[0]: value})
            else:
                update_values['update_ori'].append({item[0]})
        else:
            update_values['update_ori'].append({})
        if item[1] is not None:
            if show_value:
                dataset.checkout(tar_commit_id, False)
                value = dataset[tensor_name][item[1]].numpy(aslist=True)
                update_values['update_tar'].append({item[1]: value})
            else:
                update_values['update_tar'].append({item[1]})
        else:
            update_values['update_tar'].append({})
    dataset.checkout(original_id, False)
    return update_values


def handle_update_ids(detect_conflicts, conflict_resolution):
    """Function to generate update indexes according to conflict solution."""
    resurrect_indexes: List[int] = []
    conflict_update_indexes: List[Tuple[Optional[int], int]] = []
    for item in detect_conflicts:
        if item[0]:
            conflict_update_indexes.append((item[0], item[1]))
        else:
            if conflict_resolution == "theirs":
                resurrect_indexes.append(item[1])
            elif not conflict_resolution:
                conflict_update_indexes.append((None, item[1]))
    return resurrect_indexes, conflict_update_indexes


def detect_update_conflicts(
        original_update_commit_map,
        target_update_commit_map,
        original_id_to_index_map,
        target_id_to_index_map,
        conflict_resolution,
        is_detect:bool = True,
) -> Tuple[List[Tuple[int, int]], List[Tuple[Optional[int], int]], List[int], List[Tuple[Optional[int], int]]]:
    """Finds the conflicts between the original commit and target id.

    Args:
        original_update_commit_map: A dictionary mapping sample ids to a list of commit ids that updated the sample.
        target_update_commit_map: A dictionary mapping sample ids to a list of commit ids that updated the sample.
        original_id_to_index_map: A dictionary mapping sample ids to their index in the original commit.
        target_id_to_index_map: A dictionary mapping sample ids to their index in the target id.
        conflict_resolution: The strategy to use to resolve merge conflicts.
        is_detect: Whether it is detect.

    Returns:
        updated indexes, conflict_update_indexes, resurrect_indexes, detect_conflicts
    """

    updated_indexes: List[Tuple[int, int]] = []
    resurrect_indexes: List[int] = []
    conflict_update_indexes: List[Tuple[Optional[int], int]] = []
    detect_conflicts: List[Tuple[Optional[int], int]] = []

    for temp_id in target_update_commit_map:
        idx = None
        for i, item in enumerate(target_update_commit_map[temp_id]):
            if item in set(original_update_commit_map[temp_id]):
                idx = i
                break

        # this means that the sample was only updated in the target commit, no conflict
        if not original_update_commit_map[temp_id] or (
                idx is not None and target_update_commit_map[temp_id][idx] == original_update_commit_map[temp_id][0]
        ):
            (updated_indexes, resurrect_indexes,
             conflict_update_indexes, detect_conflicts) = _get_updated_and_conflict_indexes(temp_id,
                                      target_id_to_index_map,
                                      original_id_to_index_map,
                                      updated_indexes,
                                      conflict_update_indexes,
                                      is_detect,
                                      conflict_resolution,
                                      resurrect_indexes,
                                      detect_conflicts)

        # if no id is common or if a commit id other than the most recent commit_id is in common, there's a conflict
        elif idx is None or idx > 0:
            if is_detect:
                detect_conflicts.append((original_id_to_index_map[temp_id], target_id_to_index_map[temp_id]))
            else:
                conflict_update_indexes.append((original_id_to_index_map[temp_id], target_id_to_index_map[temp_id]))

    return updated_indexes, conflict_update_indexes, resurrect_indexes, detect_conflicts


def _get_updated_and_conflict_indexes(temp_id,
                                      target_id_to_index_map,
                                      original_id_to_index_map,
                                      updated_indexes,
                                      conflict_update_indexes,
                                      is_detect,
                                      conflict_resolution,
                                      resurrect_indexes,
                                      detect_conflicts):

    target_idx: int = target_id_to_index_map[temp_id]
    try:
        original_idx: int = original_id_to_index_map[temp_id]
        updated_indexes.append((original_idx, target_idx))
    except KeyError:  # this means the update in target has been popped in original
        if not is_detect and conflict_resolution == "theirs":
            resurrect_indexes.append(target_idx)
        elif not is_detect and not conflict_resolution:
            conflict_update_indexes.append((None, target_idx))
        elif is_detect:
            detect_conflicts.append((None, target_idx))
    return updated_indexes, resurrect_indexes, conflict_update_indexes, detect_conflicts


def common_tensor_detect(
        tensor_name: str,
        dataset,
        target_dataset,
        nodes: Dict[str, CommitNode],
        show_value:bool = False
):
    """Function to detect conflicts on common tensors."""
    if tensor_name == "_uuid":
        show_value = False

    original_id_to_index_map, target_id_to_index_map, delete_ori, delete_tar, detail_dict \
        = _get_diff_ids_and_values(dataset,
                                   tensor_name,
                                   nodes["original"],
                                   target_dataset,
                                   nodes["target"],
                                   nodes["lca"],
                                   show_value)

    common_dict = _get_common_tensor_records(target_dataset,
                               tensor_name,
                               nodes,
                               dataset,
                               original_id_to_index_map,
                               target_id_to_index_map,
                               show_value)

    dataset.storage.add_records_cache_merge(RecordsCache(nodes["target"].commit_id,
                                                         nodes["original"].commit_id,
                                                         tensor_name,
                                                         detail_dict["app_ori_idx"],
                                                         detail_dict["app_tar_idx"],
                                                         delete_ori,
                                                         delete_tar,
                                                         original_id_to_index_map,
                                                         target_id_to_index_map,
                                                         common_dict["updated_indexes"],
                                                         common_dict["detect_conflicts"]))

    records = Records(detail_dict["del_ori_idx"],
                      detail_dict["del_ori_values"],
                      detail_dict["del_tar_idx"],
                      detail_dict["del_tar_values"],
                      detail_dict["app_ori_idx"],
                      detail_dict["app_ori_values"],
                      detail_dict["app_tar_idx"],
                      detail_dict["app_tar_values"],
                      common_dict["update_values"])
    return records


def _get_diff_ids_and_values(dataset, tensor_name, original_node, target_dataset, target_node, lca_node, show_value):
    delete_ori, delete_tar, append_ori, append_tar, index_map_dict = _get_diff_in_delete_and_appends(dataset,
                                                                                                     tensor_name,
                                                                                                     original_node,
                                                                                                     target_dataset,
                                                                                                     target_node,
                                                                                                     lca_node)

    del_ori_idx, del_ori_values = get_indexes_values(dataset,
                             tensor_name,
                             delete_ori,
                             lca_node.commit_id,
                             index_map_dict.get("lca_id_to_index_map"),
                             show_value,
                             True)
    del_tar_idx, del_tar_values = get_indexes_values(dataset,
                             tensor_name,
                             delete_tar,
                             lca_node.commit_id,
                             index_map_dict.get("lca_id_to_index_map"),
                             show_value,
                             True)
    app_ori_idx, app_ori_values = get_indexes_values(dataset,
                             tensor_name,
                             append_ori,
                             original_node.commit_id,
                             index_map_dict.get("original_id_to_index_map"),
                             show_value)
    app_tar_idx, app_tar_values = get_indexes_values(dataset,
                             tensor_name,
                             append_tar,
                             target_node.commit_id,
                             index_map_dict.get("target_id_to_index_map"),
                             show_value)
    detail_dict = {"del_ori_idx": del_ori_idx,
                   "del_ori_values": del_ori_values,
                   "del_tar_idx": del_tar_idx,
                   "del_tar_values": del_tar_values,
                   "app_ori_idx": app_ori_idx,
                   "app_ori_values": app_ori_values,
                   "app_tar_idx": app_tar_idx,
                   "app_tar_values": app_tar_values}
    return (index_map_dict.get("original_id_to_index_map"),
            index_map_dict.get("target_id_to_index_map"),
            delete_ori, delete_tar, detail_dict)


def _get_diff_in_delete_and_appends(dataset, tensor_name, original_node, target_dataset, target_node, lca_node):
    index_map_dict = {}
    original_ids = dataset.get_tensor_uuids(tensor_name, original_node.commit_id)
    index_map_dict.update({"original_id_to_index_map": {id: idx for idx, id in enumerate(original_ids)}})

    target_ids = target_dataset.get_tensor_uuids(tensor_name, target_node.commit_id)
    index_map_dict.update({"target_id_to_index_map": {id: idx for idx, id in enumerate(target_ids)}})

    lca_ids = dataset.get_tensor_uuids(tensor_name, lca_node.commit_id)
    index_map_dict.update({"lca_id_to_index_map": {id: idx for idx, id in enumerate(lca_ids)}})

    delete_ori, delete_tar, _ = get_deleted_ids(original_ids, target_ids, lca_ids)
    append_ori, append_tar = get_append_ids(original_ids, target_ids, lca_ids)

    return delete_ori, delete_tar, append_ori, append_tar, index_map_dict


def _get_common_tensor_records(target_dataset,
                               tensor_name,
                               nodes,
                               dataset,
                               original_id_to_index_map,
                               target_id_to_index_map,
                               show_value):
    target_update_commit_map = get_updates_commit_ids_for_node(
        target_dataset, tensor_name, nodes["target"], nodes["lca"]
    )

    original_update_commit_map = get_updates_commit_ids_for_node(
        dataset, tensor_name, nodes["original"], nodes["lca"]
    )

    updated_indexes, _, _, detect_conflicts = detect_update_conflicts(
        original_update_commit_map, target_update_commit_map,
        original_id_to_index_map, target_id_to_index_map,
        None, True
    )
    update_values = get_update_values(dataset, tensor_name, nodes["original"].commit_id, nodes["target"].commit_id,
                                      detect_conflicts, show_value)
    return {"updated_indexes": updated_indexes, "detect_conflicts": detect_conflicts, "update_values": update_values}



def direct_detect(tensor_name: str, nodes: Dict[str, CommitNode], tar_ds_1, tar_ds_2,
                  commits_id: Dict[str, str]):
    """Function to return the direct difference between target_1 and target_2."""
    tar_1_ids = tar_ds_1.get_tensor_uuids(tensor_name, commits_id['tar_1'])
    target_1_id_to_index_map = {id: idx for idx, id in enumerate(tar_1_ids)}

    tar_2_ids = tar_ds_2.get_tensor_uuids(tensor_name, commits_id['tar_2'])
    target_2_id_to_index_map = {id: idx for idx, id in enumerate(tar_2_ids)}

    diff_list = defaultdict(dict)
    diff_list["added_values"] = get_uuids_values(tar_ds_2,
                                                 tensor_name,
                                                 set(tar_2_ids) - set(tar_1_ids),
                                                commits_id['tar_2'],
                                                 target_2_id_to_index_map)
    diff_list["removed_values"] = get_uuids_values(tar_ds_1,
                                                   tensor_name,
                                                   set(tar_1_ids) - set(tar_2_ids),
                                                   commits_id['tar_1'],
                                                   target_1_id_to_index_map)

    target2_update_commit_map = get_updates_commit_ids_for_node(
        tar_ds_2, tensor_name, nodes["target_2"], nodes["lca"]
    )
    target1_update_commit_map = get_updates_commit_ids_for_node(
        tar_ds_1, tensor_name, nodes["target_1"], nodes["lca"]
    )

    combined_keys = set(target2_update_commit_map.keys()).union(set(target1_update_commit_map.keys()))

    return _get_diff_list(diff_list,
                   list(combined_keys),
                   tar_2_ids,
                   tar_1_ids,
                   target_2_id_to_index_map,
                   target_1_id_to_index_map,
                   tar_ds_1,
                   tar_ds_2,
                   tensor_name)



def _get_diff_list(diff_list,
                   combined_updated_uuids,
                   tar_2_ids,
                   tar_1_ids,
                   target_2_id_to_index_map,
                   target_1_id_to_index_map,
                   tar_ds_1,
                   tar_ds_2,
                   tensor_name):
    for sample_uuid in combined_updated_uuids:
        if sample_uuid in set(tar_2_ids) - set(tar_1_ids) or sample_uuid in set(tar_1_ids) - set(tar_2_ids):
            continue # update priority is less than more and less
        if sample_uuid in tar_2_ids and sample_uuid in tar_1_ids:
            target2_index: int = target_2_id_to_index_map[sample_uuid]
            target1_index: int = target_1_id_to_index_map[sample_uuid]
            tar_1_value = tar_ds_1[tensor_name][target1_index].numpy(aslist=True)[0]
            tar_2_value = tar_ds_2[tensor_name][target2_index].numpy(aslist=True)[0]
            if tar_1_value != tar_2_value:
                diff_list["edited_values_tar1"][sample_uuid] = tar_1_value
                diff_list["edited_values_tar2"][sample_uuid] = tar_2_value

    return diff_list


def merge(
        dataset,
        target_id: str,
        conflict_resolution: dict,
        delete_removed_tensors: bool = False,
        force: bool = False,
):
    """
    Merge works by comparing the states of the dataset at the target commit and the current commit.
    The new tensors in the target are added.
    The deleted tensors in the target are removed if delete_removed_tensors is True.
    For the common tensors, we compare ids of the samples. The samples with newer ids are added to the dataset.
    For samples with the same ids, we compare the changes history of the sample and resolve conflicts according
        to the conflict_resolution argument.
    """

    commit_node_map = dataset.version_state["commit_node_map"]

    auto_checkout(dataset, flush_version_control_info=False)
    target_commit_id = sanitize_commit(target_id, dataset.version_state)
    target_commit_id = auto_commit_target_commit(
        dataset, target_commit_id, flush_version_control_info=False
    )
    nodes: Dict[str, CommitNode] = {}
    nodes["original"] = original_node = dataset.version_state["commit_node"]
    nodes["target"] = target_node = commit_node_map[target_commit_id]
    lca_id = get_lowest_common_ancestor(original_node, target_node)
    target_ds = create_read_copy_dataset(dataset, target_commit_id)

    if lca_id == target_commit_id:
        logging.info("No merge needed, target id is an ancestor of the current commit")
        return
    nodes["lca"] = commit_node_map[lca_id]
    (
        new_tensors,
        common_tensors,
        deleted_tensors,
        _,
    ) = get_new_common_deleted_tensors(dataset, target_ds, lca_id, force, False)

    merge_common_tensors(common_tensors, dataset, target_ds, nodes, conflict_resolution)
    copy_new_tensors(new_tensors, dataset, target_ds)
    delete_tensors(deleted_tensors, dataset, delete_removed_tensors)
    finalize_merge(dataset, nodes)


def get_new_common_deleted_tensors(
        dataset, target_ds, lca_id: str, force: bool, is_detect: bool,
) -> Tuple[Set[str], Set[str], Set[str], dict]:
    """Gets the names of tensors, that are new, common and deleted in the target commit"""
    original_tensors: Set[str] = set(dataset.tensors)
    if not CREATE_TENSOR_HIDDEN_UUID:
        original_tensors = original_tensors | {DATASET_UUID_NAME}
    check_id_tensors_exist(original_tensors, set(dataset.all_tensors_filtered()))
    target_tensors: Set[str] = set(target_ds.tensors)
    if not CREATE_TENSOR_HIDDEN_UUID:
        target_tensors = target_tensors | {DATASET_UUID_NAME}
    check_id_tensors_exist(target_tensors, set(target_ds.all_tensors_filtered()))

    return _get_new_common_deleted_tensors(dataset,
                                    lca_id,
                                    target_tensors,
                                    original_tensors,
                                    target_ds,
                                    is_detect,
                                    force)


def _get_new_common_deleted_tensors(dataset,
                                    lca_id,
                                    target_tensors,
                                    original_tensors,
                                    target_ds,
                                    is_detect,
                                    force):
    tensor_dict = {"lca_tensors": get_node_tensors(dataset, lca_id),
                   "new_tensors": target_tensors - original_tensors,
                   "common_tensors": target_tensors & original_tensors}
    # present in dataset at lca, but deleted or renamed in target
    tensor_dict["target_deleted_tensors"] = tensor_dict["lca_tensors"] - target_tensors
    # present in dataset at lca, but deleted or renamed in original
    tensor_dict["original_deleted_tensors"] = tensor_dict["lca_tensors"] - original_tensors

    target_changes = target_ds.diff(lca_id, as_dict=True)
    target_tensor_diff, _ = target_changes["tensor"]
    target_dataset_diff, _ = target_changes["dataset"]
    original_dataset_diff, _ = dataset.diff(lca_id, as_dict=True)["dataset"]
    target_renamed_tensors, _ = merge_renamed_deleted(target_dataset_diff)
    original_renamed_tensors, _ = merge_renamed_deleted(original_dataset_diff)
    if is_detect:
        conflict_tensors = detect_renamed_tensors(
            dataset=dataset,
            new_tensors=tensor_dict["new_tensors"],
            common_tensors=tensor_dict["common_tensors"],
            original_deleted_tensors=tensor_dict["original_deleted_tensors"],
            original_renamed_tensors=original_renamed_tensors,
            target_renamed_tensors=target_renamed_tensors,
        )
        tensor_dict["new_tensors"] = process_deleted_tensors(tensor_dict["new_tensors"],
                                                             tensor_dict["original_deleted_tensors"],
                                                             target_tensor_diff)
    else:
        conflict_tensors = None
        (tensor_dict["new_tensors"], tensor_dict["common_tensors"],
         tensor_dict["target_deleted_tensors"], tensor_dict["original_deleted_tensors"]) = (
            process_renamed_tensors(
            dataset=dataset,
            force=force,
            new_tensors=tensor_dict["new_tensors"],
            common_tensors=tensor_dict["common_tensors"],
            original_deleted_tensors=tensor_dict["original_deleted_tensors"],
            target_deleted_tensors=tensor_dict["target_deleted_tensors"],
            original_renamed_tensors=original_renamed_tensors,
            target_renamed_tensors=target_renamed_tensors,
        ))

        tensor_dict["new_tensors"] = process_deleted_tensors(tensor_dict["new_tensors"],
                                                             tensor_dict["original_deleted_tensors"],
                                                             target_tensor_diff)
    return (tensor_dict["new_tensors"],
            tensor_dict["common_tensors"],
            tensor_dict["target_deleted_tensors"],
            conflict_tensors)


def detect_renamed_tensors(
        dataset,
        new_tensors,
        common_tensors,
        original_deleted_tensors,
        original_renamed_tensors,
        target_renamed_tensors,
):
    """Function to detect renamed tensors and return the conflicts"""
    conflict_tensors: dict = {}
    for old_tensor, new_tensor in target_renamed_tensors.items():
        if new_tensor in new_tensors:
            if old_tensor in original_renamed_tensors:
                conflict_tensors['target'] = {old_tensor: new_tensor}
                conflict_tensors['original'] = {old_tensor: original_renamed_tensors[old_tensor]}
            elif old_tensor in original_deleted_tensors: # not rename but delete
                conflict_tensors['target'] = {old_tensor: new_tensor}
                conflict_tensors['original'] = {old_tensor: 'Deleted'}
        elif new_tensor in common_tensors: # original rename to another name, but create a new name in common
            if original_renamed_tensors.get(old_tensor) != new_tensor:
                conflict_tensors['target'] = {old_tensor: new_tensor}
                conflict_tensors['original'] = {old_tensor: original_renamed_tensors[old_tensor]}
    return conflict_tensors


def process_renamed_tensors(
        dataset,
        force,
        new_tensors,
        common_tensors,
        original_deleted_tensors,
        target_deleted_tensors,
        original_renamed_tensors,
        target_renamed_tensors,
):
    """Process renamed tensors."""
    for old_tensor, new_tensor in target_renamed_tensors.items():
        if new_tensor in new_tensors:
            if not force:
                if old_tensor in original_renamed_tensors:
                    raise MergeConflictError(
                        message=f"{old_tensor} was renamed in both branches. "
                                f"Rename tensors to the same name to resolve the conflict or use `force=True` "
                                f"to register {new_tensor} as a new tensor on current branch."
                    )
                if old_tensor in original_deleted_tensors:
                    raise MergeConflictError(
                        message=f"{old_tensor} was renamed to {new_tensor} in target but is missing from "
                                f"current branch. Use `force=True` to register {new_tensor} as a "
                                f"new tensor on current branch."
                    )
                new_tensors.discard(new_tensor)
                target_deleted_tensors.discard(old_tensor)
                dataset.rename_tensor(old_tensor, new_tensor)
                common_tensors.add(new_tensor)

        elif new_tensor in common_tensors:
            # no merge conflict if same tensor was renamed to same name on both branches
            if original_renamed_tensors.get(old_tensor) != new_tensor and not force:
                raise MergeConflictError(
                    message=f"{old_tensor} was renamed to {new_tensor} in target but another {new_tensor} exists "
                            f"on the current branch. Rename tensors to resolve the conflict or use `force=True` to "
                            f"merge {new_tensor} of target with {new_tensor} of current branch."
                )

        target_deleted_tensors.discard(old_tensor)
        original_deleted_tensors.discard(old_tensor)
    return new_tensors, common_tensors, target_deleted_tensors, original_deleted_tensors


def process_deleted_tensors(new_tensors, original_deleted_tensors, target_tensor_diff):
    """Process deleted tensors."""
    for tensor in original_deleted_tensors:
        if tensor in new_tensors:
            tensor_changed = False
            for commit_diff in target_tensor_diff:
                diff = commit_diff[tensor]
                if has_change(diff):
                    tensor_changed = True
                    break
            if not tensor_changed:
                new_tensors.discard(tensor)
    return new_tensors


def finalize_merge(dataset, nodes: Dict[str, CommitNode]):
    """Finalizes the merge operation by linking the nodes and subsequently committing."""
    original_node = nodes["original"]
    target_node = nodes["target"]
    original_node.merge_from(target_node)
    target_id = target_node.commit_id
    commit(dataset, f"Merge {target_id} into {dataset.branch}")


def get_node_tensors(dataset, target_id: str) -> Set[str]:
    """Gets the names of tensors present in the given commit"""
    meta_key = get_dataset_meta_key(target_id)
    dataset_meta = dataset.storage.get_muller_object(meta_key, DatasetMeta)
    lca_tensors: Set[str] = set(dataset_meta.visible_tensors)
    return lca_tensors


def auto_commit_target_commit(
        dataset, target_commit_id: str, flush_version_control_info: bool = True
) -> str:
    """Automatically commits the dataset at the target id if it is the head of a branch."""
    original_id = dataset.pending_commit_id
    original_branch = dataset.branch
    checkout(dataset, target_commit_id)
    auto_commit(
        dataset,
        f"Auto commit before merging into {original_branch}",
        True,
        flush_version_control_info=flush_version_control_info,
    )
    target_commit_id = dataset.pending_commit_id
    checkout(dataset, original_id)
    return target_commit_id


def get_updates_commit_ids_for_node(
        dataset, tensor_name: str, commit_node: Optional[CommitNode], lca_node: CommitNode
):
    """Get the updated commit ids."""
    updated_commit_map = defaultdict(list)
    current_node = commit_node
    tensor_key = dataset.version_state["tensor_names"][tensor_name]
    while current_node and current_node.commit_id != lca_node.commit_id:
        commit_id = current_node.commit_id
        if current_node.is_merge_node:
            # Use dataset.version_state to obtain the actual CommitNode object (not just a string)
            # of the merge_parent of the current node.
            merge_parent_node = dataset.version_state["commit_node_map"][current_node.merge_parent]
            changes = get_updates_commit_ids_for_node(
                dataset, tensor_key, merge_parent_node, lca_node
            )
            for idx in changes:
                updated_commit_map[idx].extend(changes[idx])
        else:
            diff = _get_tensor_commit_diff(dataset, tensor_key, commit_id)
            if diff is not None:
                data_updated = diff.updated_data
                uuids = dataset.get_tensor_uuids(tensor_key, current_node.commit_id)
                for idx in data_updated:
                    sample_id = uuids[idx]
                    updated_commit_map[sample_id].append(commit_id)
        current_node = current_node.parent
    return updated_commit_map


def _get_tensor_commit_diff(dataset, tensor_key, commit_id):
    diff_key = get_tensor_commit_diff_key(tensor_key, commit_id)
    diff: Optional[CommitDiff]
    try:
        diff = dataset.storage.get_muller_object(diff_key, CommitDiff)
    except KeyError:
        diff = None
    return diff


def delete_tensors(tensor_names: Set[str], dataset, delete_removed_tensors: bool):
    """Deletes tensors from the dataset if delete_removed_tensors is True."""
    if delete_removed_tensors:
        for tensor_name in tensor_names:
            try:
                dataset.delete_tensor(tensor_name)
            # tensor could have been renamed.
            except TensorDoesNotExistError:
                pass


def clear_tensors(tensor_names: Set[str], dataset):
    """Clear tensors."""
    for tensor_name in tensor_names:
        dataset[tensor_name].clear()


def copy_new_tensors(
        tensor_names: Set[str],
        dataset,
        target_dataset,
):
    """Copies tensors from the target_commit to the dataset."""
    copy_tensors(
        target_dataset,
        dataset,
        tensor_names,
    )


def get_tensors_chunk_info(tensor_names, dest_ds, src_ds):
    """Use get_items to obtain the chunk index and chunk set results in advance."""
    dest_enc_keys = set()
    chunk_map_keys = set()
    for tensor_name in tensor_names:
        enc_key = get_chunk_id_encoder_key(tensor_name, dest_ds.version_state["commit_id"])
        dest_enc_keys.add(enc_key)
        cur_node: Optional[CommitNode] = src_ds.version_state["commit_node"]
        while cur_node is not None:
            commit_id = cur_node.commit_id
            if commit_id == FIRST_COMMIT_ID:
                break
            chunk_map_key = get_tensor_commit_chunk_map_key(tensor_name, commit_id)
            chunk_map_keys.add(chunk_map_key)
            cur_node = cur_node.parent
    dest_ds.storage.get_items(keys=dest_enc_keys, ignore_key_error=True)
    src_ds.storage.get_items(keys=chunk_map_keys, ignore_key_error=True)


def handle_records(records, conflict_resolution
                   ) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]], List[int]]:
    """Function to handle records and generate the indexes."""
    new_indexes, del_indexes = handle_append_ids(records['app_ori_idx'], records['app_tar_idx'],
                                                 conflict_resolution["append"])
    delete_ori = records['delete_ori']
    delete_tar = records['delete_tar']
    original_id_to_index_map = records['original_id_to_index_map']
    target_id_to_index_map = records['target_id_to_index_map']
    copy_indexes, remove_indexes = find_pop_conflicts(delete_ori, delete_tar,
                                                      target_id_to_index_map, original_id_to_index_map,
                                                      conflict_resolution["pop"])
    new_indexes.extend(copy_indexes)
    updated_indexes = records['updated_indexes']
    detect_conflicts = records['detect_conflicts']
    resurrect_indexes, conflict_update_indexes = handle_update_ids(detect_conflicts, conflict_resolution["update"])
    new_indexes.extend(resurrect_indexes)
    del_indexes.extend(remove_indexes)
    append_indexes = sorted(set(new_indexes))
    return append_indexes, updated_indexes, conflict_update_indexes, del_indexes


def merge_common_tensors(
        tensor_names: Set[str],
        dataset,
        target_dataset,
        nodes: Dict[str, CommitNode],
        conflict_resolution: dict,
):
    """Merge common tensors."""
    check_common_tensor_mismatches(tensor_names, dataset, target_dataset)
    idxs = _get_idxs(dataset, tensor_names, nodes, conflict_resolution, target_dataset)

    new_samples_dict, updated_samples_dict, conflict_samples_dict, _, dataset_delete_indexes = (
        _obtain_sample_dicts(idxs, tensor_names, conflict_resolution))

    get_tensors_chunk_info(tensor_names, dataset, target_dataset)

    for tensor_name in tensor_names:
        merge_tensor_data(
            tensor_name,
            dataset,
            target_dataset,
            new_samples_dict,
            updated_samples_dict,
            conflict_samples_dict,
            conflict_resolution["update"],
        )
        dataset.storage.clear_target_upper_cache('uuids', nodes["original"].commit_id, tensor_name)
    if len(dataset_delete_indexes) > 0:
        dataset.pop(dataset_delete_indexes)
    dataset.storage.flush()


def _get_idxs(dataset, tensor_names, nodes, conflict_resolution, target_dataset):
    idxs = {}
    for tensor_name in tensor_names:
        records = dataset.storage.get_records_cache_merge(nodes["target"].commit_id, tensor_name,
                                                          nodes["original"].commit_id)
        if records:
            idxs_results = handle_records(records, conflict_resolution)
        else:
            idxs_results = find_new_updated_and_conflict_indexes(
                tensor_name, dataset, target_dataset, nodes, conflict_resolution
            )
        idxs[tensor_name] = idxs_results

    idxs = _get_new_idxs(idxs)
    return idxs


def _obtain_sample_dicts(idxs, tensor_names, conflict_resolution):
    new_samples_dict: Dict[str, List[int]] = {}
    updated_samples_dict: Dict[str, List[Tuple[int, int]]] = {}
    conflict_samples_dict: Dict[str, List[Tuple[int, int]]] = {}
    deleted_sample_dict: Dict[str, List[int]] = {}
    conflict_tensors = set()
    dataset_delete_indexes = []

    for tensor_name in tensor_names:
        (
            new_indexes,
            updated_indexes,
            conflict_indexes,
            delete_indexes,
        ) = idxs[tensor_name]
        new_samples_dict[tensor_name] = new_indexes
        updated_samples_dict[tensor_name] = updated_indexes
        if conflict_indexes:
            conflict_samples_dict[tensor_name] = conflict_indexes
            conflict_tensors.add(tensor_name)
        if delete_indexes:
            if tensor_name == DATASET_UUID_NAME:
                dataset_delete_indexes = delete_indexes
            else:
                deleted_sample_dict[tensor_name] = delete_indexes

    dataset_delete_indexes = _validate_dataset_delete_indexes(dataset_delete_indexes, deleted_sample_dict)

    if conflict_tensors and conflict_resolution["update"] is None:
        # There are conflicts and a conflict resolution strategy has not been specified, unable to merge
        raise MergeConflictError(conflict_tensors)

    return new_samples_dict, updated_samples_dict, conflict_samples_dict, deleted_sample_dict, dataset_delete_indexes


def _validate_dataset_delete_indexes(dataset_delete_indexes, deleted_sample_dict):
    if not dataset_delete_indexes and deleted_sample_dict:
        values_iter = iter(deleted_sample_dict.values())
        first_value = next(values_iter)
        are_values_equal = all(value == first_value for value in values_iter)
        if are_values_equal:
            dataset_delete_indexes = list(deleted_sample_dict.values())[0]
        else:
            raise ValueError(f"Pop in tensors are different, which is invalid, please recheck.")
    return dataset_delete_indexes


def check_common_tensor_mismatches(tensor_names: Set[str], dataset, target_dataset):
    """Checks common tensors for mismatches in htype, sample_compression and chunk_compression."""
    for tensor_name in tensor_names:
        target_meta = target_dataset[tensor_name].meta
        original_meta = dataset[tensor_name].meta
        original_details = {
            "htype": original_meta.htype or "generic",
            "sample_compression": original_meta.sample_compression,
            "chunk_compression": original_meta.chunk_compression,
        }
        target_details = {
            "htype": target_meta.htype or "generic",
            "sample_compression": target_meta.sample_compression,
            "chunk_compression": target_meta.chunk_compression,
        }
        for key, value in original_details.items():
            if value != target_details.get(key, None):
                raise MergeMismatchError(tensor_name, key, value, target_details.get(key, None))


def remove_update(
        append_new_ids,
        target_update_commit_map,
        append_indexes,
):
    """Remove updates."""
    if len(append_indexes) > 0: # conflict solution == "theirs" and has appending samples
        for temp_id in append_new_ids:  # new index in target compared with lca
            target_update_commit_map.pop(temp_id, None) # append priority is higher than update


def find_pop_conflicts(
        delete_idx_ori,
        delete_idx_tar,
        target_id_to_index_map,
        original_id_to_index_map,
        conflict_resolution
) -> Tuple[List[int], List[int]]:
    """Finds the popped changes conflicts between the original commit and target id.

    Args:
        delete_idx_ori: A set of uuids of popped samples in original node, excludes both popped samples.
        delete_idx_tar: A set of uuids of popped samples in target node, excludes both popped samples.
        target_id_to_index_map: A dictionary mapping sample ids to a list of commit ids that updated the sample.
        original_id_to_index_map: A dictionary mapping sample ids to a list of commit ids that updated the sample.
        conflict_resolution: The strategy to use to resolve merge conflicts.

    Returns:
        popped_indexes_ori
    """
    copy_indexes: List[int] = [] # new indexes
    remove_indexes: List[int] = []
    if delete_idx_ori or delete_idx_tar:
        if conflict_resolution is None:
            raise MergeConflictError(message="Unable to merge, because original dataset and target dataset deleted the "
                                             "different samples, without any pop resolution argument provided."
                                             "Please consider using pop_resolution='ours' or pop_resolution"
                                             "='theirs' or pop_resolution='both' to solve the conflict.")
        if conflict_resolution == "theirs": # if theirs, append pop history in original and pop the delete ids in target
            for idx in delete_idx_ori:
                try:
                    # compared with lca, only popped in original, so we can get it in target.
                    copy_back_idx: int = target_id_to_index_map[idx]
                    copy_indexes.append(copy_back_idx)
                except KeyError as e:
                    raise MergeLostUUid from e
                    # compared with lca, only popped in target, so we can get it in original, and popped it in original.
            for idx in delete_idx_tar:
                try:
                    remove_idx: int = original_id_to_index_map[idx]
                    remove_indexes.append(remove_idx)
                except KeyError as e:
                    raise MergeLostUUid from e
        elif conflict_resolution == "both":
            for idx in delete_idx_tar:
                try:
                    remove_idx: int = original_id_to_index_map[idx]
                    remove_indexes.append(remove_idx)
                except KeyError as e:
                    raise MergeLostUUid from e
    return copy_indexes, remove_indexes


def handle_append_ids(
        append_ids_ori,
        append_ids_tar,
        conflict_resolution,
) -> Tuple[List[int], List[int]]:
    """Function to generate new indexes and delete indexes according to conflict solution."""
    new_indexes: List[int] = []
    delete_indexes: List[int] = []
    if append_ids_ori and append_ids_tar: # means 3 way merge
        if conflict_resolution is None:
            raise MergeConflictError(message="Unable to merge, because original dataset and target dataset append the "
                                             "different samples, without any append_resolution argument provided."
                                             "Please consider using append_resolution='ours' or append_resolution"
                                             "='theirs' or append_resolution='both' to solve the conflict.")
        if conflict_resolution == "theirs":
            new_indexes = append_ids_tar
            delete_indexes = append_ids_ori
        elif conflict_resolution == "both":
            new_indexes = append_ids_tar
    elif append_ids_tar:
        new_indexes = append_ids_tar
    return new_indexes, delete_indexes


def find_append_conflicts(
        app_uuids_ori,
        app_uuids_tar,
        target_id_to_index_map,
        original_id_to_index_map,
        conflict_resolution,
) -> Tuple[List[int], List[int]]:
    """Finds the append changes conflicts between the original commit and target id.

    Args:
        app_uuids_ori: A set of uuids of appended samples in original node, excludes both appended samples.
        app_uuids_tar: A set of uuids of appended samples in target node, excludes both appended samples.
        target_id_to_index_map: A dictionary mapping sample ids to a list of commit ids that updated the sample.
        original_id_to_index_map: A dictionary mapping sample ids to a list of commit ids that updated the sample.
        conflict_resolution: The strategy to use to resolve merge conflicts.

    Returns:
        new_indexes, delete_indexes
    """
    new_indexes: List[int] = []
    delete_indexes: List[int] = []
    if app_uuids_tar and app_uuids_ori: # means it must be 3-way-merge
        if conflict_resolution is None:
            raise MergeConflictError(message="Unable to merge, because original dataset and target dataset append the "
                                             "different samples, without any conflict resolution argument provided."
                                             "Please consider using append_resolution='ours' or append_resolution"
                                             "='theirs' or append_resolution='both' to solve the conflict.")
        if conflict_resolution == "theirs": # if theirs, delete the original appended samples, append target samples.
            for idx in app_uuids_tar:
                try:
                    new_app_idx: int = target_id_to_index_map[idx]
                    new_indexes.append(new_app_idx)
                except KeyError as e:
                    raise MergeLostUUid from e
            for idx in app_uuids_ori:
                try:
                    delete_app_idx: int = original_id_to_index_map[idx]
                    delete_indexes.append(delete_app_idx)
                except KeyError as e:
                    raise MergeLostUUid from e
        elif conflict_resolution == "both":
            for idx in app_uuids_tar:
                try:
                    new_app_idx: int = target_id_to_index_map[idx]
                    new_indexes.append(new_app_idx)
                except KeyError as e:
                    raise MergeLostUUid from e
    elif app_uuids_tar:
        for idx in app_uuids_tar:
            try:
                new_app_idx: int = target_id_to_index_map[idx]
                new_indexes.append(new_app_idx)
            except KeyError as e:
                raise MergeLostUUid from e
    return new_indexes, delete_indexes


def find_new_updated_and_conflict_indexes(
        tensor_name: str,
        dataset,
        target_dataset,
        nodes: Dict[str, CommitNode],
        conflict_resolution: dict,
) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]], List[int]]:
    """Finds the new, deleted, updated and conflict indexes between the original commit and target commit.

    Args:
        tensor_name (str): The name of the tensor to find the new and conflict indexes for.
        dataset: The original state of the dataset.
        target_dataset: The target state of the dataset.
        nodes (dict): A dictionary containing original, target and lca nodes.
        conflict_resolution (str, Optional): The strategy to use to resolve merge conflicts.

    Returns:
        A tuple of the form (new_indexes, updated_indexes, conflict_indexes, deleted_indexes)
        - new_indexes is a list of indexes for new samples
        - updated_indexes is a list of tuples of the form (original_idx, target_idx)
        - conflict_indexes is a list of tuples of the form (original_idx, target_idx)
        - deleted_indexes is a list of indexes for popped samples
    """

    target_update_commit_map = get_updates_commit_ids_for_node(
        target_dataset, tensor_name, nodes["target"], nodes["lca"]
    )

    delete_ori, delete_tar, append_ori, append_tar, index_map_dict = (
        _get_diff_in_delete_and_appends(dataset,
                                        dataset.version_state["tensor_names"][tensor_name],
                                        nodes["original"],
                                        target_dataset,
                                        nodes["target"],
                                        nodes["lca"]))

    return _get_conflict_indexes(dataset,
                          append_ori,
                          append_tar,
                          delete_ori,
                          delete_tar,
                          index_map_dict,
                          conflict_resolution,
                          target_update_commit_map,
                          tensor_name,
                          nodes)


def _get_conflict_indexes(dataset,
                          append_ori,
                          append_tar,
                          delete_ori,
                          delete_tar,
                          index_map_dict,
                          conflict_resolution,
                          target_update_commit_map,
                          tensor_name,
                          nodes):

    # if conflict = theirs, append_indexes contains the indexes need copy from target,
    # delete_indexes contains the indexes need remove from original.
    try:
        append_indexes, deleted_indexes = find_append_conflicts(append_ori,
                                                                append_tar,
                                                                index_map_dict.get("target_id_to_index_map"),
                                                                index_map_dict.get("original_id_to_index_map"),
                                                                conflict_resolution["append"])
        index_dict = {"append_indexes": append_indexes,
                      "deleted_indexes": deleted_indexes}
        # if conflict = theirs, copy_indexes contains the indexes need copy from target,
        # remove_indexes contains the indexes need remove from original.
        copy_indexes, remove_indexes = find_pop_conflicts(delete_ori,
                                                          delete_tar,
                                                          index_map_dict.get("target_id_to_index_map"),
                                                          index_map_dict.get("original_id_to_index_map"),
                                                          conflict_resolution["pop"])
        index_dict["append_indexes"].extend(copy_indexes)

        # The priority of append is higher than update.
        remove_update(append_tar, target_update_commit_map, index_dict.get("append_indexes"))

        updated_indexes, conflict_indexes, resurrect_indexes, _ = (
            detect_update_conflicts(original_update_commit_map=get_updates_commit_ids_for_node(dataset,
                                                                                               tensor_name,
                                                                                               nodes["original"],
                                                                                               nodes["lca"]),
                                    target_update_commit_map=target_update_commit_map,
                                    original_id_to_index_map=index_map_dict.get("original_id_to_index_map"),
                                    target_id_to_index_map=index_map_dict.get("target_id_to_index_map"),
                                    conflict_resolution=conflict_resolution["update"],
                                    is_detect=False
                                    ))
        index_dict["append_indexes"].extend(resurrect_indexes)
        index_dict["deleted_indexes"].extend(remove_indexes)
    except KeyError as e:
        raise Exception from e

    return (sorted(set(index_dict.get("append_indexes", None))),
            updated_indexes,
            conflict_indexes,
            sorted(set(index_dict.get("deleted_indexes", None))))


def get_deleted_ids(original_all_ids, target_all_ids, lca_all_ids):
    """Return the deleted ids in the target branch."""
    deleted_ids_in_target = set(lca_all_ids) - set(target_all_ids)  # target deleted in lca
    deleted_ids_in_original = set(lca_all_ids) - set(original_all_ids)  # original deleted in lca
    deleted_in_both = deleted_ids_in_target & deleted_ids_in_original  # both deleted in lca
    deleted_ids_in_original = deleted_ids_in_original - deleted_in_both  # only deleted in original
    deleted_ids_in_target = deleted_ids_in_target - deleted_in_both  # only deleted in target
    return deleted_ids_in_original, deleted_ids_in_target, deleted_in_both


def get_append_ids(original_all_ids, target_all_ids, lca_all_ids):
    """Function to get the append uuids in target and original node."""
    append_ids_in_target = set(target_all_ids) - set(lca_all_ids) # target append compared with lca
    append_ids_in_original = set(original_all_ids) - set(lca_all_ids) # original append compared with lca
    return append_ids_in_original, append_ids_in_target


def merge_tensor_data(
        tensor_name: str,
        dataset,
        target_dataset,
        new_samples_dict,
        updated_samples_dict,
        conflict_samples_dict,
        conflict_resolution,
):
    """Merges actual data present in 2 versions of a common tensor."""
    if conflict_resolution == "theirs" and tensor_name in conflict_samples_dict: # apply target update
        updated_samples_dict[tensor_name].extend(conflict_samples_dict[tensor_name])

    original_tensor = dataset[tensor_name]
    target_tensor = target_dataset[tensor_name]

    new_indexes = new_samples_dict[tensor_name]
    new_indexes.sort()
    original_tensor, target_class_names, is_class_label = _get_original_tensor(target_tensor,
                                                                               original_tensor,
                                                                               new_indexes)

    copy_tensor_slice(
        target_dataset,
        dataset,
        tensor_name,
        tensor_name,
        new_indexes,
    )
    for original_idx, target_idx in updated_samples_dict[tensor_name]:
        sample = target_tensor[target_idx]
        if is_class_label and target_class_names:
            sample = convert_to_text(
                sample.numpy(), target_class_names, return_original=True
            )
        original_tensor[original_idx] = sample


def _get_original_tensor(target_tensor, original_tensor, new_indexes):
    target_class_names = None
    is_class_label = target_tensor.meta.htype == "class_label"
    copy_class_labels = is_class_label
    if is_class_label:
        target_class_names = target_tensor.info.class_names
        original_class_names = original_tensor.info.class_names
        if target_class_names:
            if target_class_names == original_class_names:
                copy_class_labels = False
            elif original_class_names[: len(target_class_names)] == target_class_names:
                copy_class_labels = False
            elif (
                    target_class_names[: len(original_class_names)] == original_class_names
            ):
                copy_class_labels = False
                original_tensor.info.class_names = original_class_names
        else:
            copy_class_labels = False

    if copy_class_labels:
        # Sherry: Need to optimize this
        with original_tensor.dataset:
            for index in new_indexes:
                sample = target_tensor[index]
                sample = convert_to_text(
                    sample.numpy(), target_class_names, return_original=True
                )
                original_tensor.append(sample)
    return original_tensor, target_class_names, is_class_label


def check_id_tensors_exist(visible_tensors: Set[str], all_tensors: Set[str]):
    """Checks whether hidden id tensors exist for each tensor."""

    """
    for tensor_name in visible_tensors:
        id_tensor = get_sample_id_tensor_key(tensor_name)
        if id_tensor not in all_tensors:
            raise MergeNotSupportedError
    """
    pass


def _get_meta_files_for_tensor(tensor_name, commit_id, split_tensor_meta):
    fns = [
        get_chunk_id_encoder_key,
        get_tensor_tile_encoder_key,
        get_creds_encoder_key,
    ]
    if split_tensor_meta:
        fns.append(get_tensor_meta_key)
    return [fn(tensor_name, commit_id) for fn in fns]


def _get_chunks_for_tensor(src_tensor, dest_commit_id, dest_key):
    eng = src_tensor.chunk_engine
    enc = eng.chunk_id_encoder

    chunkids = enc.encoded[:, 0]
    ret = []
    for cid in chunkids:
        cname = enc.name_from_id(cid)
        return_commit, key = _get_chunk_commit(eng, cname)
        same_commit = return_commit == dest_commit_id
        same_key = key == dest_key
        if same_commit and same_key:
            ret.append((cname,))
        elif same_key:
            ret.append((cname, return_commit))
        else:
            ret.append((cname, return_commit, key))
    return ret


def _copy_objects(key_pairs, src_storage, dest_storage):
    for src_key, dest_key in zip(*key_pairs):
        try:
            dest_storage[dest_key] = src_storage[src_key]
        except KeyError:
            pass  # This must be a pass; otherwise, raising a KeyError here would cause an error.


def copy_tensors(
        src_ds,
        dest_ds,
        src_tensor_names,
        dest_tensor_names=None,
):
    """Copy tensors."""
    if not src_tensor_names:
        return
    if not src_ds.read_only:
        src_ds.flush()
    dest_ds.flush()
    src_tensor_names = list(src_tensor_names)

    dest_ds_meta = dest_ds.meta
    hidden_tensors = []
    dest_new_ori = {}
    src_tensor_names += hidden_tensors
    if dest_tensor_names is None:
        dest_tensor_names = src_tensor_names
    else:
        if len(src_tensor_names) != len(dest_tensor_names):
            raise Exception
    for dest_tensor_name in dest_tensor_names:
        if src_ds.version_state.get('tensor_names', {}).get(dest_tensor_name, {}):
            dest_new_ori[dest_tensor_name] = src_ds.version_state.get('tensor_names', {}).get(dest_tensor_name, {})

    src_keys, dest_keys = _get_src_and_dst_keys(src_ds,
                          src_tensor_names,
                          dest_ds,
                          dest_tensor_names)

    _copy_objects((src_keys, dest_keys), src_ds.base_storage, dest_ds.base_storage)
    dest_ds_meta.tensors += dest_tensor_names
    if dest_new_ori:
        dest_ds_meta.tensor_names.update({k: dest_new_ori.get(k, k) for k in dest_new_ori})
    else:
        dest_ds_meta.tensor_names.update({k: k for k in dest_tensor_names})
    dest_ds_meta.hidden_tensors += hidden_tensors
    dest_ds.base_storage[get_dataset_meta_key(dest_ds.pending_commit_id)] = dest_ds_meta.tobytes()
    dest_ds.storage.clear_cache_without_flush()
    dest_ds.populate_meta()


def _get_src_and_dst_keys(src_ds,
                          src_tensor_names,
                          dest_ds,
                          dest_tensor_names):
    src_keys = []
    dest_keys = []
    if not dest_ds.split_tensor_meta:
        src_keys.append(get_tensor_meta_key("", src_ds.pending_commit_id))
        dest_keys.append(get_tensor_meta_key("", dest_ds.pending_commit_id))

    for src_tensor_name, dest_tensor_name in zip(src_tensor_names, dest_tensor_names):
        chunks = _get_chunks_for_tensor(src_ds[src_tensor_name], dest_ds.pending_commit_id, dest_tensor_name)
        dest_chunk_map_key = get_tensor_commit_chunk_map_key(
            dest_tensor_name, dest_ds.pending_commit_id
        )
        dest_chunk_map = CommitChunkMap()
        for chunk in chunks:
            dest_chunk_map.add(*chunk)
        dest_ds.base_storage[dest_chunk_map_key] = dest_chunk_map.tobytes()
        src_keys += _get_meta_files_for_tensor(src_ds[src_tensor_name].key,
                                               src_ds.pending_commit_id,
                                               dest_ds.split_tensor_meta)
        dest_keys += _get_meta_files_for_tensor(dest_tensor_name,
                                                dest_ds.pending_commit_id,
                                                dest_ds.split_tensor_meta)
        dest_commit_diff = CommitDiff(0,
                                      True,
                                      commit_id=dest_ds.pending_commit_id,
                                      storage=dest_ds.storage)
        dest_commit_diff.add_data(src_ds[src_tensor_name].meta.length)
        dest_commit_diff_key = get_tensor_commit_diff_key(
            dest_tensor_name, dest_ds.pending_commit_id
        )
        dest_ds.base_storage[dest_commit_diff_key] = dest_commit_diff.tobytes()

    return src_keys, dest_keys


def _group_ranges(x):
    ret = []
    s = x[0]
    e = s + 1
    for i in range(1, len(x)):
        xi = x[i]
        if xi == e:
            e += 1
        else:
            ret.append((s, e))
            s = xi
            e = s + 1
    ret.append((s, e))
    return ret


def _merge_encodings(enc1, enc2, start, end, off1=None, off2=None):
    n1 = len(enc1)
    if not n1:
        return enc2[start:end]
    n2 = len(enc2)
    if not n2:
        return enc1
    if off1 is not None:
        old_offset = off1
    elif start == 0:
        old_offset = 0
    else:
        old_offset = enc2[start - 1, -1:] + 1
    new_offset = enc1[-1, -1:] + 1
    if enc1[-1, 0] == enc2[start, 0]:
        enc1 = enc1[:-1]
    ret = np.concatenate([enc1, enc2[start:end]], axis=0)
    ret[n1:, -1] += new_offset - old_offset
    if off2 is not None:
        ret[-1, -1] = off2 - 1 + new_offset - old_offset
    return ret


def _get_required_chunks_for_range(tensor, start, end):
    eng = tensor.chunk_engine
    enc = eng.chunk_id_encoder
    arr = enc.encoded
    start_row = enc.translate_index(start)
    end_row = enc.translate_index(end - 1)
    last_index = arr[end_row, 1]
    nrows = len(arr)
    nxt = end_row + 1
    while nxt < nrows and arr[nxt, 1] == last_index:
        end_row = nxt
        nxt += 1
    num_required_chunks = end_row + 1 - start_row
    start_chunk_aligned = False
    end_chunk_aligned = False
    if start_row == 0:
        if start == 0:
            start_chunk_aligned = True
    else:
        prev_row = start_row - 1
        if start == arr[prev_row, 1] + 1:
            start_chunk_aligned = True

    if arr[end_row, 1] == end - 1:
        end_chunk_aligned = True

    if num_required_chunks == 1:
        if not (start_chunk_aligned and end_chunk_aligned):
            return None, (start, end), None
        return (start_row, start_row + 1), None, None

    if num_required_chunks == 2:
        return _get_required_chunks_from_2_chunks(start_chunk_aligned, end_chunk_aligned, start_row, end_row, start,
                                                  end, arr)

    if start_chunk_aligned and not end_chunk_aligned:
        return (start_row, end_row), None, (int(arr[end_row - 1, 1] + 1), end)

    if end_chunk_aligned and not start_chunk_aligned:
        return (start_row + 1, end_row + 1), (start, int(arr[start_row, 1] + 1)), None

    if not start_chunk_aligned and not end_chunk_aligned:
        return (
            (start_row + 1, end_row),
            (start, int(arr[start_row, 1] + 1)),
            (int(arr[end_row - 1, 1] + 1), end),
        )

    return (start_row, end_row + 1), None, None


def _get_required_chunks_from_2_chunks(start_chunk_aligned, end_chunk_aligned, start_row, end_row, start, end, arr):
    if start_chunk_aligned and end_chunk_aligned:
        return (start_row, end_row + 1), None, None
    if not start_chunk_aligned and not end_chunk_aligned:
        return None, (start, end), None
    if start_chunk_aligned:
        return (start_row, start_row + 1), None, (int(arr[start_row, 1] + 1), end)
    return (end_row, end_row + 1), (start, int(arr[start_row, 1] + 1)), None


@contextmanager
def _as_flat_tensors(*tensors):
    yield


def _copy_samples(src_tensor, dest_tensor, start: int, end: int):
    with _as_flat_tensors(src_tensor, dest_tensor):
        dest_tensor.extend(src_tensor[start:end])


def _merge_tile_encoders(
        src_tile_encoder, dest_tile_encoder, start: int, end: int
) -> None:
    src_entries = src_tile_encoder.entries
    dest_entries = dest_tile_encoder.entries
    for i in range(start, end):
        e = src_entries.get(i)
        if e:
            dest_entries[i] = e
            dest_tile_encoder.is_dirty = True


def _setup_chunk_pointers(
        src_eng,
        src_enc_arr,
        dest_enc,
        dest_chunk_map,
        dest_commit,
        dest_key,
        start: int,
        end: int,
):
    chunk_ids = src_enc_arr[start:end, 0]
    chunk_names = list(map(ChunkIdEncoder.name_from_id, chunk_ids))
    _get_chunk_commit_with_chunk_engine = partial(_get_chunk_commit, src_eng)
    listed_commit_key_pairs = list(map(_get_chunk_commit_with_chunk_engine, chunk_names))
    for chunk_name, (listed_commit, key) in zip(chunk_names, listed_commit_key_pairs):
        if listed_commit == dest_commit:
            listed_commit = None
        elif key == dest_key:
            key = None
        dest_chunk_map.add(chunk_name, listed_commit, key)
    dest_enc.encoded = _merge_encodings(dest_enc.encoded, src_enc_arr, start, end)
    dest_enc.is_dirty = True


def copy_tensor_slice(
        src_ds,
        dest_ds,
        src_tensor_name,
        dest_tensor_name,
        indices=None,
        ranges=None,
):
    """Copy tensor slices."""
    if not ranges:
        if not indices:
            return
        ranges = _group_ranges(indices)
    src_tensor = src_ds[src_tensor_name]
    dest_tensor = dest_ds[dest_tensor_name]

    if dest_tensor.meta.dtype is None:
        dest_tensor.meta.dtype = src_tensor.meta.dtype
    if dest_tensor.meta.htype is None:
        dest_tensor.meta.htype = src_tensor.meta.htype
    dest_meta_orig_length = dest_tensor.meta.length
    dest_meta_length = (
        len(indices) if indices else sum(end - start for start, end in ranges)
    )

    for start, end in ranges:
        if src_tensor.chunk_engine.enable_tile_encoder and dest_tensor.chunk_engine.enable_tile_encoder:
            _merge_tile_encoders(src_tensor.chunk_engine.tile_encoder,
                                 dest_tensor.chunk_engine.tile_encoder,
                                 start,
                                 end)
        if not _handle_chunks_in_ranges(src_tensor,
                                 src_tensor_name,
                                 dest_tensor,
                                 dest_tensor_name,
                                 start,
                                 end,
                                 dest_ds
                                 ):
            return
    if src_tensor.meta.min_shape:
        dest_tensor.meta.update_shape_interval(src_tensor.meta.min_shape)
        dest_tensor.meta.update_shape_interval(src_tensor.meta.max_shape)
    dest_tensor.meta.length = dest_meta_orig_length + dest_meta_length
    dest_tensor.meta.is_dirty = True


def _handle_chunks_in_ranges(src_tensor,
                             src_tensor_name,
                             dest_tensor,
                             dest_tensor_name,
                             start,
                             end,
                             dest_ds
                             ):
    (
        chunks_to_copy,
        left_edge_samples,
        right_edge_samples,
    ) = _get_required_chunks_for_range(src_tensor, start, end)
    if left_edge_samples:
        s, e = left_edge_samples
        if src_tensor_name == dest_tensor_name == DATASET_UUID_NAME:
            return 0
        _copy_samples(src_tensor, dest_tensor, s, e)
    if chunks_to_copy:
        _setup_chunk_pointers(
            src_tensor.chunk_engine,
            src_tensor.chunk_engine.chunk_id_encoder.encoded,
            dest_tensor.chunk_engine.chunk_id_encoder,
            dest_tensor.chunk_engine.commit_chunk_map,
            dest_ds.pending_commit_id,
            dest_tensor.key,
            *chunks_to_copy,
        )
    if right_edge_samples:
        s, e = right_edge_samples
        if src_tensor_name == dest_tensor_name == DATASET_UUID_NAME:
            return 0
        _copy_samples(src_tensor, dest_tensor, s, e)
    return 1


def _get_chunk_commit(chunk_engine, chunk_name) -> Tuple[str, str]:
    """Returns the commit id and tensor key that contains the chunk_name."""
    cur_node: Optional[CommitNode] = chunk_engine.version_state["commit_node"]
    key = chunk_engine.key
    while cur_node is not None:
        commit_id = cur_node.commit_id
        chunk_map_key = get_tensor_commit_chunk_map_key(key, commit_id)
        try:
            # the first commit doesn't contain a chunk map, don't repeatedly try to fetch from storage
            if commit_id == FIRST_COMMIT_ID:
                chunk_map = dict()
            else:
                chunk_map = chunk_engine.meta_cache.get_muller_object(chunk_map_key, CommitChunkMap).chunks
        except Exception:
            commit_chunk_map = CommitChunkMap()
            try:
                chunk_engine.meta_cache[chunk_map_key] = commit_chunk_map
            except ReadOnlyModeError:
                chunk_engine.meta_cache.muller_objects[chunk_map_key] = commit_chunk_map
            chunk_map = dict()
        v = chunk_map.get(chunk_name)
        if v is not None:
            commit_id = v.get("commit_id", commit_id)
            key = v.get("key", key)
            return commit_id, key
        cur_node = cur_node.parent  # type: ignore
    # the first commit doesn't have a commit chunk map, so any chunk that wasn't found belongs to the first commit
    return FIRST_COMMIT_ID, key


def _get_new_idxs(idxs):
    all_new_idxs = set()
    for new_idxs, _, _, _ in idxs.values():
        all_new_idxs.update(new_idxs)
    return idxs
