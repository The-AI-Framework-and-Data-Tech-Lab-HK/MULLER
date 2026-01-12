# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/diff.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import OrderedDict, defaultdict

import math
import numpy as np

from muller.constants import DEFAULT_MAX_CHUNK_SIZE, DEFAULT_TILING_THRESHOLD
from muller.core.chunk.base_chunk import BaseChunk
from muller.core.chunk.chunk_compressed_chunk import ChunkCompressedChunk
from muller.core.chunk.sample_compressed_chunk import SampleCompressedChunk
from muller.core.chunk.uncompressed_chunk import UncompressedChunk
from muller.core.meta.dataset_meta import DatasetMeta
from muller.core.meta.encode.chunk_id import ChunkIdEncoder
from muller.core.meta.tensor_meta import TensorMeta
from muller.core.storage import LRUCache
from muller.core.version_control.commit_diff import CommitDiff
from muller.core.version_control.commit_node import CommitNode
from muller.core.version_control.dataset_diff import DatasetDiff
from muller.util.keys import get_tensor_commit_diff_key, get_dataset_meta_key, get_dataset_diff_key, get_chunk_key, \
    get_tensor_meta_key


def _get_chunk(storage, chunk_id, commit_id, tensor):
    chunk_name = ChunkIdEncoder.name_from_id(int(chunk_id))
    tar_chunk_key = get_chunk_key(tensor, chunk_name)
    chunk_cache = storage.get_item_from_cache(tar_chunk_key)
    if chunk_cache and isinstance(chunk_cache, BaseChunk):
        return chunk_cache
    tar_chunk = _get_chunk_from_bytes(storage, tensor, commit_id, tar_chunk_key)
    return tar_chunk


def _get_chunk_from_bytes(storage, tensor, commit_id, tar_chunk_key):
    meta = storage.get_muller_object(get_tensor_meta_key(tensor, commit_id), TensorMeta)
    if meta.chunk_compression:
        expected_class = ChunkCompressedChunk
        compression = meta.chunk_compression
    elif meta.sample_compression:
        expected_class = SampleCompressedChunk
        compression = meta.sample_compression
    else:
        expected_class = UncompressedChunk
        compression = None
    max_chunk_size = meta.max_chunk_size if meta.max_chunk_size else DEFAULT_MAX_CHUNK_SIZE
    tiling_threshold = meta.tiling_threshold if meta.tiling_threshold else DEFAULT_TILING_THRESHOLD
    chunk_args = [
        max_chunk_size // 2,
        max_chunk_size,
        tiling_threshold,
        meta,
        compression
    ]
    tar_chunk = storage.get_muller_object(tar_chunk_key, expected_class=expected_class, meta=chunk_args)
    if isinstance(tar_chunk, BaseChunk):
        return tar_chunk
    raise ValueError(f"Can not load chunk through this path: {tar_chunk_key}")


def handle_append_ranges(pairs_sorted, index_sorted, append_ranges):
    """Handle the append ranges. """
    left_side = append_ranges[0]
    right_side = append_ranges[1] - 1
    append_records = defaultdict(list)
    if right_side < left_side:  # no samples added
        return append_records
    if index_sorted[-1] >= right_side:
        lower_bound = int(
            np.searchsorted(index_sorted, left_side))  # locate the left index of chunks in chunk_id_encoder
        if lower_bound > 0:  # not the first chunk
            lower_left = left_side - int(index_sorted[lower_bound - 1]) - 1
        else:  # it is the first chunk
            lower_left = left_side
        up_bound = int(
            np.searchsorted(index_sorted, right_side))  # locate the right index of chunks in chunk_id_encoder
        if up_bound > 0:
            up_right = right_side - int(index_sorted[up_bound - 1]) - 1
        else:
            up_right = right_side
    else:
        raise ValueError(f"The recodes of data appended in CommitDiff is out of range, please recheck.")
    if up_bound == lower_bound:  # samples in one chunk situation
        append_records[str(pairs_sorted[lower_bound, 0])] = [lower_left, up_right]
        return append_records

    for i in range(lower_bound, up_bound + 1):
        if i == lower_bound:
            append_records[str(pairs_sorted[i, 0])] = [lower_left, None]
        elif i == up_bound:
            append_records[str(pairs_sorted[i, 0])] = [None, up_right]
        else:
            append_records[str(pairs_sorted[i, 0])] = [None, None]
    return append_records


def handle_update_ranges(pairs_sorted, index_sorted, update_ranges):
    """Handle the update ranges. """
    update_records = defaultdict(list)
    for update_idx in update_ranges:
        if index_sorted[-1] >= update_idx:
            tar_chunk_idx = int(np.searchsorted(index_sorted, update_idx))
            if tar_chunk_idx == 0:
                lid = update_idx
            else:
                lid = update_idx - int(index_sorted[tar_chunk_idx - 1]) - 1
            update_records[str(pairs_sorted[tar_chunk_idx, 0])].append(lid)
        else:
            raise ValueError(f"The recodes of data updated in CommitDiff is out of range, please recheck.")
    return update_records


def handle_delete_ranges(par_indexes, par_chunk_gid, par_uuids, delete_ranges):
    """Handle the delete ranges. """
    delete_records = defaultdict(list)
    gid_par = set()
    for uuid in delete_ranges:
        try:
            global_idx_par = par_uuids.index(uuid)
        except ValueError as e:
            raise ValueError(f"The deleted uuid {uuid} is not existed in its "
                             f"parent version: {par_uuids}") from e
        gid_par.add(global_idx_par)

    for gid in gid_par:
        if par_indexes[-1] >= gid:
            tar_chunk_idx = int(np.searchsorted(par_indexes, gid))
            if tar_chunk_idx == 0:
                lid = gid
            else:
                lid = gid - int(par_indexes[tar_chunk_idx - 1]) - 1
            delete_records[str(par_chunk_gid[tar_chunk_idx, 0])].append(lid)
        else:
            raise ValueError(f"The recodes of data deleted in CommitDiff is out of range, please recheck.")
    return delete_records, gid_par


def generate_add_values(storage, tensor, commit_id, append_records):
    """Generate the added values. """
    all_add_sample = []
    for chunk_id, local_range in append_records.items():
        tar_chunk = _get_chunk(storage, chunk_id, commit_id, tensor)
        start_idx = local_range[0] if local_range[0] else 0
        end_idx = local_range[1] if local_range[1] else tar_chunk.num_samples - 1
        for idx in range(start_idx, end_idx + 1):
            sample_data = tar_chunk.read_sample(idx, decompress=True)
            all_add_sample.append(sample_data)
    return all_add_sample


def generate_update_values(storage, tensor, commit_id, update_records):
    """Function to generate update values according to update records."""
    all_update_sample = []
    for chunk_id, local_idx in update_records.items():
        tar_chunk = _get_chunk(storage, chunk_id, commit_id, tensor)
        for idx in local_idx:
            sample_data = tar_chunk.read_sample(idx, decompress=True)
            all_update_sample.append(sample_data)
    return all_update_sample


def generate_delete_values(storage, tensor, par_commit_id, delete_records):
    """Function to generate delete values according to delete records."""
    all_deleted_sample = []
    for chunk_id, local_idx in delete_records.items():
        chunk_name = ChunkIdEncoder.name_from_id(int(chunk_id))
        par_chunk_key = get_chunk_key(tensor, chunk_name)
        tar_chunk = storage[par_chunk_key]
        if isinstance(tar_chunk, bytes):
            tar_chunk = _get_chunk_from_bytes(storage, tensor, par_commit_id, par_chunk_key)
        for idx in local_idx:
            all_deleted_sample.append(tar_chunk.read_sample(idx, decompress=True))
    return all_deleted_sample


def get_changes_and_messages(
    version_state, storage, id_1, id_2
) -> Tuple[
    List[dict],
    Optional[List[dict]],
    List[dict],
    Optional[List[dict]],
    dict,
]:
    """Get changes and messages. """
    if id_1 is None and id_2 is None:
        return get_changes_and_messages_compared_to_prev(version_state, storage)
    return get_changes_and_message_2_ids(version_state, storage, id_1, id_2)


def get_common_ancestor_main(version_state, target_node):
    """Get the common ancestor main branch. """
    main_head = version_state["branch_commit_map"]["main"]
    main_head_node: CommitNode = version_state["commit_node_map"][main_head]
    common_id = get_lowest_common_ancestor(target_node, main_head_node)
    return common_id


def get_commits_and_messages(
        version_state, id_1, id_2
):
    """Get commits and messages. """
    msg_1 = None
    msg_2 = None
    if id_1 is None and id_2 is None:
        if version_state["commit_node"].branch == "main":
            # current node is in main branch, cannot compare the changes between main branch by default
            raise ValueError("Can't return the changes on main branch, without specifying ids.")

        commit_node = version_state["commit_node"]
        id_1 = commit_node.commit_id
        id_1_head = commit_node.is_head_node
        s = "HEAD" if id_1_head else f"{id_1} (current commit)"
        id_2 = get_common_ancestor_main(version_state, commit_node)
        lca_id = id_2
        msg_0 = f"Commits between {s} and the common ancestor ({lca_id}) in main branch:\n"
    else:
        if id_1 is None:
            id_2 = sanitize_commit(id_2, version_state)
            if version_state["commit_node_map"][id_2].branch == "main":
                raise ValueError("Can't return the changes on main branch, without specifying id_1.")

            id_1 = get_common_ancestor_main(version_state, version_state["commit_node_map"][id_2])
        if id_2 is None:
            id_1 = sanitize_commit(id_1, version_state)
            if version_state["commit_node_map"][id_1].branch == "main":
                raise ValueError("Can't return the changes on main branch, without specifying id_2.")

            id_2 = get_common_ancestor_main(version_state, version_state["commit_node_map"][id_1])
            lca_id = id_2
            msg_0 = f"Commits between the given id: {id_1} relative to the common ancestor ({lca_id}) in main branch:\n"
        else:
            id_1 = sanitize_commit(id_1, version_state)
            id_2 = sanitize_commit(id_2, version_state)
            lca_id = get_lowest_common_ancestor(version_state["commit_node_map"][id_1],
                                                version_state["commit_node_map"][id_2])
            msg_0 = f"Commits between the given id: {id_1} and id: {id_2}:\n"
            msg_1 = f"Diff in {id_1} (target id 1):\n"
            msg_2 = f"Diff in {id_2} (target id 2):\n"
    commit_changes_1: List[dict] = []
    commit_changes_2: List[dict] = []
    for commit_node, commit_changes in [
        (version_state["commit_node_map"][id_1], commit_changes_1),
        (version_state["commit_node_map"][id_2], commit_changes_2),
    ]:
        while commit_node.commit_id != lca_id:
            commit_node = commit_node.parent
            get_commit_changes_for_node(commit_node, commit_changes)
    return commit_changes_1, commit_changes_2, msg_0, msg_1, msg_2


def get_changes_and_messages_compared_to_prev(
    version_state, storage
) -> Tuple[List[dict], None, List[dict], None, dict]:
    """Get the changes compared to the previous. """
    commit_node = version_state["commit_node"]
    commit_id = commit_node.commit_id
    head = commit_node.is_head_node

    tensor_changes: List[dict] = []
    ds_changes: List[dict] = []
    s = "HEAD" if head else f"{commit_id} (current commit)"
    msg_1 = f"Diff in {s} relative to the previous commit:\n"
    get_tensor_changes_for_id(commit_node, storage, tensor_changes)
    get_dataset_changes_for_id(commit_node, storage, ds_changes)

    msg_dict = {"msg_0": None, "msg_1": msg_1, "msg_2": None}
    return ds_changes, None, tensor_changes, None, msg_dict


def get_changes_and_message_2_ids(
    version_state, storage, id_1, id_2
) -> Tuple[List[dict], List[dict], List[dict], List[dict], dict]:
    """Get changes and messages for two ids. """
    commit_node = version_state.get("commit_node", None)
    if not id_1:
        raise ValueError("Can't specify id_2 without specifying id_1")
    msg_0 = "The two diffs are computed based on the most recent common ancestor (%s) of the "
    if not id_2:
        msg_0 += "current state and the commit passed."
        id_2, id_1, head = id_1, commit_node.commit_id, commit_node.is_head_node
        msg_1 = "The diff in HEAD:\n" if head else f"Diff in {id_1} (current commit):\n"
        msg_2 = f"The diff in {id_2} (target id):\n"
    else:
        msg_0 += "two commits passed."
        msg_1 = f"The diff in {id_1} (target id 1):\n"
        msg_2 = f"The diff in {id_2} (target id 2):\n"

    result = compare_commits(id_1, id_2, version_state, storage)
    msg_0 %= result[4]
    msg_dict = {"msg_0": msg_0, "msg_1": msg_1, "msg_2": msg_2}
    return (
        result[0],
        result[1],
        result[2],
        result[3],
        msg_dict
    )


def compare_commits(
    id_1: str, id_2: str, version_state: Dict[str, Any], storage: LRUCache
) -> Tuple[List[dict], List[dict], List[dict], List[dict], str]:
    """Compares two commits and returns the differences.

    Args:
        id_1: The first commit_id or branch name.
        id_2: The second commit_id or branch name.
        version_state: The version state.
        storage: The underlying storage of the dataset.

    Returns:
        The differences between the two commits and the id of the lowest common ancestor.
    """
    id_1 = sanitize_commit(id_1, version_state)
    id_2 = sanitize_commit(id_2, version_state)
    commit_node_1: CommitNode = version_state["commit_node_map"][id_1]
    commit_node_2: CommitNode = version_state["commit_node_map"][id_2]
    lca_id = get_lowest_common_ancestor(commit_node_1, commit_node_2)
    lca_node: CommitNode = version_state["commit_node_map"][lca_id]

    tensor_changes_1: List[dict] = []
    tensor_changes_2: List[dict] = []
    ds_changes_1: List[dict] = []
    ds_changes_2: List[dict] = []

    for commit_node, tensor_changes, dataset_changes in [
        (commit_node_1, tensor_changes_1, ds_changes_1),
        (commit_node_2, tensor_changes_2, ds_changes_2),
    ]:
        while commit_node.commit_id != lca_node.commit_id:
            get_tensor_changes_for_id(commit_node, storage, tensor_changes)
            get_dataset_changes_for_id(commit_node, storage, dataset_changes)
            commit_node = commit_node.parent  # type: ignore

    return (
        ds_changes_1,
        ds_changes_2,
        tensor_changes_1,
        tensor_changes_2,
        lca_id,
    )


def sanitize_commit(temp_id: str, version_state: Dict[str, Any]) -> str:
    """Checks the id.
    If it's a valid commit_id, it is returned.
    If it's a branch name, the commit_id of the branch's head is returned.
    Otherwise a ValueError is raised.
    """
    if temp_id in version_state["commit_node_map"]:
        return temp_id
    if temp_id in version_state["branch_commit_map"]:
        return version_state["branch_commit_map"][temp_id]
    raise KeyError(f"The commit/branch {temp_id} does not exist in the dataset.")


def get_lowest_common_ancestor(p: CommitNode, q: CommitNode):
    """Returns the lowest common ancestor of two commits."""
    if p == q:
        return p.commit_id

    p_family = []
    while p:
        p_family.append(p.commit_id)
        p = p.parent

    q_family = set()
    while q:
        q_family.add(q.commit_id)
        q = q.parent

    for temp_id in p_family:
        if temp_id in q_family:
            return temp_id


def fetch_each_range(target_changes, limit, offset: int, is_add: bool = False):
    """Fetch the ranges. """
    if is_add:
        num_values = target_changes[1] - target_changes[0]
        if offset > num_values:
            return [target_changes[1], target_changes[1]], offset - num_values, limit
        if offset + limit > num_values:
            num_added_samples = num_values - offset
            return [target_changes[0] + offset, target_changes[1]], 0, limit - num_added_samples
        return [target_changes[0] + offset, target_changes[0] + offset + limit], 0, 0

    if offset > len(target_changes):
        return [], offset - len(target_changes), limit
    if offset + limit > len(target_changes):
        return target_changes[offset:], 0, limit - (len(target_changes) - offset)
    return target_changes[offset:offset + limit], 0, 0


def calcul_range(tensor_changes, offset, limit) -> Tuple[List[Any], List[Any], Set[Any]]:
    """Calculates the range of tensor changes."""
    if not limit:
        limit = math.inf
    data_added = tensor_changes['data_added'] # [3,10]
    data_updated = list(tensor_changes['data_updated']) # {0,1,2}
    data_deleted_ids = tensor_changes['data_deleted_ids'] # {uuid1, uuid2...}

    added_range, offset, limit = fetch_each_range(data_added, limit, offset, True)
    updated_range, offset, limit = fetch_each_range(data_updated, limit, offset)
    del_range, _, _ = fetch_each_range(data_deleted_ids, limit, offset)

    return added_range, updated_range, set(del_range)


def get_all_commits_string(
        commit_changes_1, commit_changes_2, msg_0, msg_1=None, msg_2=None
):
    """Returns a string with all commits."""
    all_commits = ["\n## Commits Records"]
    if msg_0:
        all_commits.append(msg_0)

    separator = "-" * 120
    message_1 = colour_string(msg_1, "blue") if msg_1 else None
    message_2 = colour_string(msg_2, "blue") if msg_2 else None
    commits1_str = get_commits_str(
        commit_changes_1, separator, message_1
    )
    all_commits.append(commits1_str)
    commits2_str = get_commits_str(
        commit_changes_2, separator, message_2
    )
    all_commits.append(commits2_str)
    all_commits.append(separator)
    return "\n".join(all_commits)


def get_all_changes_string(
    ds_changes_1,
    ds_changes_2,
    tensor_changes_1,
    tensor_changes_2,
    msg_dict,
    asrow,
    show_value
):
    """Returns a string with all changes."""
    all_changes = ["\n## MULLER_F Diff"]
    msg_0 = msg_dict.get("msg_0", None)
    msg_1 = msg_dict.get("msg_1", None)
    msg_2 = msg_dict.get("msg_2", None)
    if msg_0:
        all_changes.append(msg_0)

    separator = "-" * 120
    if tensor_changes_1 is not None:
        changes1_str = get_changes_str(
            ds_changes_1, tensor_changes_1, colour_string(msg_1, "blue"), separator, asrow, show_value
        )
        all_changes.append(changes1_str)
    if tensor_changes_2 is not None:
        changes2_str = get_changes_str(
            ds_changes_2, tensor_changes_2, colour_string(msg_2, "blue"), separator, asrow, show_value
        )
        all_changes.append(changes2_str)
    all_changes.append(separator)
    return "\n".join(all_changes)


def colour_string(string: str, colour: str) -> str:
    """Returns a coloured string."""
    if colour == "yellow":
        return "\033[93m" + string + "\033[0m"
    if colour == "blue":
        return "\033[94m" + string + "\033[0m"
    return string


def get_commits_str(
    commits_changes: List, separator: str, message: str = None
):
    """Returns a string with commits."""
    all_changes = [separator, message] if message else [separator]
    local_separator = "*" * 80
    for commit_change in commits_changes:
        commit_id = commit_change["commit_id"]
        author = commit_change["author"]
        message = commit_change["message"]
        date = commit_change["date"]
        if date is None:
            commit_id = "checkout node or uncommitted node"
        else:
            date = str(date)

        all_commits = [
            local_separator,
            colour_string(f"commit {commit_id}", "yellow"),
            f"Author: {author}",
            f"Date: {date}",
            f"Message: {message}",
            "",
        ]
        all_changes.extend(all_commits)
    if len(all_changes) <= 2:
        all_changes.append("No commits were made.\n")
    return "\n".join(all_changes)


def get_changes_str(
    ds_changes: List, tensor_changes: List, message: str, separator: str, asrow: bool, show_value: bool
):
    """Returns a string with changes made."""
    all_changes = [separator, message]
    local_separator = "*" * 80
    for ds_change, tensor_change in zip(ds_changes, tensor_changes):
        commit_id = ds_change["commit_id"]
        author = ds_change["author"]
        message = ds_change["message"]
        date = ds_change["date"]
        if commit_id != tensor_change["commit_id"]:
            raise Exception
        if date is None:
            commit_id = "UNCOMMITTED HEAD"
        else:
            date = str(date)

        all_changes_for_commit = [
            local_separator,
            colour_string(f"commit {commit_id}", "yellow"),
            f"Author: {author}",
            f"Date: {date}",
            f"Message: {message}",
            "",
        ]
        _get_dataset_changes_str_list(ds_change, all_changes_for_commit)
        _get_tensor_changes_str_list(tensor_change, all_changes_for_commit, asrow, show_value)
        if len(all_changes_for_commit) == 6:
            all_changes_for_commit.append("No changes were made in this commit.")
        all_changes.extend(all_changes_for_commit)
    if len(all_changes) == 2:
        all_changes.append("No changes were made.\n")
    return "\n".join(all_changes)


def _get_dataset_changes_str_list(ds_change: Dict, all_changes_for_commit: List[str]):
    if ds_change.get("info_updated", False):
        all_changes_for_commit.append("- Updated dataset info \n")
    if ds_change.get("deleted"):
        for name in ds_change["deleted"]:
            all_changes_for_commit.append(f"- Deleted:\t{name}")
    if ds_change.get("renamed"):
        for old, new in ds_change["renamed"].items():
            all_changes_for_commit.append(f"- Renamed:\t{old} -> {new}")
    if len(all_changes_for_commit) > 6:
        all_changes_for_commit.append("\n")


def _get_tensor_changes_str_list(
        tensor_change: Dict,
        all_changes_for_commit: List[str],
        asrow: bool,
        show_value: bool
):
    if asrow:
        data_added_str = convert_adds_to_string(tensor_change['data_added'], tensor_change['add_value'], show_value)
        if data_added_str:
            all_changes_for_commit.append(data_added_str)

        if tensor_change['data_updated']:
            data_updated_str = convert_updates_deletes_to_string(tensor_change['data_updated'], "Updated",
                                                       tensor_change, show_value)
            all_changes_for_commit.append(data_updated_str)

        if tensor_change['data_updated']:
            output = convert_updates_deletes_to_string(tensor_change['data_deleted'],
                                                       "Deleted", tensor_change, show_value)
            all_changes_for_commit.append(output)
        all_changes_for_commit.append("")
    else:
        tensors = sorted(tensor_change.keys())
        for tensor in tensors:
            if tensor == "commit_id":
                continue
            change = tensor_change[tensor]
            if not has_change(change):
                continue
            all_changes_for_commit.append(tensor)
            if change["created"]:
                all_changes_for_commit.append("* Created tensor")

            if change["cleared"]:
                all_changes_for_commit.append("* Cleared tensor")

            data_added = change.get("data_added", [0, 0])
            data_added_str = convert_adds_to_string(data_added, change, show_value)
            if data_added_str:
                all_changes_for_commit.append(data_added_str)

            data_updated = change["data_updated"]
            if data_updated:
                data_updated_str = convert_updates_deletes_to_string(data_updated, "Updated",
                                                           change, show_value)
                all_changes_for_commit.append(data_updated_str)

            data_deleted = change["data_deleted"]
            if data_deleted:
                output = convert_updates_deletes_to_string(data_deleted,
                                                           "Deleted", change, show_value)
                all_changes_for_commit.append(output)

            info_updated = change["info_updated"]
            if info_updated:
                all_changes_for_commit.append("* Updated tensor info")
            all_changes_for_commit.append("")


def has_change(change: Dict) -> bool:
    """Returns whether there is changes. """
    data_added = change.get("data_added", [0, 0])
    return any([
        change.get("created", False),
        change.get("cleared", False),
        data_added[1] - data_added[0] > 0,
        change.get("data_updated", set()),
        change.get("info_updated", False),
        change.get("data_deleted", set())
    ])


def get_dataset_changes_for_id(
    commit_node,
    storage: LRUCache,
    dataset_changes,
):
    """Returns the changes made in the dataset for a commit."""
    commit_id = commit_node.commit_id
    dataset_diff_key = get_dataset_diff_key(commit_id)

    time = str(commit_node.commit_time)[:-7] if commit_node.commit_time else None
    dataset_change = {
        "commit_id": commit_id,
        "author": commit_node.commit_user_name,
        "message": commit_node.commit_message,
        "date": time,
    }
    try:
        dataset_diff = storage.get_muller_object(dataset_diff_key, DatasetDiff)
    except KeyError:
        changes = {"info_updated": False, "renamed": {}, "deleted": [],
                   "commit_diff_exist": False, "tensor_info_updated": False}
        dataset_change.update(changes)
        dataset_changes.append(dataset_change)
        return

    changes = {
        "info_updated": dataset_diff.info_updated,
        "renamed": dataset_diff.renamed.copy(),
        "deleted": dataset_diff.deleted.copy(),
        "commit_diff_exist": dataset_diff.commit_diff_exist,
        "tensor_info_updated": dataset_diff.tensor_info_updated,
    }
    dataset_change.update(changes)
    dataset_changes.append(dataset_change)


def get_commit_changes_for_node(
        commit_node,
        commit_changes: List[Dict],
):
    """Identifies the changes made in the given commit_id and updates them in the changes dict."""
    commit_id = commit_node.commit_id
    time = str(commit_node.commit_time)[:-7] if commit_node.commit_time else None
    commit_change = {
        "commit_id": commit_id,
        "author": commit_node.commit_user_name,
        "message": commit_node.commit_message,
        "date": time,
    }
    commit_changes.append(commit_change)


def get_tensor_changes_for_id(
    commit_node,
    storage: LRUCache,
    tensor_changes: List[Dict],
):
    """Identifies the changes made in the given commit_id and updates them in the changes dict."""
    commit_id = commit_node.commit_id
    meta_key = get_dataset_meta_key(commit_id)
    meta: DatasetMeta = storage.get_muller_object(meta_key, DatasetMeta)
    tensors = meta.visible_tensors

    commit_changes = {"commit_id": commit_id}
    for tensor in tensors:
        key = meta.tensor_names[tensor]
        commit_diff_key = get_tensor_commit_diff_key(key, commit_id)
        try:
            commit_diff: CommitDiff = storage.get_muller_object(commit_diff_key, CommitDiff)
        except KeyError:
            tensor_change = {
                "created": False,
                "cleared": False,
                "info_updated": False,
                "data_added": [0, 0],
                "data_updated": set(),
                "data_deleted": set(),
                "data_deleted_ids": [],
                "data_transformed_in_place": False,
            }

            commit_changes[tensor] = tensor_change
            continue
        tensor_change = {
            "created": commit_diff.created,
            "cleared": commit_diff.cleared,
            "info_updated": commit_diff.info_updated,
            "data_added": commit_diff.data_added.copy(),
            "data_updated": commit_diff.data_updated.copy(),
            "data_deleted": commit_diff.data_deleted.copy(),
            "data_deleted_ids": commit_diff.data_deleted_ids.copy(),
            "data_transformed_in_place": commit_diff.data_transformed,
        }
        commit_changes[tensor] = tensor_change

    tensor_changes.append(commit_changes)


def convert_updates_deletes_to_string(indexes: Set[int],
                                      operation: str,
                                      tensor_change_values: Dict,
                                      show_value: bool
                                      ) -> str:
    """Converts the updates and deletes to strings. """
    if operation == "Deleted":
        num_samples = len(indexes)
        output = indexes
        if show_value:
            handle_value = tensor_change_values['data_deleted_values']
        else:
            handle_value = None
    else:
        range_intervals = _compress_into_range_intervals(indexes)
        output = _range_interval_list_to_string(range_intervals)
        num_samples = len(indexes)
        if show_value:
            handle_value = tensor_change_values['updated_values']
        else:
            handle_value = None
    sample_string = "sample" if num_samples == 1 else "samples"
    if handle_value:
        return (f"* {operation} {num_samples} {sample_string}: [{output}]" + "," +
                f"\nThe {operation} data values are {handle_value}")

    return f"* {operation} {num_samples} {sample_string}: [{output}]"


def convert_adds_to_string(index_range: List[int], add_values: List[Dict], show_value: bool) -> str:
    """Function to convert changes to string"""
    num_samples = index_range[1] - index_range[0]
    if num_samples == 0:
        return ""
    sample_string = "sample" if num_samples == 1 else "samples"
    if show_value:
        if add_values:
            return (f"* Added {num_samples} {sample_string}: [{index_range[0]}-{index_range[1]}]" + "," +
                    f"\nThe appended data values are {add_values}")
    else:
        return f"* Added {num_samples} {sample_string}: [{index_range[0]}-{index_range[1]}]"
    return ""


def merge_renamed_deleted(dataset_changes):
    """Merge the renamed deletions. """
    deleted = []
    renamed = OrderedDict()
    done = set()
    merge_renamed = {}
    for dataset_change in dataset_changes:
        for old, new in dataset_change["renamed"].items():
            if deleted and new in deleted and new not in done:
                deleted[deleted.index(new)] = old
                done.add(new)
                continue
            if renamed and renamed.get(new):
                merge_renamed[old] = renamed.get(new)
                renamed.pop(new)
            else:
                merge_renamed[old] = new
        deleted.extend(dataset_change["deleted"])
    return merge_renamed, deleted


def _range_interval_list_to_string(range_intervals: List[Tuple[int, int]]) -> str:
    """Converts the range intervals to a string."""
    return ", ".join(
        f"{start}-{end}" if start != end else f"{start}"
        for start, end in range_intervals
    )


def _compress_into_range_intervals(indexes: Set[int]) -> List[Tuple[int, int]]:
    """Compresses the indexes into range intervals."""

    if not indexes:
        return []

    sorted_indexes = sorted(indexes)
    result = []
    start = end = sorted_indexes[0]

    for idx in sorted_indexes[1:]:
        if idx == end + 1:
            end = idx
        else:
            result.append((start, end))
            start = end = idx

    result.append((start, end))
    return result
