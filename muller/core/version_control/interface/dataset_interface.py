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

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import muller
from muller.constants import TO_DATAFRAME_SAFE_LIMIT, FIRST_COMMIT_ID, DATASET_UUID_NAME
from muller.core.chunk.uncompressed_chunk import UncompressedChunk
from muller.core.lock import lock_dataset, unlock_dataset
from muller.core.meta.dataset_meta import DatasetMeta
from muller.core.meta.encode.chunk_id import ChunkIdEncoder
from muller.core.meta.tensor_meta import TensorMeta
from muller.core.serialize import deserialize_chunkids, deserialize_chunk
from muller.core.storage.local import LocalProvider
from muller.core.version_control.commit_node import CommitNode
from muller.core.version_control.interface.diff_interface import (sanitize_commit, get_lowest_common_ancestor,
                                                                 get_changes_and_messages, \
                                                                 get_all_changes_string, get_tensor_changes_for_id,
                                                                 get_dataset_changes_for_id, get_commits_and_messages, \
                                                                 get_all_commits_string, calcul_range,
                                                                 handle_append_ranges, handle_update_ranges,
                                                                 handle_delete_ranges, \
                                                                 generate_add_values, generate_update_values,
                                                                 generate_delete_values)
from muller.util.dataset import try_flushing
from muller.util.exceptions import EmptyCommitError, ReadOnlyModeError, CheckoutError, UnAuthorizationError, \
    DatasetCorruptError, ExportDataFrameLimit, VersionControlError
from muller.util.keys import get_tensor_meta_key, get_dataset_meta_key, get_sample_id_tensor_key, \
    get_chunk_id_encoder_key, get_chunk_key
from muller.core.version_control.interface.merge_interface import merge_detect, get_node_tensors, direct_detect
from muller.util.remove_cache import create_read_copy_dataset
from muller.util.version_control import integrity_check, reset_and_checkout, current_commit_has_change, \
    warn_node_checkout, load_meta, replace_head

_LOCKABLE_STORAGES = {LocalProvider}


def commits(ds, ordered_by_date=False) -> List[Dict]:
    """Lists all the commits leading to the current dataset state.

    Args:
        ds: Dataset object.
        ordered_by_date (bool): If ``True``, the commit messages will be sorted by date from newest to oldest;
        If ``False``, they will be displayed in topological order.

    Returns:
        List of dictionaries containing commit information.
    """
    temp_commits = []
    commit_nodes = _get_commit_nodes(ds, ordered_by_date=ordered_by_date)

    for commit_node in commit_nodes:
        if not commit_node.is_head_node:
            commit_info = {
                "commit": commit_node.commit_id,
                "author": commit_node.commit_user_name,
                "time": str(commit_node.commit_time)[:-7],
                "message": commit_node.commit_message,
            }
            temp_commits.append(commit_info)
    return temp_commits


def get_commit_details(ds, commit_id) -> Dict:
    """Get details of a particular commit.

    Args:
        ds: Dataset object.
        commit_id (str): commit id of the commit.

    Returns:
        Dict: Dictionary of details with keys - ``commit``, ``author``, ``time``, ``message``.

    Raises:
        KeyError: If given ``commit_id`` was not found in the dataset.
    """
    commit_node: CommitNode = ds.version_state["commit_node_map"].get(commit_id)
    if commit_node is None:
        raise KeyError(f"Commit {commit_id} not found in dataset.")

    tmp_time = str(commit_node.commit_time)[:-7] if commit_node.commit_time else None
    return {
        "commit": commit_node.commit_id,
        "author": commit_node.commit_user_name,
        "time": tmp_time,
        "message": commit_node.commit_message,
    }


def commit(ds, message: Optional[str] = None, allow_empty=False) -> str:
    """Stores a snapshot of the current state of the dataset.

    Args:
        ds: Dataset object.
        message (str, Optional): Used to describe the commit.
        allow_empty (bool): If ``True``, commit even if there are no changes.

    Returns:
        str: the commit id of the saved commit that can be used to access the snapshot.

    Raises:
        Exception: If dataset is a filtered view.
        EmptyCommitError: if there are no changes and user does not forced to commit unchanged data.

    Note:
        - Committing from a non-head node in any branch, will lead to an automatic checkout to a new branch.
        - This same behaviour happens if new samples are added or existing samples are updated from a non-head node.
    """

    if not allow_empty and not ds.has_head_changes:
        raise EmptyCommitError(
            "There are no changes, commit is not done. Try again with allow_empty=True."
        )

    return protected_commit(ds, message, None, False)


def protected_commit(
        ds,
        message: Optional[str] = None,
        commit_hash: Optional[str] = None,
        flush_version_control_info: bool = True,
        *,
        is_checkpoint: bool = False,
        total_samples_processed: int = 0,
) -> str:
    """Official commit. """
    if ds.is_filtered_view:
        raise Exception(
            "Cannot perform version control operations on a filtered dataset view."
        )

    try_flushing(ds)

    ds.initial_autoflush.append(ds.storage.autoflush)
    ds.storage.autoflush = False
    try:
        unlock_dataset(ds)
        muller.util.version_control.commit(
            ds,
            message,
            commit_hash,
            flush_version_control_info,
            is_checkpoint=is_checkpoint,
            total_samples_processed=total_samples_processed,
        )
        if not flush_version_control_info:
            ds.vc_info_updated = True
        ds.lock()
    finally:
        ds.storage.autoflush = ds.initial_autoflush.pop()
    ds.info = None
    ds.ds_diff = None
    ds.maybe_flush()
    return ds.commit_id  # type: ignore


def checkout(
        ds, address: str, create: bool = False, reset_flag: bool = False
) -> Optional[str]:
    """Checks out to a specific commit_id or branch. If ``create = True``, creates a new branch named ``address``.

    Args:
        ds: Dataset object.
        address (str): The commit_id or branch to checkout to.
        create (bool): If ``True``, creates a new branch with name as address.
        reset_flag (bool): If checkout fails due to a corrupted HEAD state of the branch, setting ``reset=True`` will
                      reset HEAD changes and attempt the checkout again.

    Returns:
        Optional[str]: The commit_id of the dataset after checkout.

    Raises:
        CheckoutError: If ``address`` could not be found.
        ReadOnlyModeError: If branch creation or reset is attempted in read-only mode.
        DatasetCorruptError: If checkout failed due to dataset corruption and ``reset`` is not ``True``.
        Exception: If the dataset is a filtered view.

    Examples:

        >>> ds = muller.empty("../test/test_ds")
        >>> ds.create_tensor("abc")
        Tensor(key='abc')
        >>> ds.abc.append([1, 2, 3])
        >>> first_commit = ds.commit()
        >>> ds.checkout("alt", create=True)
        'firstdbf9474d461a19e9333c2fd19b46115348f'
        >>> ds.abc.append([4, 5, 6])
        >>> ds.abc.numpy()
        array([[1, 2, 3],
               [4, 5, 6]])
        >>> ds.checkout(first_commit)
        'firstdbf9474d461a19e9333c2fd19b46115348f'
        >>> ds.abc.numpy()
        array([[1, 2, 3]])

    Note:
        Checkout from a head node in any branch that contains uncommitted data will lead to
        an automatic commit before the checkout.
    """

    try:
        ret = ds.protect_checkout(address, create, None, False)
        integrity_check(ds)
        return ret
    except (ReadOnlyModeError, CheckoutError, UnAuthorizationError) as e:
        raise e from None
    except Exception as e:
        if create:
            raise e
        if not reset_flag:
            if isinstance(e, DatasetCorruptError):
                raise DatasetCorruptError(
                    message=e.message,
                    action="Try using `reset_flag=True` to reset HEAD changes and load the previous commit.",
                    cause=e.__cause__,
                ) from e
            raise DatasetCorruptError(
                "Exception occurred (see Traceback). The branch you are checking out to maybe corrupted."
                "Try using `reset_flag=True` to reset HEAD changes and load the previous commit."
                "This will delete all uncommitted changes on the branch you are trying to load."
            ) from e
        if ds.read_only:
            raise ReadOnlyModeError("Cannot reset HEAD in read-only mode.") from e
        return reset_and_checkout(ds, address, e)


def detect_merge_conflict(ds, target_id: str, show_value: bool = False):
    """Detect the conflict between current stage and target stage of given commit id.

    Args:
        ds: Dataset Object.
        target_id (str): The commit_id or branch to detect.
        show_value: show the conflict values of each branch.
    Returns: dict of conflict_tensors and the dict of visible tensors and their conflicts records.

    """
    if ds.is_filtered_view:
        raise Exception(
            "Cannot perform version control operations on a filtered dataset view."
        )
    target_commit_id = sanitize_commit(target_id, ds.version_state)

    if not ds.version_state["commit_node"].is_head_node:
        raise DatasetCorruptError(f"current commit node is not the head node of current branch, "
                                  f"please checkout to the new branch first.")

    original_id = ds.pending_commit_id
    _ = checkout(ds, target_commit_id, False)
    version_state = ds.version_state
    commit_node = version_state["commit_node"]
    if commit_node.is_head_node:
        if current_commit_has_change(version_state, ds.storage):
            raise DatasetCorruptError(f"The given target state:{target_commit_id} has uncommitted changes. "
                                      f"Please checkout to the target state and commit first.")
        _ = checkout(ds, commit_node.parent.commit_id, False)
    target_id = ds.pending_commit_id
    _ = checkout(ds, original_id, False)

    conflict_tensors, records = merge_detect(ds, version_state["commit_node"],
                                             version_state["commit_node_map"][target_id], show_value)
    conflict_records = {}
    for tensor_name, conflict_message in records.items():
        if tensor_name == "_uuid":
            continue
        conflict_records[tensor_name] = {'del_ori_idx': conflict_message[0],
                                         'del_ori_values': conflict_message[1],
                                         'del_tar_idx': conflict_message[2],
                                         'del_tar_values': conflict_message[3],
                                         'app_ori_idx': conflict_message[4],
                                         'app_ori_values': conflict_message[5],
                                         'app_tar_idx': conflict_message[6],
                                         'app_tar_values': conflict_message[7],
                                         'update_values': conflict_message[8]
                                         }
    return conflict_tensors, conflict_records


def merge(
        ds,
        target_id: str,
        append_resolution: Optional[str] = None,
        update_resolution: Optional[str] = None,
        pop_resolution: Optional[str] = None,
        delete_removed_tensors: bool = False,
        force: bool = False,
):
    """Merges the target_id into the current dataset.

    Args:
        ds: Dataset object.
        target_id (str): The commit_id or branch to merge.
        append_resolution (str, Optional):
            - The strategy to use to resolve append conflicts.
            - Conflicts are scenarios where both the current dataset and the target have appended sample/s since
            their common ancestor.
            - Must be one of the following
                - None - this is the default value, will raise an exception if there are conflicts.
                - "ours" - during conflicts, values from the current dataset will be used.
                - "theirs" - during conflicts, values from target will be used.
                - "both" - during conflicts, values from current dataset and target are both be used.
        update_resolution (str, Optional):
            - The strategy to use to resolve update conflicts.
            - Conflicts are scenarios where both the current dataset and the target have updated same sample/s of
            different values.
            - Must be one of the following
                - None - this is the default value, will raise an exception if there are conflicts.
                - "ours" - during conflicts, values from the current dataset will be used.
                - "theirs" - during conflicts, values from target will be used.
        pop_resolution (str, Optional):
            - The strategy to use to resolve pop conflicts.
            - Conflicts are scenarios where both the current dataset and the target have popped different sample/s
            of their common ancestor.
            - Must be one of the following
                - None - this is the default value, will raise an exception if there are conflicts.
                - "ours" - during conflicts, current dataset's pop records will be used.
                - "theirs" - during conflicts, target dataset's pop will be used.
                - "both" - during conflicts, current dataset and target dataset's pop are both be used.
        delete_removed_tensors (bool): If ``True``, deleted tensors will be deleted from the dataset.
        force (bool):
            - Forces merge.
            - ``force=True`` will have these effects in the following cases of merge conflicts:
                - If tensor is renamed on target but is missing from HEAD, renamed tensor will be registered
                    as a new tensor on current branch.
                - If tensor is renamed on both target and current branch, tensor on target will be registered
                    as a new tensor on current branch.
                - If tensor is renamed on target and a new tensor of the new name was created on the current branch,
                    they will be merged.

    Raises:
        Exception: if dataset is a filtered view.
        ValueError: if the append_resolution must be one of None, 'ours', or 'theirs' or 'both', update_resolution
        must be one of None, 'ours', 'theirs', pop resolution must be one of None, 'ours', 'theirs' or 'both'.
    """

    if ds.is_filtered_view:
        raise Exception(
            "Cannot perform version control operations on a filtered dataset view."
        )

    if (append_resolution not in [None, "ours", "theirs", "both"] or update_resolution
            not in [None, "ours", "theirs"] or pop_resolution not in [None, "ours", "theirs", "both"]):
        raise ValueError(
            f"append_resolution must be one of None, 'ours', or 'theirs' or 'both', update_resolution must be"
            f" one of None, 'ours', 'theirs', pop resolution must be one of None, 'ours', 'theirs' or 'both'."
        )

    conflict_resolution = {"append": append_resolution, "update": update_resolution, "pop": pop_resolution}

    try_flushing(ds)

    target_commit = target_id
    try:
        target_commit = ds.version_state["branch_commit_map"][target_id]
    except KeyError:
        pass
    if isinstance(ds.base_storage, tuple(_LOCKABLE_STORAGES)) and not (
            isinstance(ds.base_storage, LocalProvider)
            and not muller.constants.LOCK_LOCAL_DATASETS
    ):
        lock_dataset(ds, version=target_commit)
        locked = True
    else:
        locked = False
    ds.initial_autoflush.append(ds.storage.autoflush)
    ds.storage.autoflush = False
    try:
        muller.core.version_control.interface.merge_interface.merge(ds, target_id, conflict_resolution,
                                                                   delete_removed_tensors, force)
    finally:
        if locked:
            unlock_dataset(ds, version=target_commit)
        ds.storage.autoflush = ds.initial_autoflush.pop()
        ds.storage.maybe_flush()


def protect_checkout(
        ds,
        address: str,
        create: bool = False,
        commit_hash: Optional[str] = None,
        verbose: bool = True,
        flush_version_control_info: bool = False,
) -> Optional[str]:
    """Protected checkout with auto-commit and branch metadata support."""
    if ds.is_filtered_view:
        raise Exception(
            "Cannot perform version control operations on a filtered dataset view."
        )
    read_only = ds.read_only
    if read_only and create:
        raise ReadOnlyModeError()
    
    # Auto-commit before checkout if enabled and there are uncommitted changes
    if muller.constants.AUTO_COMMIT_BEFORE_CHECKOUT and ds.has_head_changes:
        from muller.util.version_control import auto_commit_before_checkout
        auto_commit_before_checkout(ds, address)
    
    try_flushing(ds)
    
    # Store old branch for lock management
    old_branch = ds.version_state.get("branch", "main")
    
    ds.initial_autoflush.append(ds.storage.autoflush)
    ds.storage.autoflush = False
    try:
        # Release lock on old branch before checkout
        unlock_dataset(ds, branch=old_branch)
        
        # Save branch metadata when creating new branch
        if create:
            from muller.util.authorization import obtain_current_user
            from muller.util.version_control import save_branch_metadata
            current_user = obtain_current_user()
            save_branch_metadata(
                ds.storage,
                branch_name=address,
                owner=current_user,
                parent_branch=old_branch
            )
        
        # Perform the actual checkout
        muller.util.version_control.checkout(ds, address, create, commit_hash, flush_version_control_info)
        
        if not flush_version_control_info and create:
            ds.vc_info_updated = True
        
        # Acquire lock on new branch after checkout
        new_branch = ds.version_state.get("branch", "main")
        lock_dataset(ds, branch=new_branch)
        
    finally:
        ds.set_read_only(read_only, err=True)
        ds.storage.autoflush = ds.initial_autoflush.pop()
    
    ds.info = None
    ds.ds_diff = None

    commit_node = ds.version_state["commit_node"]
    if ds.verbose:
        warn_node_checkout(commit_node, create)
    if create:
        ds.maybe_flush()
    return ds.commit_id


def generate_add_update_value(ds, commit_changes, offset, limit, asrow, tensors=None):
    """Obtain the details of the add/update/delete samples."""
    tensor_names = list(commit_changes.keys())[1:]
    commit_id = commit_changes['commit_id']

    if ds.version_state['commit_node_map'][commit_id].parent:
        par_id = ds.version_state['commit_node_map'][commit_id].parent.commit_id
    else:
        par_id = None

    changes_records_ranges, changes_records_add, changes_records_update, changes_records_del, commit_changes = (
        _process_update_add_value_of_tensor(ds, tensor_names, tensors, commit_changes, offset, limit,
                                                 commit_id, par_id, asrow))
    if asrow:
        if len(set(changes_records_ranges.values())) == 1:
            commit_changes.clear()
            commit_changes['commit_id'] = commit_id
            commit_changes['data_added'] = list(set(changes_records_ranges.values()).pop()[0])
            commit_changes['data_updated'] = set(set(changes_records_ranges.values()).pop()[1])
            commit_changes['data_deleted'] = set(set(changes_records_ranges.values()).pop()[2])
            commit_changes['add_value'] = [dict(zip(changes_records_add, t))
                                           for t in zip(*changes_records_add.values())]
            commit_changes['updated_values'] = [dict(zip(changes_records_update, t))
                                                for t in zip(*changes_records_update.values())]
            commit_changes['data_deleted_values'] = [dict(zip(changes_records_del, t))
                                                     for t in zip(*changes_records_del.values())]
        else:
            raise Exception("changes in tensors are different, can not show in row, please set row to False.")


def direct_diff(ds, id_1: str = None, id_2: str = None,
                as_dataframe: Optional[bool] = False, force: Optional[bool] = False):
    """Detect the direct difference of id_2 compared with id_1.

    Args:
        ds: Dataset object.
        id_1 (str): The commit_id or branch to compare.
        id_2 (str): The commit_id or branch to detect.
        as_dataframe (bool, Optional): Whether to return as dataframe or not.
        force (bool, Optional) - Dataset with more than TO_DATAFRAME_SAFE_LIMIT samples might take a long time to
                    export. If force = True, the dataset will be exported regardless.
                    An error will be raised otherwise.

    Returns:
        A dictionary result_dict that contains the direct diff of id_2 compared with id_1.
        If as_dataframe = True, result_dict contains 5 pandas dataframes,
            i.e., added columns, removed columns, added rows, removed row, edited rows.
        Otherwise, result_dict contains 3 elements,
            The new tensors(name) set of id_2,
            The removed tensors(name) set of id_1,
            and the records of dict contains the added values, removed values and edited values of each tensor.

    Raises:
        ExportDataFrameLimit: If as_dataframe is True and the length of dataset exceeds the TO_DATAFRAME_SAFE_LIMIT.

    """
    tar_id_1 = sanitize_commit(id_1, ds.version_state)
    tar_id_2 = sanitize_commit(id_2, ds.version_state)

    tar_1_tensors = get_node_tensors(ds, tar_id_1)
    tar_2_tensors = get_node_tensors(ds, tar_id_2)
    removed_tensors = tar_1_tensors - tar_2_tensors
    new_tensors = tar_2_tensors - tar_1_tensors
    common_tensors = tar_1_tensors & tar_2_tensors
    target_ds_1 = create_read_copy_dataset(ds, tar_id_1)
    target_ds_2 = create_read_copy_dataset(ds, tar_id_2)

    records = _get_valid_records(ds, tar_id_1, tar_id_2, target_ds_1, target_ds_2, common_tensors)

    return _get_results_dict(records, common_tensors, as_dataframe, new_tensors, removed_tensors, force)


def _get_valid_records(ds, tar_id_1, tar_id_2, target_ds_1, target_ds_2, common_tensors):
    nodes: Dict[str, CommitNode] = {}
    try:
        nodes["target_1"] = ds.version_state["commit_node_map"][tar_id_1]
    except KeyError as e:
        raise ValueError(f"Can not find the commit id {tar_id_1} from commit_node_map.") from e

    try:
        nodes["target_2"] = ds.version_state["commit_node_map"][tar_id_2]
    except KeyError as e:
        raise ValueError(f"Can not find the commit id {tar_id_2} from commit_node_map.") from e
    lca_id = get_lowest_common_ancestor(nodes["target_1"], nodes["target_2"])
    try:
        nodes["lca"] = ds.version_state["commit_node_map"][lca_id]
    except KeyError as e:
        raise ValueError(f"Can not find the commit id {lca_id} from commit_node_map.") from e

    commits_id: Dict[str, str] = {'tar_1': tar_id_1, 'tar_2': tar_id_2}

    records = {}
    for tensor_name in common_tensors:
        records.update({tensor_name: direct_detect(tensor_name, nodes, target_ds_1, target_ds_2, commits_id)})

    return records


def _get_results_dict(records, common_tensors, as_dataframe, new_tensors,
                      removed_tensors, force):
    if as_dataframe:
        result_dict = {"added_columns": pd.DataFrame.from_dict({"added_columns": list(new_tensors)}),
                       "removed_columns": pd.DataFrame.from_dict({"removed_columns": list(removed_tensors)}),
                       "added_rows": _data_to_dataframe(records, common_tensors,
                                                        "added_values", force),
                       "removed_rows": _data_to_dataframe(records, common_tensors,
                                                          "removed_values", force),
                       "edited_rows": _data_to_dataframe(records, common_tensors,
                                                         "edited_values_tar1", force)}
    else:
        result_dict = {"added_columns": new_tensors,
                       "removed_columns": removed_tensors,
                       "modified_records": records}

    return result_dict


def diff(
        ds,
        id_1: Optional[str] = None,
        id_2: Optional[str] = None,
        as_dict: bool = False,
        show_value: bool = False,
        offset: int = 0,
        limit: Optional[int] = None,
        asrow: bool = False
) -> Optional[Dict]:
    """Returns/displays the differences between commits/branches.

    For each tensor this contains information about the sample indexes that were added/modified
    as well as whether the tensor was created.

    Args:
        ds: Dataset object.
        id_1 (str, Optional): The first commit_id or branch name.
        id_2 (str, Optional): The second commit_id or branch name.
        as_dict (bool, Optional): If ``True``, returns the diff as lists of commit wise dictionaries.
        show_value (bool): If ``True``, returns the value of appended data, updated data and popped data.
        offset(int): The number of returned data to skip, only applicable if ``show_value`` is True.
        Defaults to ``0``.
        limit(int, Optional): The number of returned data, only applicable if ``show_value`` is True.
        Defaults be ``None``.
        asrow(bool): Whether return data according to row instead of column,
        only applicable if ``show_value`` is True. Defaults to ``False``.

    Returns:
        Optional[Dict]

    Raises:
        ValueError: If ``id_1`` is None and ``id_2`` is not None.

    Note:
        - If both ``id_1`` and ``id_2`` are None, the differences between the current state
            and the previous commit will be calculated.
            If you're at the head of the branch, this will show the uncommitted changes, if any.
        - If only ``id_1`` is provided, the differences between the current state and id_1 will be calculated.
            If you're at the head of the branch, this will take into account the uncommitted changes, if any.
        - If only ``id_2`` is provided, a ValueError will be raised.
        - If both ``id_1`` and ``id_2`` are provided, the differences between ``id_1`` and ``id_2`` will be
            calculated.

    Note:
        A dictionary of the differences between the commits/branches is returned if ``as_dict`` is ``True``.
        The dictionary will always have 2 keys, "dataset" and "tensors".
        The values corresponding to these keys are detailed below:

            - If ``id_1`` and ``id_2`` are None, both the keys will have a single list as their value.
                This list will contain a dictionary describing changes compared to the previous commit.
            - If only ``id_1`` is provided, both keys will have a tuple of 2 lists as their value.
                The lists will contain dictionaries describing commitwise differences between commits.
                The 2 lists will range from current state and ``id_1`` to most recent common ancestor
                the commits respectively.
            - If only ``id_2`` is provided, a ValueError will be raised.
            - If both ``id_1`` and ``id_2`` are provided, both keys will have a tuple of 2 lists as their value.
                The lists will contain dictionaries describing commitwise differences between commits.
                The 2 lists will range from ``id_1`` and ``id_2`` to most recent common ancestor the commits
                respectively.

        ``None`` is returned if ``as_dict`` is ``False``.
    """

    version_state, storage = ds.version_state, ds.storage
    res = get_changes_and_messages(version_state, storage, id_1, id_2)
    # res[0]: The changes between id_1 and common ancestor on dataset
    # res[1]: The changes between id_2 and common ancestor on dataset
    # res[2]: The changes between id_1 and common ancestor on tensor
    # res[3]: The changes between id_2 and common ancestor on tensor

    if show_value:
        if res[2] is not None and len(res[2]) > 0:
            for tensor_change_ver in res[2]:
                ds.generate_add_update_value(tensor_change_ver, offset, limit, asrow)
        if res[3] is not None and len(res[3]) > 0:
            for tensor_change_ver in res[3]:
                ds.generate_add_update_value(tensor_change_ver, offset, limit, asrow)

    if as_dict:
        changes = {}
        if id_1 is None and id_2 is None:
            changes["dataset"] = res[0]
            changes["tensor"] = res[2]
            return changes
        changes["dataset"] = res[0], res[1]
        changes["tensor"] = res[2], res[3]
        return changes
    all_changes = get_all_changes_string(*res, asrow, show_value)
    print(all_changes)
    return {}


def diff_to_prev(
        ds,
        commit_id: str = None,
        as_dict=False,
        show_value=False,
        offset: int = 0,
        limit: Optional[int] = None,
        asrow: bool = False
) -> Optional[Dict]:
    """
    Returns/displays the differences between the given commit/current commit and its previous commit.

    Args:
        ds: Dataset object.
        commit_id (str): The commit id to compare. If it is ``None``, use current commit id. Defaults to ``None``.
        as_dict (bool, Optional): If ``True``, returns the diff as lists of commit wise dictionaries.
        show_value (bool): If ``True``, returns the value of appended data, updated data and popped data.
        offset(int): The number of returned data to skip, only applicable if ``show_value`` is True.
            Defaults to ``0``.
        limit(int): The number of returned data, only applicable if ``show_value`` is True. Defaults to ``1000``.
        asrow(bool): Whether return data according to row instead of column,
        only applicable if ``show_value`` is True. Defaults to ``False``.

    Returns:
        Optional[Dict]

    Raises:
        ValueError: if ``commit_id`` is not exist.

    """
    if not commit_id:
        commit_node = ds.version_state["commit_node"]
        commit_id = commit_node.commit_id
    else:
        try:
            commit_node = ds.version_state["commit_node_map"][commit_id]
        except KeyError as e:
            raise ValueError(f"Given commit_id: {commit_id} is not exist, please check and retry.") from e

    tensor_changes: List[dict] = []
    ds_changes: List[dict] = []
    s = "HEAD" if commit_node.is_head_node else f"{commit_id} (current commit)"
    get_tensor_changes_for_id(commit_node, ds.storage, tensor_changes)
    get_dataset_changes_for_id(commit_node, ds.storage, ds_changes)
    if show_value:
        if tensor_changes is not None and len(tensor_changes) > 0:
            for tensor_change_ver in tensor_changes:
                ds.generate_add_update_value(tensor_change_ver, offset, limit, asrow)
    if as_dict:
        return {"dataset": ds_changes, "tensor": tensor_changes}
    all_changes = get_all_changes_string(*(ds_changes,
                                           None,
                                           tensor_changes,
                                           None,
                                           None,
                                           f"Diff in {s} relative to the previous commit:\n",
                                           None),
                                           asrow, show_value)
    print(all_changes)
    return {}


def commits_under(
        ds,
        branch: str = None,
        ordered_by_date: bool = False
) -> List[CommitNode]:
    """
    Return the list of commits under the given branch.

    Args:
        ds: Dataset object.
        branch (str): The branch name to execute. Default to ``None``, if it is ``None``, commits under
        current branch will be return.
        ordered_by_date (bool): If ``True``, the commit messages will be sorted by date from newest to oldest;
        If ``False``, they will be displayed in topological order.

    Returns:
        List[CommitNode]

    Raises:
        ValueError: if ``commit_id`` is not exist.
        VersionControlError: KeyError in version_state.
    """
    if branch:
        try:
            branches = ds.version_state["branch_commit_map"]
            if branch in branches:
                try:
                    target_node_id = ds.version_state["branch_commit_map"][branch]
                    target_node = ds.version_state["commit_node_map"][target_node_id]
                except KeyError as e:
                    raise VersionControlError(f"Unable to obtain field on version state") from e
            else:
                raise ValueError("Given branch name does not exist, "
                                 "please use ds.branches() check all the branches and try again.")
        except KeyError as e:
            raise VersionControlError(f"Unable to obtain field on version state") from e
    else:
        target_node = None
    commits_node = _get_commit_nodes(ds, target_node, ordered_by_date)
    return commits_node


def commits_between(ds, id_1: Optional[str] = None, id_2: Optional[str] = None, as_dict=False):
    """ Show the commits history between given ids or branch names
    Args:
        ds: Dataset object.
        id_1 (str, Optional): The first commit_id or branch name.
        id_2 (str, Optional): The second commit_id or branch name.
        as_dict (bool, Optional): If ``True``, returns the diff as lists of commit wise dictionaries.

    Returns:
        Optional[List]

    Raises:
        ValueError: If ``id_1`` is None and ``id_2`` is None and current node is in main branch.
        ValueError: If ``id_1`` is None and ``id_2`` is not None and current node is in main branch.
        ValueError: If ``id_1`` is not None and ``id_2`` is None and current node is in main branch.

    Note:
        - If both ``id_1`` and ``id_2`` are None, the commits between the current state
            and the common ancestor in the main branch will be calculated.
            If you're at the head of the branch, this will show the uncommitted changes, if any.
        - If only ``id_1`` is provided, the commits between the common ancestor and id_1 will be calculated.
            If you're at the head of the branch, this will take into account the uncommitted changes, if any.
        - If only ``id_2`` is provided, the commits between the common ancestor and id_2 will be calculated.
            If you're at the head of the branch, this will take into account the uncommitted changes, if any.
        - If both ``id_1`` and ``id_2`` are provided, the commits between ``id_1`` and ``id_2`` will be
            calculated.
    """

    version_state = ds.version_state
    commit_changes_1, commit_changes_2, msg_0, msg_1, msg_2 = get_commits_and_messages(version_state, id_1, id_2)
    if as_dict:
        all_commits = [commit_changes_1, commit_changes_2]
        return all_commits
    all_commits = get_all_commits_string(commit_changes_1, commit_changes_2, msg_0, msg_1, msg_2)
    print(all_commits)
    return []


def get_children_nodes(ds, target_commit_id: str = ""):
    """
    Obtain the sub-node tree of the target commit ID.
    """
    if not target_commit_id:
        raise ValueError("target_commit_id cannot be empty.")

    commit_node = ds.version_state['commit_node_map'].get(FIRST_COMMIT_ID)
    if commit_node:
        return _find_subtree(ds, commit_node, target_commit_id)
    return None


def log(ds, ordered_by_date=False):
    """Displays the details of all the past commits."""
    commit_nodes = _get_commit_nodes(ds, ordered_by_date=ordered_by_date)
    print("---------------\nMULLER_F Version Log\n---------------\n")
    print(f"Current Branch: {ds.version_state['branch']}")
    for commit_node in commit_nodes:
        if not commit_node.is_head_node:
            print(f"{commit_node}\n")


def delete_branch(ds, name: str) -> None:
    """Delete a branch of the dataset."""
    _delete_branch(ds, name)
    integrity_check(ds)


def reset(ds, force: bool = False):
    """Resets the uncommitted changes present in the branch.

    Note:
        The uncommitted data is deleted from underlying storage, this is not a reversible operation.
    """

    storage, version_state = ds.storage, ds.version_state
    if version_state["commit_node"].children:
        print("You are not at the head node of the branch, cannot reset.")
        return
    if not ds.has_head_changes and not force:
        print("There are no uncommitted changes on this branch.")
        return

    if ds.commit_id is None:
        storage.clear()
        ds.populate_meta()
        load_meta(ds)
    else:
        parent_commit_id = ds.commit_id
        reset_commit_id = ds.pending_commit_id

        # checkout to get list of tensors in previous commit, needed for copying metas and create_commit_chunk_set
        try:
            ds.checkout(parent_commit_id)
        except Exception as e:
            raise e

        new_commit_id = replace_head(storage, version_state, ds.tensors, reset_commit_id)

        try:
            ds.checkout(new_commit_id)
        except Exception as e:
            raise e
    ds.check_uuid()


def parse_changes(ds, temp_diff, tensor_name, last_indexed_commit):
    """Parse the changes of target tensor.

    Args:
        ds: Dataset object.
        temp_diff (dict): Dictionary of commit diff that need to be parsed.
        tensor_name (String): The target tensor name.
        last_indexed_commit (String): Last indexed commit id of the target tensor.
    """
    tensor_changes = temp_diff["tensor"]
    temp_diff = {"deleted": set(), "added": dict(), "updated": dict()}
    for tensor_change in tensor_changes:
        change_stack = []
        for change in tensor_change:  # every commit
            change_stack.append(change)
        temp_diff = _process_diff_parse_changes(ds, change_stack, tensor_name, temp_diff)
    return temp_diff


def get_tensor_uuids(ds, tensor_name, target_commit_id) -> List[int]:
    """Retrieve all UUIDs of tensor_name in the version specified by target_commit_id, sorted in order.

    Note: This function is designed to fetch tensor UUIDs from a version other than the current one without checking
    out that version. It directly reads the UUID data from the target version and returns it.

    This implementation mirrors the logic used in the numpy() method of the tensor's _sample_id_tensor attribute,
    but its correctness is not guaranteed. If issues arise, fall back to the original approach:

    1. Check out the target version,
    2. Retrieve the UUIDs,
    3. Record the result,
    4. Then check out back to the original version to preserve the dataset’s state.
    Design Note: This function logically belongs in ChunkEngine,
    but due to historical reasons, it remains here as a temporary workaround.
    """

    # This line appears to be for compatibility with the unified UUID scheme (i.e., one UUID per row).
    tar_tensor_meta_key = get_tensor_meta_key(tensor_name, target_commit_id)
    try:
        # If tensor_meta cannot be retrieved, it means the tensor does not exist,
        # and an empty list should be returned for UUIDs.
        tar_tensor_meta = ds.storage.get_muller_object(tar_tensor_meta_key,
                                                       TensorMeta)
    except KeyError as e:
        tar_dataset_meta_key = get_dataset_meta_key(target_commit_id)
        tar_dataset_meta = ds.storage.get_muller_object(tar_dataset_meta_key,
                                                        DatasetMeta)
        if tensor_name not in tar_dataset_meta.tensors:
            return []
        raise KeyError(f"{tar_tensor_meta_key} is broken.") from e
    tensor_size = tar_tensor_meta.length

    # If it exists in the cache, read it directly from the cache.
    uuids = _read_from_upper_cache(ds, target_commit_id, tensor_name)
    if uuids is not None:
        return uuids[0:tensor_size]  # 截取tensor_size兼容统一uuid.

    # (1). Obtain the file paths of the chunks containing the UUID column data, following the implementation logic in
    # ChunkEngine. First, access the target version's directory to retrieve the names of all required chunks.
    # Then, since the actual storage paths of these chunks may not all be recorded in the current version,
    # further lookup is needed to locate their true physical locations.

    # First, obtain the tensor name corresponding to the UUID, ensuring compatibility with the unified UUID scheme
    # (where one UUID is assigned per row).
    hidden_tensor_name = DATASET_UUID_NAME if ds.use_dataset_uuid else get_sample_id_tensor_key(tensor_name)

    # 1、Retrieve the names of all required chunks.
    all_chunk_names = _get_chunk_names_from_path(ds, get_chunk_id_encoder_key(hidden_tensor_name, target_commit_id))
    uuid_list = []

    # 2、Read chunks directly from tensor_name/chunks without traversal.
    for chunk_name in all_chunk_names:  # Here, we could potentially read chunks in parallel using the
        # Provider.get_items interface—this is noted for future optimization but not implemented for now.
        uuid_list.extend(_deserialize_uuids(ds, get_chunk_key(hidden_tensor_name, chunk_name)))

    # Put in cache
    _update_upper_cache(ds, uuid_list, target_commit_id, tensor_name)
    return uuid_list[0:tensor_size]  # Truncate tensor_size to maintain compatibility with the unified UUID scheme.


def _delete_branch(ds, name: str) -> None:
    if ds.is_filtered_view:
        raise Exception(
            "Cannot perform version control operations on a filtered dataset view."
        )
    read_only = ds.read_only
    if read_only:
        raise ReadOnlyModeError()
    try_flushing(ds)
    ds.initial_autoflush.append(ds.storage.autoflush)
    ds.storage.autoflush = False
    try:
        unlock_dataset(ds)
        muller.util.version_control.delete_branch(ds, name)
    finally:
        ds.set_read_only(read_only, err=True)
        ds.storage.autoflush = ds.initial_autoflush.pop()


def _data_to_dataframe(original_dict, common_tensors, target_operations, force):
    # First fetch the uuids
    common_tensors = list(common_tensors)
    data = {}
    uuid_set = {}
    for tensor_name in common_tensors:
        # Note: In the case of append or delete operations, the IDs should generally be aligned, so we could directly
        # take the values from id_1 or id_2. However, to be safe—accounting for scenarios where some columns
        # might have undergone independent append or pop operations—we instead compute the union of the ID sets and
        # iterate over that.
        uuid_set = uuid_set | original_dict[tensor_name][target_operations].keys()

    # If the data volume is too large, an exception may be raised.
    if not force and len(uuid_set) > TO_DATAFRAME_SAFE_LIMIT:
        raise ExportDataFrameLimit(len(uuid_set), TO_DATAFRAME_SAFE_LIMIT)
    data.update({"uuid": list(uuid_set)})

    # Then, generate a DataFrame based on everyone's operations.
    if target_operations != "edited_values_tar1":
        tensor_dict = {}
        for tensor_name in common_tensors:
            temp = []
            for my_uuid in uuid_set:
                temp.append(original_dict[tensor_name][target_operations][my_uuid])
            tensor_dict.update({
                tensor_name: temp
            })

        for tensor_name in common_tensors:
            data.update({tensor_name: tensor_dict.get(tensor_name, [])})
        df = pd.DataFrame.from_dict(data)
    else:
        for tensor_name in common_tensors:
            temp_1 = []
            temp_2 = []
            for my_uuid in uuid_set:
                temp_1.append(original_dict[tensor_name]["edited_values_tar1"].get(my_uuid, None))
                temp_2.append(original_dict[tensor_name]["edited_values_tar2"].get(my_uuid, None))
            data.update({
                tensor_name + " (id_1)": temp_1
            })
            data.update({
                tensor_name + " (id_2)": temp_2
            })
        df = pd.DataFrame.from_dict(data)
    return df


def _read_from_upper_cache(ds, commit_id, tensor_name):
    if 'uuids' not in ds.storage.upper_cache:
        ds.storage.upper_cache['uuids'] = {}
    if commit_id in ds.storage.upper_cache['uuids']:
        if ds.use_dataset_uuid and DATASET_UUID_NAME in ds.storage.upper_cache['uuids'][commit_id]:
            return ds.storage.upper_cache['uuids'][commit_id][DATASET_UUID_NAME]
        if tensor_name in ds.storage.upper_cache['uuids'][commit_id]:
            return ds.storage.upper_cache['uuids'][commit_id][tensor_name]
    return None


def _process_diff_parse_changes(ds, change_stack, tensor_name, temp_diff):
    while len(change_stack) > 0:
        change = change_stack.pop()
        commit_id = change["commit_id"]
        uuid_list = ds.get_tensor_uuids(tensor_name, commit_id)  # new version
        uuid_list = [str(i) for i in uuid_list]  # int to str

        temp_diff["deleted"].update(change[tensor_name]["data_deleted_ids"])  # set of uuids
        add_index = list(range(change[tensor_name]["data_added"][0],
                               change[tensor_name]["data_added"][1]))  # index list
        add_uuids = [uuid_list[i] for i in add_index]
        add_values = change[tensor_name]["add_value"]
        add_dict = {add_uuids[i]: add_values[i].tolist() for i in range(len(add_uuids))}
        temp_diff["added"].update(add_dict)

        update_index = change[tensor_name]["data_updated"]
        update_uuids = [uuid_list[i] for i in update_index]
        updated_values = change[tensor_name]["updated_values"]
        update_dict = {update_uuids[i]: updated_values[i].tolist() for i in range(len(update_uuids))}
        temp_diff["updated"].update(update_dict)
    return temp_diff


def _get_commit_nodes(ds, target_node: CommitNode = None, ordered_by_date=False):
    visited_nodes = set()
    commit_nodes = []
    try:
        commit_node = ds.version_state["commit_node"]
    except KeyError as e:
        raise VersionControlError(f"Unable to obtain field on version state") from e
    commit_node = target_node if target_node else commit_node

    def traverse_commit_node(cur_commit_node):
        if cur_commit_node in visited_nodes:
            return
        visited_nodes.add(cur_commit_node)
        if cur_commit_node.merge_parent:
            try:
                merge_parent_node = ds.version_state["commit_node_map"][cur_commit_node.merge_parent]
            except KeyError as e1:
                raise VersionControlError(f"Unable to obtain field on version state") from e1
            traverse_commit_node(merge_parent_node)
        if cur_commit_node.parent:
            traverse_commit_node(cur_commit_node.parent)
        if not cur_commit_node.is_head_node:
            commit_nodes.append(cur_commit_node)

    if commit_node:
        traverse_commit_node(commit_node)
    commit_nodes = commit_nodes[::-1]
    if ordered_by_date:
        commit_nodes.sort(key=lambda x: x.commit_time, reverse=True)
    return commit_nodes


def _find_subtree(ds, node: CommitNode, target_commit_id: str):
    if not node:
        return None

    if node.commit_id == target_commit_id:
        return _subtree_to_dict(node)

    for child in node.children:
        result = _find_subtree(ds, child, target_commit_id)
        if result:
            return result

    return None


def _subtree_to_dict(node):
    if not node:
        return None

    subtree_dict = {
        'commit_id': node.commit_id,
        'children': []
    }

    for child in node.children:
        subtree_dict['children'].append(_subtree_to_dict(child))

    return subtree_dict


def _process_update_add_value_of_tensor(ds, tensor_names, tensors, commit_changes, offset, limit,
                                        commit_id, par_id, asrow):

    changes_records = {
        "changes_records_ranges": {},
        "changes_records_add": {},
        "changes_records_update": {},
        "changes_records_del": {}
    }
    for tname in tensor_names:
        if tensors is not None and len(tensors) > 0 and tname not in tensors:
            continue
        added_range, update_range, del_range = calcul_range(commit_changes[tname], offset, limit)

        chunk_global_idx, sorted_idx = _sort_chunk_pairs(ds, commit_id, tname)

        add_values = _get_add_or_update_values("append", ds, tname, commit_id, chunk_global_idx, sorted_idx,
                                               added_range)
        update_values = _get_add_or_update_values("update", ds, tname, commit_id, chunk_global_idx, sorted_idx,
                                                  update_range)
        delete_values, gids = _get_delete_values(ds, tname, par_id, del_range)

        if asrow:
            changes_records["changes_records_ranges"][tname] = (tuple(added_range),
                                                                tuple(sorted(update_range)),
                                                                tuple(sorted(gids)))
            changes_records["changes_records_add"][tname] = add_values
            changes_records["changes_records_update"][tname] = update_values
            changes_records["changes_records_del"][tname] = delete_values
        else:
            commit_changes[tname]['add_value'] = add_values
            commit_changes[tname]['updated_values'] = update_values
            commit_changes[tname]['data_deleted_values'] = delete_values
    return (changes_records["changes_records_ranges"],
            changes_records["changes_records_add"],
            changes_records["changes_records_update"],
            changes_records["changes_records_del"],
            commit_changes)


def _get_add_or_update_values(mode, ds, tname, commit_id, chunk_global_idx, sorted_idx, my_range):
    if mode == "append":
        records = handle_append_ranges(chunk_global_idx, sorted_idx, my_range)
        values = generate_add_values(ds.storage, tname, commit_id, records)
    else:
        records = handle_update_ranges(chunk_global_idx, sorted_idx, my_range)
        values = generate_update_values(ds.storage, tname, commit_id, records)
    return values


def _get_delete_values(ds, tname, par_id, del_range):
    if len(del_range) == 0:
        delete_records = {}
        gids = []
    else:
        if not par_id:
            raise ValueError(f"Cannot without parent version but have popped samples.")
        par_chunk_gid, par_indexes = _sort_chunk_pairs(ds, par_id, tname)
        delete_records, gids = handle_delete_ranges(par_indexes,
                                                    par_chunk_gid,
                                                    ds.get_tensor_uuids(tname, par_id),
                                                    del_range)
    delete_values = generate_delete_values(ds.storage, tname, par_id, delete_records)
    return delete_values, gids


def _sort_chunk_pairs(ds, commit_id, tensor):
    chunk_encoder_path = get_chunk_id_encoder_key(tensor, commit_id)
    chunk_pairs = _get_chunk_names_from_path(ds, chunk_encoder_path, as_pair=True)
    index_sorted = chunk_pairs[:, 1]
    if (len(index_sorted) > 1 and index_sorted[-1] > index_sorted[-2]) or len(index_sorted) == 1:
        return chunk_pairs, chunk_pairs[:, 1]
    pairs_sorted = np.sort(chunk_pairs, axis=1, kind='quicksort')
    index_sorted = pairs_sorted[:, 0]
    return pairs_sorted, index_sorted


def _get_chunk_names_from_path(ds, chunks_index_path, as_pair=False):
    chunk_ids = ds.storage[chunks_index_path]
    if isinstance(chunk_ids, ChunkIdEncoder):
        ids = chunk_ids.encoded
    else:  # bytes
        _, ids, _ = deserialize_chunkids(chunk_ids)
    if as_pair:
        return ids
    all_chunk_names = [hex(item[0]).split('x')[-1] for item in ids]  # complete chunk names of this version
    return all_chunk_names


def _deserialize_uuids(ds, chunk_path):
    """Deserialize uuids given the chunk path."""
    chunks = ds.storage[chunk_path]
    if isinstance(chunks, UncompressedChunk):
        data = chunks.data_bytes
    else:  # type : bytes
        _, _, _, data = deserialize_chunk(chunks)
    uuids = np.frombuffer(bytearray(data), 'uint64').tolist()
    return uuids


def _update_upper_cache(ds, uuids, commit_id, tensor_name):
    if commit_id not in ds.storage.upper_cache['uuids']:
        ds.storage.upper_cache['uuids'][commit_id] = {}
    if ds.use_dataset_uuid and DATASET_UUID_NAME not in ds.storage.upper_cache['uuids'][commit_id]:
        ds.storage.upper_cache['uuids'][commit_id][DATASET_UUID_NAME] = {}
    elif tensor_name not in ds.storage.upper_cache['uuids'][commit_id]:
        ds.storage.upper_cache['uuids'][commit_id][tensor_name] = {}
    if ds.use_dataset_uuid:
        ds.storage.upper_cache['uuids'][commit_id][DATASET_UUID_NAME] = uuids
    else:
        ds.storage.upper_cache['uuids'][commit_id][tensor_name] = uuids
