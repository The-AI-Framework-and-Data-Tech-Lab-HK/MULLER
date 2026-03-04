# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Version control mixin for Dataset class."""

from typing import Dict, List, Optional

import muller.core.version_control
from muller.core.auth.permission.invalid_view_op import invalid_view_op
from muller.core.auth.permission.user_permission_check import user_permission_check
from muller.core.version_control.commit_node import CommitNode
from muller.util.exceptions import EmptyCommitError
from muller.util.iteration_warning import suppress_iteration_warning
from muller.util.spinner import spinner


class VersionControlMixin:
    """Mixin providing version control operations for Dataset."""

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
        """Detect the conflict between current stage and target stage of given commit id."""
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
        """Returns/displays the differences between the given commit/current commit and its previous commit."""
        return muller.core.version_control.diff_to_prev(self, commit_id, as_dict, show_value, offset, limit, asrow)

    def commits_under(
            self,
            branch: str = None,
            ordered_by_date: bool = False
    ) -> List[CommitNode]:
        """Return the list of commits under the given branch."""
        return muller.core.version_control.commits_under(self, branch, ordered_by_date)

    def commits_between(self, id_1: Optional[str] = None, id_2: Optional[str] = None, as_dict=False):
        """Show the commits history between given ids or branch names."""
        return muller.core.version_control.commits_between(self, id_1, id_2, as_dict=as_dict)

    def get_children_nodes(self, target_commit_id: str = ""):
        """Obtain the sub-node tree of the target commit ID."""
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
        Note: The uncommitted data is deleted from underlying storage, this is not a reversible operation.
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

    def tensor_diff(self, id_1, id_2, tensors: List[str] = None):
        """Displays the differences between commits (in the same branch) for certain tensor."""
        from muller.core.version_control.operations.diff import get_changes_and_messages

        version_state, storage = self.version_state, self.storage
        res = get_changes_and_messages(version_state, storage, id_1, id_2)
        tensor_changes = res[3]
        if tensor_changes is not None and len(tensor_changes) > 0:
            for tensor_change_ver in tensor_changes:
                _ = self.generate_add_update_value(tensor_change_ver, 0, None, False, tensors)

        changes = {"tensor": (tensor_changes,)}
        return changes
