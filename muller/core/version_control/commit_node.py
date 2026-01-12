# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/version_control/commit_node.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from datetime import datetime
from typing import List, Optional
from muller.util.authorization import obtain_current_user


class CommitNode:
    """Contains all the Version Control information about a particular commit."""

    def __init__(self, branch: str, commit_id: str, total_samples_processed: int = 0,
                 commit_user_name: Optional[str] = None):
        self.commit_id = commit_id
        self.branch = branch
        self.children: List["CommitNode"] = []
        self.parent: Optional["CommitNode"] = None
        self.commit_message: Optional[str] = None
        self.commit_time: Optional[datetime] = None
        if commit_user_name:
            self.commit_user_name = commit_user_name
        else:
            self.commit_user_name = obtain_current_user()
        self.merge_parent: str = ""
        self._info_updated: bool = False
        self.is_checkpoint: bool = False
        self.total_samples_processed: int = total_samples_processed
        self.checkout_node: bool = False

    def __hash__(self):
        return hash(self.commit_id)

    def __eq__(self, other):
        if isinstance(other, CommitNode):
            return self.commit_id == other.commit_id
        return False

    def __repr__(self) -> str:
        return (
            f"Commit : {self.commit_id} ({self.branch}) \nAuthor : {self.commit_user_name}\nTime   : "
            f"{str(self.commit_time)[:-7]}\nMessage: {self.commit_message}"
            + (
                f"\nTotal samples processed in transform: {self.total_samples_processed}"
                if self.is_checkpoint
                else ""
            )
        )

    __str__ = __repr__

    @property
    def is_head_node(self) -> bool:
        """Returns True if the node is the head node of the branch."""
        if self.commit_time is None:
            return True
        return not self.children

    @property
    def is_merge_node(self):
        """Returns True if the node is a merge node."""
        return len(self.merge_parent) != 0

    @property
    def info_updated(self):
        """Returns self._info_updated value."""
        return self._info_updated

    @info_updated.setter
    def info_updated(self, value):
        """Set self._info_updated value."""
        self._info_updated = value

    def add_child(self, node: "CommitNode"):
        """Adds a child to the node, used for branching."""
        node.parent = self
        self.children.append(node)

    def copy(self):
        """Copy commit node. """
        node = CommitNode(self.branch, self.commit_id)
        node.commit_message = self.commit_message
        node.commit_user_name = self.commit_user_name
        node.commit_time = self.commit_time
        node.merge_parent = self.merge_parent
        node.is_checkpoint = self.is_checkpoint
        node.total_samples_processed = self.total_samples_processed
        node.checkout_node = self.checkout_node
        return node

    def merge_from(self, node: "CommitNode"):
        """Merges the given node into this node."""
        self.merge_parent = node.commit_id

    def to_json(self):
        """Converts this node to a JSON string."""
        return {
            "branch": self.branch,
            "children": [node.commit_id for node in self.children],
            "parent": self.parent.commit_id if self.parent else None,
            "commit_message": self.commit_message,
            "commit_time": self.commit_time.timestamp() if self.commit_time else None,
            "commit_user_name": self.commit_user_name,
            "merge_parent": self.merge_parent,
            "is_checkpoint": self.is_checkpoint,
            "total_samples_processed": self.total_samples_processed,
            "checkout_node": self.checkout_node,
        }

    def add_successor(
        self, node: "CommitNode", author: str, message: Optional[str] = None
    ):
        """Adds a successor (a type of child) to the node, used for commits."""
        node.parent = self
        self.children.append(node)
        self.commit_message = message
        self.commit_user_name = author
        self.commit_time = datetime.utcnow()

    def add_checkout_node(self):
        """Function to add checkout meta info"""
        self.commit_time = datetime.utcnow()
        self.checkout_node = True
