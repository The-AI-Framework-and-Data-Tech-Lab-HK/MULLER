# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/version_control/dataset_diff.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from muller.core.storage.muller_memory_object import MULLERMemoryObject
import typing
from collections import OrderedDict


class DatasetDiff(MULLERMemoryObject):
    def __init__(self) -> None:
        super().__init__()
        self.info_updated = False
        self.renamed: typing.OrderedDict = OrderedDict()
        self.deleted: typing.List[str] = []
        self.commit_diff_exist = False
        self.tensor_info_updated = False

    def tobytes(self) -> bytes:
        return b"".join(
            [
                self.info_updated.to_bytes(1, "big"),
                len(self.renamed).to_bytes(8, "big"),
                *(
                    b"".join(
                        [
                            len(old_name).to_bytes(8, "big"),
                            len(new_name).to_bytes(8, "big"),
                            (old_name + new_name),
                        ]
                    )
                    for old_name, new_name in map(
                        lambda n: (n[0].encode("utf-8"), n[1].encode("utf-8")),
                        self.renamed.items(),
                    )
                ),
                len(self.deleted).to_bytes(8, "big"),
                *(
                    b"".join([len(name).to_bytes(8, "big"), name.encode("utf-8")])
                    for name in self.deleted
                ),
                self.commit_diff_exist.to_bytes(1, "big"),
                self.tensor_info_updated.to_bytes(1, "big"),
            ]
        )

    @classmethod
    def frombuffer(cls, data: bytes) -> "DatasetDiff":
        """Creates a DatasetDiff object from bytes"""
        dataset_diff = cls()
        dataset_diff.info_updated = bool(int.from_bytes(data[:1], "big"))
        len_renamed = int.from_bytes(data[1:9], "big")
        pos = 9
        for _ in range(len_renamed):
            len_old, len_new = (
                int.from_bytes(data[pos : pos + 8], "big"),
                int.from_bytes(data[pos + 8 : pos + 16], "big"),
            )
            pos += 16
            old_name, new_name = (
                data[pos : pos + len_old].decode("utf-8"),
                data[pos + len_old : pos + len_old + len_new].decode("utf-8"),
            )
            pos += len_old + len_new
            dataset_diff.renamed[old_name] = new_name
        len_deleted = int.from_bytes(data[pos : pos + 8], "big")
        pos += 8
        for _ in range(len_deleted):
            len_name = int.from_bytes(data[pos : pos + 8], "big")
            pos += 8
            name = data[pos : pos + len_name].decode("utf-8")
            pos += len_name
            dataset_diff.deleted.append(name)
        dataset_diff.commit_diff_exist = bool(int.from_bytes(data[17:18], "big"))
        dataset_diff.tensor_info_updated = bool(int.from_bytes(data[18:19], "big"))
        dataset_diff.is_dirty = False
        return dataset_diff

    @property
    def nbytes(self):
        """Returns number of bytes required to store the dataset diff"""
        return 1

    def modify_info(self) -> None:
        """Stores information that the info has changed"""
        self.info_updated = True
        self.is_dirty = True

    def modify_tensor_info(self) -> None:
        """Stores information that the tensor info has changed"""
        self.tensor_info_updated = True
        self.is_dirty = True

    def tensor_renamed(self, old_name, new_name):
        """Adds old and new name of a tensor that was renamed to renamed"""
        for old, new in self.renamed.items():
            if old_name == new:
                if old == new_name:
                    self.renamed.pop(old)
                else:
                    self.renamed[old] = new_name
                break
        else:
            self.renamed[old_name] = new_name
        self.is_dirty = True

    def tensor_deleted(self, name):
        """Adds name of deleted tensor to deleted"""
        if name not in self.deleted:
            for old, new in self.renamed.items():
                if name == new:
                    self.renamed.pop(old)
                    self.deleted.append(old)
                    break
                self.deleted.append(name)
            self.is_dirty = True

    def set_commit_diff_exist(self):
        """Records the commit_diff created in one of the tensors of the dataset"""
        self.commit_diff_exist = True
        self.is_dirty = True
