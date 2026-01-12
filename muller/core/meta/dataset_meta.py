# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/meta/dataset_meta.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from typing import Any, Dict
from muller.core.meta.meta import Meta
from muller.core.index import Index


class DatasetMeta(Meta):
    """Stores dataset metadata."""

    def __init__(self):
        super().__init__()
        self.statistics = {}
        self.tensors = []
        self.tensor_names = {}
        self.hidden_tensors = []
        self.default_index = Index().to_json()
        self.info = {}
        self.dataset_creator = "public"

    def __getstate__(self) -> Dict[str, Any]:
        d = super().__getstate__()
        d["statistics"] = self.statistics.copy()
        d["tensors"] = self.tensors.copy()
        d["tensor_names"] = self.tensor_names.copy()
        d["hidden_tensors"] = self.hidden_tensors.copy()
        d["default_index"] = self.default_index.copy()
        d["info"] = self.info.copy()
        d["dataset_creator"] = self.dataset_creator
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @property
    def visible_tensors(self):
        """Returns list of tensors that are not hidden."""
        return list(
            filter(
                lambda t: self.tensor_names[t] not in self.hidden_tensors,
                self.tensor_names.keys(),
            )
        )

    @property
    def nbytes(self):
        """Returns size of the metadata stored in bytes."""
        # Sherry: can optimize this?
        return len(self.tobytes())

    def add_tensor(self, name, key, hidden=False):
        """Reflect addition of tensor in dataset's meta."""
        if key not in self.tensors:
            self.tensor_names[name] = key
            self.tensors.append(key)
            if hidden:
                self.hidden_tensors.append(key)
            self.is_dirty = True

    def delete_tensor(self, name):
        """Reflect tensor deletion in dataset's meta."""
        key = self.tensor_names.pop(name)
        self.tensors.remove(key)
        try:
            self.hidden_tensors.remove(key)
        except ValueError:
            pass
        self.is_dirty = True

    def rename_tensor(self, name, new_name):
        """Reflect a tensor rename in dataset's meta."""
        key = self.tensor_names.pop(name)
        self.tensor_names[new_name] = key
        self.is_dirty = True
        return self.tensor_names.copy()

    def set_dataset_creator(self, dataset_creator):
        """set dataset creator"""
        self.dataset_creator = dataset_creator
        self.is_dirty = True
