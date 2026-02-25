# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/api/info.py
#
# Modifications Copyright (c) 2026 Xueling Lin

from typing import Any, Dict

from muller.core.storage.muller_memory_object import MULLERMemoryObject
from muller.util.exceptions import InfoError


class Info(MULLERMemoryObject):
    """Contains optional key-value pairs that can be stored for datasets/tensors."""

    def __init__(self, dataset, key: str = None):
        super().__init__()
        self._dataset = dataset
        # the key to info in case of Tensor Info, None in case of Dataset Info
        self._key = key

        if key:
            self._info = dataset[key].meta.info
        else:
            self._info = dataset.meta.info
        self.is_dirty = False

    def __enter__(self):
        from muller.core.tensor import Tensor

        ds = self._dataset
        key = self._key
        if ds is not None:
            ds.storage.check_readonly()
            if not ds.version_state["commit_node"].is_head_node:
                raise InfoError("Cannot modify info from a non-head commit.")
            if key:
                Tensor(key, ds).chunk_engine.commit_diff.modify_info()
                ds._dataset_diff.modify_tensor_info()
            else:
                ds._dataset_diff.modify_info()
            self.is_dirty = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dataset is not None:
            self._dataset.maybe_flush()

    def __len__(self):
        return len(self._info)

    def __getstate__(self) -> Dict[str, Any]:
        return self._info

    def __setstate__(self, state: Dict[str, Any]):
        self._info = state

    # implement all the methods of dictionary
    def __getitem__(self, key: str):
        return self._info[key]

    def __str__(self):
        return self._info.__str__()

    def __repr__(self):
        return self._info.__repr__()

    def __setitem__(self, key, value):
        with self:
            self._info[key] = value

    def __delitem__(self, key):
        with self:
            del self._info[key]

    def __contains__(self, key):
        return key in self._info

    def __iter__(self):
        return iter(self._info)

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key == "_info":
                info = {}
                self._info = info
                return info
            try:
                return self._info[key]
            except KeyError:
                raise KeyError from e

    def __setattr__(self, key: str, value):
        if key in {"_info", "_dataset", "_key", "is_dirty"}:
            object.__setattr__(self, key, value)
        else:
            with self:
                self[key] = value

    @property
    def nbytes(self):
        """Returns size of info stored in bytes."""
        return len(self.tobytes())

    def get(self, key, default=None):
        """Get value for key from info."""
        return self._info.get(key, default)

    def setdefault(self, key, default=None):
        """Set default value for a key in info."""
        with self:
            ret = self._info.setdefault(key, default)
        return ret

    def clear(self):
        """Clear info."""
        with self:
            self._info.clear()

    def pop(self, key, default=None):
        """Pop item from info by key."""
        with self:
            popped = self._info.pop(key, default)
        return popped

    def popitem(self):
        """Pop item from info."""
        with self:
            popped = self._info.popitem()
        return popped

    def update(self, *args, **kwargs):
        """Update info."""
        with self:
            self._info.update(*args, **kwargs)
            if self._key:
                self._dataset[self._key].meta.info.update(*args, **kwargs)
                self._dataset[self._key].meta.is_dirty = True
            else:
                self._dataset.meta.info.update(*args, **kwargs)
                self._dataset.meta.is_dirty = True

    def keys(self):
        """Return all keys in info."""
        return self._info.keys()

    def values(self):
        """Return all values in info."""
        return self._info.values()

    def items(self):
        """Return all items in info."""
        return self._info.items()

    def replace_with(self, d):
        """Replace info with another dictionary."""
        with self:
            self._info.clear()
            self._info.update(d)
