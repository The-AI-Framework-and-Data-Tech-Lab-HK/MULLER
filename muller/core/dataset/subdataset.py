# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Subdataset loading utilities for Dataset."""

from typing import Optional

import muller.core.dataset
from muller.constants import DEFAULT_LOCAL_CACHE_SIZE, DEFAULT_MEMORY_CACHE_SIZE, MB
from muller.core.storage.cache_chain import generate_chain


def sub_ds(
        dataset,
        path,
        empty=False,
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        read_only=None,
        verbose=True,
):
    """Loads a nested dataset. Internal."""
    sub_storage = dataset.base_storage.subdir(path, read_only=read_only)

    if empty:
        sub_storage.clear()

    # IMPORTANT: do NOT pass ``sub_storage`` as the ``path=`` kwarg to
    # ``Dataset(...)``. The constructor expects a ``str | pathlib.Path``
    # and stores it on ``self.path``; assigning a ``StorageProvider`` there
    # makes any downstream call site that treats ``ds.path`` as a string
    # (``ds.path.startswith("mem://")``, ``os.path.join(ds.path, ...)``,
    # the view-entry reload chain that funnels through
    # ``storage_factory(path)``, …) raise
    # ``AttributeError: 'LocalProvider' object has no attribute 'startswith'``.
    # Leaving ``path=None`` lets ``Dataset.__init__`` derive a real string
    # path from the storage via ``get_path_from_storage``.
    cls = muller.core.dataset.Dataset

    ret = cls(
        generate_chain(
            sub_storage,
            memory_cache_size * MB,
            local_cache_size * MB,
        ),
        read_only=read_only,
        verbose=verbose,
    )
    ret.parent_dataset = dataset
    return ret
