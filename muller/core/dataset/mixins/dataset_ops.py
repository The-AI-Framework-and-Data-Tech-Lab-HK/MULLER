# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Dataset-level operations mixin for Dataset class."""

import pathlib
from typing import Union

import muller.core.dataset
from muller.core.auth.permission.invalid_view_op import invalid_view_op
from muller.core.auth.permission.user_permission_check import (
    user_permission_check,
)


class DatasetOpsMixin:
    """Mixin providing dataset-level operations (not tensor-specific)."""

    @invalid_view_op
    @user_permission_check
    def delete(self, large_ok=False):
        """Delete the dataset."""
        muller.core.dataset.delete(self, large_ok)

    @invalid_view_op
    @user_permission_check
    def rename(self, path: Union[str, pathlib.Path]):
        """Renames the dataset to `path`."""
        # Note: currently we only accept the rename operation in LocalProvider and MemProvider
        muller.core.dataset.rename(self, path)
