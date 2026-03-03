# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Tensor operations mixin for Dataset class."""

import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np

import muller.core.dataset
from muller.constants import UNSPECIFIED
from muller.core.auth.permission.invalid_view_op import invalid_view_op
from muller.core.auth.permission.user_permission_check import user_permission_check


class TensorOpsMixin:
    """Mixin providing tensor operations for Dataset."""

    def create_tensor(
            self,
            name: str,
            htype: str = UNSPECIFIED,
            dtype: Union[str, np.dtype] = UNSPECIFIED,
            sample_compression: Union[str, None] = UNSPECIFIED,
            chunk_compression: str = UNSPECIFIED,
            hidden: bool = False,
            **kwargs,
    ):
        """Create tensors."""
        return muller.core.dataset.create_tensor(self, name, htype, dtype, sample_compression,
                                                                  chunk_compression, hidden,
                                                                  **kwargs)

    @invalid_view_op
    @user_permission_check
    def create_tensor_like(
            self, name: str, source: "Tensor",
    ) -> "Tensor":
        """
        Copies the ``source`` tensor's meta information and creates a new tensor with it. No samples are copied,
        only the meta/info for the tensor is.
        """
        return muller.core.dataset.create_tensor_like(self, name, source)

    @invalid_view_op
    @user_permission_check
    def delete_tensor(self, name: str, large_ok: bool = False):
        """Delete a tensor."""
        return muller.core.dataset.delete_tensor(self, name, large_ok)

    @user_permission_check
    def extend(
            self,
            samples: Dict[str, Any],
            skip_ok: bool = False,
            append_empty: bool = False,
            ignore_errors: bool = False,
            progressbar: bool = False,
    ):
        """Extend samples to the dataset."""
        muller.core.dataset.extend(self, samples, skip_ok, append_empty, ignore_errors, progressbar)

    @invalid_view_op
    @user_permission_check
    def append(
            self,
            sample: Dict[str, Any],
            skip_ok: bool = False,
            append_empty: bool = False,
    ):
        """Append samples to the dataset."""
        muller.core.dataset.append(self, sample, skip_ok, append_empty)

    @user_permission_check
    def update(self, sample: Dict[str, Any]):
        """Update samples in the dataset."""
        muller.core.dataset.update(self, sample)

    @invalid_view_op
    @user_permission_check
    def pop(self, index: Optional[Union[List, int]] = None, rechunk: bool = False):
        """Pop samples in the dataset."""
        muller.core.dataset.pop(self, index, rechunk)

    def handle_rename_tensor(self, name, new_name):
        """Function to handle rename tensor"""
        muller.core.dataset.handle_rename_tensor(self, name, new_name)

    @invalid_view_op
    @user_permission_check
    def rename_tensor(self, name: str, new_name: str):
        """Renames tensor with name ``name`` to ``new_name``"""
        return muller.core.dataset.rename_tensor(self, name, new_name)

    @user_permission_check
    def add_data_from_file(self, ori_path="", schema=None, workers=0, scheduler="processed", disable_rechunk=True,
                           progressbar=True, ignore_errors=True):
        """Add samples from external files to the dataset."""
        if not ori_path:
            raise ValueError("ori_path cannot be empty.")

        import muller.api.dataset_api
        org_dicts = muller.api.dataset_api.DatasetAPI.get_data_with_dict_from_file(ori_path, schema)
        return muller.core.dataset.add_data(self, org_dicts, schema, workers, scheduler, disable_rechunk, progressbar,
                                           ignore_errors)

    @user_permission_check
    def add_data_from_dataframes(self, dataframes=None, schema=None, workers=0, scheduler="processed",
                                 disable_rechunk=True, progressbar=True, ignore_errors=True):
        """Add samples from external dataframes to the dataset."""
        if not dataframes:
            raise ValueError("dataframes cannot be empty.")
        if not isinstance(dataframes, list):
            raise TypeError("Expected a list for dataframes")

        import muller.api.dataset_api
        org_dicts = muller.api.dataset_api.DatasetAPI.get_data_with_dict_from_dataframes(dataframes, schema)
        return muller.core.dataset.add_data(self, org_dicts, schema, workers, scheduler, disable_rechunk, progressbar,
                                           ignore_errors)

