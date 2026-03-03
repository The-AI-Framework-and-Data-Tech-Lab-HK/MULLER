# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Modifications Copyright (c) 2026 Xueling Lin

"""Dataset copy operations: like."""

from typing import List, Optional, Union

from muller.core.dataset.dataset import Dataset
from muller.util.path import convert_pathlib_to_string_if_needed


def like(
        dest: Union[str, Dataset],
        src: Union[str, Dataset],
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        verbose: bool = True,
) -> Dataset:
    """Copies the `source` dataset's structure to a new location.
    No samples are copied, only the meta/info for the dataset and its tensors.

    Args:
        dest(Union[str, Dataset]): Empty Dataset or Path where the new dataset will be created.
        src (Union[str, Dataset]): Path or dataset object that will be used as the template for the new dataset.
        tensors (List[str], optional): Names of tensors (and groups) to be replicated.
            If not specified all tensors in source dataset are considered.
        overwrite (bool): If True and a dataset exists at `destination`, it will be overwritten. Defaults to False.
        verbose (bool): If True, logs will be printed. Defaults to ``True``.

    Returns:
        Dataset: New dataset object.
    """
    # Import here to avoid circular dependency
    from muller.api.dataset.core import load, empty

    if isinstance(src, str):
        src = convert_pathlib_to_string_if_needed(src)
        source_ds = load(src, verbose=verbose)
        src_path = src
    else:
        source_ds = src
        src_path = src.path

    if tensors:
        tensors = source_ds.resolve_tensor_list(tensors)  # type: ignore
    else:
        tensors = list(source_ds.tensors.keys())  # type: ignore

    if isinstance(dest, Dataset):
        destination_ds = dest
    else:
        dest = convert_pathlib_to_string_if_needed(dest)
        dest_path = dest
        if dest_path == src_path:
            destination_ds = load(dest_path, read_only=False)
        else:
            destination_ds = empty(dest_path, overwrite=overwrite)  # type: ignore

    for tensor_name in tensors:  # type: ignore
        source_tensor = source_ds[tensor_name]
        if overwrite and tensor_name in destination_ds:
            destination_ds.delete_tensor(tensor_name)
        destination_ds.create_tensor_like(tensor_name, source_tensor)  # type: ignore

    destination_ds.info.update(source_ds.info.__getstate__())  # type: ignore

    return destination_ds
