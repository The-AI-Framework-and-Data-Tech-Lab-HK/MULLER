# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

from typing import Union

import numpy as np


def to_numpy(ds, aslist=False, fetch_chunks=False, asrow=False) -> Union[dict, list]:
    """
    Computes the contents of the dataset slices in numpy format.

    Args:
        aslist (bool): If ``True``, a list of np.ndarrays will be returned for each sample. Helpful for dynamic
            tensors. If ``False``, a single np.ndarray will be returned unless the samples are dynamically shaped,
            in which case an error is raised.
        fetch_chunks (bool): If ``True``, full chunks will be retrieved from the storage,
            otherwise only required bytes will be retrieved.
            This will always be ``True`` even if specified as ``False`` in the following cases:
            - The tensor is ChunkCompressed.
            - The chunk which is being accessed has more than 128 samples.
        asrow (bool): If ``True``, return the value of dataset slices in rows, a list of dict will be returned for
            each row of data, in which case the length of tensors is different or not equal to length of dataset
            index an error will be raised.
            If ``False``, return the value of dataset slices in columns, a dict of list which regards the
            tensor_name as keys will be returned.

    Raises:
        DynamicTensorNumpyError: If reading a dynamically-shaped array slice without ``aslist=True``.
        ValueError: If the tensor is a link and the credentials are not populated.
        ValueError: If number of samples in each tensor is different or the number not equal to the length
            of dataset index.

    Returns:
        A numpy array containing the data represented by this tensor.

    Note:
        For tensors of htype ``polygon``, aslist is always ``True``.
    """
    values_col = {}
    for tensor_name in ds.tensors.keys():
        if ds.tensors[tensor_name].index.length(ds.num_samples) == 1:
            if ds.tensors[tensor_name].index.is_slice:
                values_col[tensor_name] = ds.tensors[tensor_name].numpy(aslist=aslist, fetch_chunks=fetch_chunks)
            else:
                values_col[tensor_name] = np.expand_dims(ds.tensors[tensor_name].numpy(
                    aslist=aslist, fetch_chunks=fetch_chunks), 0)

        else:
            values_col[tensor_name] = ds.tensors[tensor_name].numpy(aslist=aslist, fetch_chunks=fetch_chunks)
    if asrow:
        if _check_values(ds, values_col):
            values_rows = [dict(zip(values_col, t)) for t in zip(*values_col.values())]
            return values_rows

        raise ValueError("The number of samples in each tensor is different or "
                         "the number not equal to the length of dataset index. Please set asrow = False.")
    return values_col


def _check_values(ds, data_values: dict):
    """Check the values dict validation"""
    samples_num = False
    if all(len(list(data_values.values())[0]) == len(v) for v in list(data_values.values())[1:]):
        if len(list(data_values.values())[0]) == ds.index.length(ds.num_samples):
            samples_num = True
    # showing data in row needs: 1. the length of values in each key is same 2. the number of samples
    # should equal to length of index 3. number of keys in the dict should equal to number of tensors
    return (
            len(data_values.keys()) == len(ds.tensors)
            and samples_num
    )
