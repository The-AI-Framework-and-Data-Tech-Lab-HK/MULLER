# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import pandas as pd
from muller.constants import TO_DATAFRAME_SAFE_LIMIT
from muller.util.exceptions import InvalidTensorList, ToDataFrameLimit


def to_dataframe(dataset, tensor_list=None, index_list=None, force=False):
    """ Export dataset as a pandas dataframe.

    Args:
        dataset: The dataset to export.
        tensor_list (List of str, Optional): The tensor columns selected to be exported as pandas dataframe.
                    If not provided, we will export all the tensor columns.
        index_list (List of int, Optional): The indices of the rows selected to be exported as pandas dataframe.
                    If not provided, we will export all the row.
        force (bool, Optional): Dataset with more than TO_DATAFRAME_SAFE_LIMIT samples might take a long time to
                    export. If force = True, the dataset will be exported regardless.
                    An error will be raised otherwise.

    Raises:
        InvalidTensorList: If ``tensor_list`` contains tensors that are not in the current columns.
        ToDataFrameLimit: If the length of ``index_list`` exceeds the TO_DATAFRAME_SAFE_LIMIT.
    """
    # Verify that the target column is correctly specified.
    if tensor_list and (len(tensor_list) > len(dataset.tensors) or
                        not all(isinstance(x, str) and x in dataset.tensors for x in tensor_list)):
        raise InvalidTensorList(tensor_list)

    max_num = -1
    if index_list and len(index_list) > TO_DATAFRAME_SAFE_LIMIT and not force:
        max_num = len(index_list)
    elif dataset.max_len > TO_DATAFRAME_SAFE_LIMIT and not force:
        max_num = dataset.max_len
    if max_num != -1:
        raise ToDataFrameLimit(max_num, TO_DATAFRAME_SAFE_LIMIT)

    data = {}
    if tensor_list:
        target_list = tensor_list
    else:
        target_list = list(dataset.tensors)
    if index_list:
        for tensor in target_list:
            data.update({tensor: list(dataset[index_list][tensor].numpy().flatten())})  # TODO: use numpy_continuous
    else:
        for tensor in target_list:
            data.update({tensor: list(dataset[tensor].numpy().flatten())})  # TODO: use numpy_full (risk of file lock)
    df = pd.DataFrame(data, columns=target_list)
    return df
