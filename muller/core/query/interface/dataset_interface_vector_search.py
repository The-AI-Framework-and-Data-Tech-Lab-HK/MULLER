# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import warnings
from typing import Union

from numpy.typing import NDArray

from muller.core.tensor import Tensor
from muller.client.log import logger


def create_vector_index(ds, tensor_name: str, index_name: str, index_type: str = 'FLAT', metric: str = 'l2',
                        **kwargs: Union[int, float, str]):
    """
    Create index for tensor in vector type.

    When ``index_name`` is not in the index of ``vector_tensor_name`` then create the full index.
    Otherwise, update the ``index_name`` of the ``vector_tensor_name``.

    Args:
        ds (Dataset): Dataset of the vector index.
        tensor_name (str): The tensor need to create index.
        index_name (str): Name of the index.
        index_type (str): Type of the index, such as `IVFPQ`, `FLAT` etc.
        metric (str): Metric to measure the distance between the vectors, such as `l2`, `cosine` etc.
        **kwargs: Hyperparameters to create KNN index.
    """

    if ds.has_head_changes:
        # if there are some change not committed, then warning and do nothins.
        warnings.warn(
            "There are uncommitted changes, try again after commit."
        )
    else:
        overwrite = kwargs.get("overwrite", False)
        vector_tensor = ds.tensors.get(tensor_name)
        tensor_index = _get_vector_index(ds)
        # get saved dataset commit id
        commit_id = ds.commit_id
        tensor_index.create_vector_index(tensor=vector_tensor, index_name=index_name, index_type=index_type,
                                         metric=metric, commit_id=commit_id, overwrite=overwrite,
                                         **kwargs)


def update_vector_index(ds, tensor_name: str, index_name: str) -> None:
    """
    Update vector index after add some samples into dataset and commit.
    Args:
        ds: The dataset of which the vector index to be updated.
        tensor_name: The name of the vector tensor.
        index_name: The name of the vector index.
    """
    try:
        from muller.core.vector.exceptions import IndexNotFoundError
    except Exception as e:
        raise ModuleNotFoundError("please install dependency for vector search first.") from e
    if ds.has_head_changes:
        # if there are some change not committed, then warning and do nothins.
        warnings.warn(
            "There are uncommitted changes, try again after commit."
        )
    else:
        vector_tensor = ds.tensors.get(tensor_name)
        tensor_index = _get_vector_index(ds)
        try:
            vector_index = tensor_index.get_vector_index(tensor=vector_tensor, index_name=index_name)
            index_commit_id = vector_index.commit_id
            if index_commit_id != ds.commit_id:
                # update index when index is exists and not overwrite
                # this new commit change the tensor
                tensor_diff = ds.tensor_diff(index_commit_id, ds.commit_id, [tensor_name])
                tensor_changes = ds.parse_changes(tensor_diff, tensor_name, index_commit_id)
                tensor_index.update_index(tensor_changes, vector_tensor, index_name, ds.commit_id)
        except IndexNotFoundError as e:
            raise IndexNotFoundError from e


def vector_search(ds, query_vector: Union[NDArray, Tensor], tensor_name: str, index_name: str, **kwargs):
    """
    KNN Search on vector index.

    Args:
        ds (Dataset): Dataset of the vector index.
        query_vector (NDArray | Tensor): Query vector to find similar vector.
        tensor_name (str): Tensor name of vector index.
        index_name (str): Index name of vector index.
        **kwargs: Hyperparameters to search on vector index.

    Returns:
        dist_list (List[float]): Distance array of top k vectors.
        id_list (List[int]): ID array of top k vectors.
    """
    vector_index = _get_vector_index(ds)
    tensor = ds.tensors.get(tensor_name)
    return vector_index.vector_search(tensor, index_name, query_vector, **kwargs)


def load_vector_index(ds, tensor_name: str, index_name: str, **kwargs):
    """
    Load vector index into memory when index is unloaded.
    Args:
        ds (Dataset): Dataset of the vector index.
        tensor_name: Then name of the vector tensor.
        index_name: Then name of the vector index.
        **kwargs: Extra parameters.
    """
    vector_index = _get_vector_index(ds)
    tensor = ds.tensors.get(tensor_name)
    vector_index.load_vector_index(tensor, index_name, **kwargs)


def unload_vector_index(ds, tensor_name: str, index_name: str):
    """
    Unload vector index from memory when index is loaded.
    Args:
        ds (Dataset): Dataset of the vector index.
        tensor_name: Then name of the vector tensor.
        index_name: Then name of the vector index.
    """
    vector_index = _get_vector_index(ds)
    tensor = ds.tensors.get(tensor_name)
    vector_index.unload_vector_index(tensor, index_name)


def drop_vector_index(ds, tensor_name: str, index_name: str):
    """
    Drop vector index permanently.
    Args:
        ds (Dataset): Dataset of the vector index.
        tensor_name: Then name of the vector tensor.
        index_name: Then name of the vector index.
    """
    vector_index = _get_vector_index(ds)
    tensor = ds.tensors.get(tensor_name)
    vector_index.drop_vector_index(tensor, index_name)


def _get_vector_index(ds):
    try:
        from muller.core.vector.vector_index import TensorVectorIndex
    except Exception as e:
        raise ModuleNotFoundError("please install dependency for vector search first.") from e
    if ds.vector_index is None:
        base_path = ds.path
        branch_name = ds.branch
        commit_id = ds.pending_commit_id
        try:
            ds.vector_index = TensorVectorIndex(parent_path=base_path, branch_name=branch_name)
        except Exception as e:
            logger.warning(f"the dataset has not been indexed for branch {branch_name} and commit id {commit_id}")
            raise Exception from e
    return ds.vector_index
