# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import gc
import importlib
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Union, Tuple, TYPE_CHECKING

import numpy as np
from numpy._typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = object

import muller.core.vector.index_param as IndexParam
from muller.core.vector.exceptions import SearchError

dap = None
_diskann_import_error = None
try:
    dap = importlib.import_module("diskannpy")
except ImportError as err:
    _diskann_import_error = err


def _require_diskann():
    if dap is None:
        raise ModuleNotFoundError(
            "please install dependency for DISKANN vector index first: diskannpy"
        ) from _diskann_import_error
    return dap


class DiskANNVectorIndex:

    @classmethod
    @abstractmethod
    def search(
        cls,
        index: "dap.StaticDiskIndex",
        query_vector: Union[NDArray[np.float32], Tensor],
        metric: str,
        **search_param,
    ):
        """
        Abstract search method for DiskANN vector index.
        Args:
            index: the DiskANN index object to call search method.
            query_vector: the 1-d ndarray to do query.
            metric: the metric to measure distance between vectors.
            **search_param: other search parameter for DiskANN.
        """
        pass

    @classmethod
    @abstractmethod
    def create(
        cls,
        vector_array: Union[NDArray[np.float32], Tensor],
        id_array: Union[NDArray[np.float32], Tensor],
        dimension: int,
        metric: str,
        **index_param,
    ):
        """
        Abstract create method for DiskANN vector index.
        Args:
            vector_array: the 1-d ndarray to create vector index
            id_array: the 1-d ndarray of vector id
            dimension: the dimension of each vector
            metric: the metric to measure the distance between vectors.
            **index_param: other create index parameter
        """
        pass

    @classmethod
    def load(cls, **kwargs) -> "dap.StaticDiskIndex":
        """
        Default method for load DiskANN vector index.
        Args:
            **kwargs: the extra load vector index parameters to load index.

        Returns:

        """
        diskann = _require_diskann()
        param = IndexParam.LoadDiskANN.model_validate(kwargs)
        return diskann.StaticDiskIndex(
            index_directory=str(param.path),
            num_threads=param.num_threads,
            num_nodes_to_cache=param.num_nodes_to_cache,
            cache_mechanism=1,
        )

    @classmethod
    def save(cls, **kwargs):
        """
        DiskANN index don't need to save.
        Args:
            **kwargs:
        """
        pass


class StaticDiskIndex(DiskANNVectorIndex):

    @classmethod
    def create(
        cls,
        vector_array: NDArray[np.float32],
        id_array: Union[NDArray[np.float32]],
        dimension: int,
        metric: str,
        **index_param,
    ):
        param = IndexParam.CreateDiskANN.model_validate(index_param)
        diskann = _require_diskann()
        path = Path(param.path)
        if not path.exists():
            path.mkdir(parents=True)
        vector_path = str(path / "ann_vectors.bin")
        dtype = vector_array.dtype
        diskann.vectors_to_file(vector_file=vector_path, vectors=vector_array)
        logging.info("Finish save vectors to file.")
        del vector_array
        gc.collect()
        diskann.build_disk_index(
            data=vector_path,
            vector_dtype=dtype,
            distance_metric=metric,
            index_directory=param.path,
            complexity=param.complexity,
            graph_degree=param.graph_degree,
            search_memory_maximum=param.search_memory_maximum,
            build_memory_maximum=param.build_memory_maximum,
            num_threads=param.num_threads,
            pq_disk_bytes=param.pq_disk_bytes,
        )
        return (
            diskann.StaticDiskIndex(
                index_directory=param.path,
                num_threads=param.num_threads,
                num_nodes_to_cache=param.num_nodes_to_cache,
            ),
            param.__dict__,
        )

    @classmethod
    def search(
        cls,
        index: "dap.StaticDiskIndex",
        query_vector: NDArray[np.float32],
        metric: str,
        **search_param,
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint64]]:
        param = IndexParam.SearchDISKANN.model_validate(search_param)
        if len(query_vector.shape) == 1:
            response = index.search(
                query=query_vector,
                k_neighbors=param.topk,
                complexity=param.complexity,
                beam_width=param.beam_width,
            )
        elif len(query_vector.shape) == 2:
            response = index.batch_search(
                queries=query_vector,
                k_neighbors=param.topk,
                complexity=param.complexity,
                beam_width=param.beam_width,
                num_threads=param.num_threads,
            )
        else:
            raise SearchError(
                f"query_vector with unsupported shape {query_vector.shape}"
            )
        id_list, dist_list = response
        return dist_list, id_list
