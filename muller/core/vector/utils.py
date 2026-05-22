# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import importlib
import os
import sys

if sys.platform == "darwin":
    # FAISS and PyTorch wheels can load separate OpenMP runtimes on macOS.
    # This must be set before FAISS imports libomp.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import faiss
except ImportError as err:
    raise ModuleNotFoundError("please install dependency for vector search first: faiss-cpu") from err
import numpy as np
from numpy._typing import NDArray

# map from metric string to faiss metric type
metric_map = {
    "l2": faiss.METRIC_L2,
    "cosine": faiss.METRIC_INNER_PRODUCT,
    "inner_product": faiss.METRIC_INNER_PRODUCT,
}

index_pkg = {
    "IVFPQ": {"class": "IVFPQIndex", "pkg": "muller.core.vector.algorithms.faiss_index"},
    "FLAT": {"class": "FlatIndex", "pkg": "muller.core.vector.algorithms.faiss_index"},
    "HNSWFLAT": {
        "class": "HNSWFLATIndex",
        "pkg": "muller.core.vector.algorithms.faiss_index",
    },
    "DISKANN": {
        "class": "StaticDiskIndex",
        "pkg": "muller.core.vector.algorithms.diskann_index",
    },
}


@staticmethod
def _cosine(
    vector: NDArray[np.float32], query: np.ndarray[(1,), np.float32]
) -> NDArray[np.float32]:
    return np.einsum("i,j,j->i", vector, query) / (
        np.linalg.norm(vector, axis=1) * np.linalg.norm(query)
    )


@staticmethod
def _l2(
    vector: NDArray[np.float32], query: np.ndarray[(1,), np.float32]
) -> NDArray[np.float32]:
    return np.linalg.norm(vector - query, axis=1)


@staticmethod
def _inner_product(
    vector: NDArray[np.float32], query: np.ndarray[(1,), np.float32]
) -> NDArray[np.float32]:
    return np.einsum("i,j,j->i", vector, query)


@staticmethod
def cal_distance(
    metric: str,
    query_vector: np.ndarray[(1,), np.float32],
    data_vector: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Calculate the distance between data vectors and query vectors.
    Args:
        metric: The metric to measure the distance between data vectors and query vectors.
        query_vector: 1-d ndarray that represents a query vector.
        data_vector: 2-d ndarray that represents the data vectors.

    Returns:

    """
    _calculator = {
        "l2": _l2,
        "cosine": _cosine,
        "inner_product": _inner_product,
    }
    return _calculator.get(metric)(data_vector, query_vector)


@staticmethod
def load_algo(index_type: str):
    """
    Load vector index algorithms package.
    Args:
        index_type:

    Returns:

    """
    if isinstance(index_pkg.get(index_type), dict):
        pkg_info = index_pkg.get(index_type)
        index_pkg.update(
            {
                index_type: getattr(
                    importlib.import_module(pkg_info.get("pkg")), pkg_info.get("class")
                )
            }
        )
    return index_pkg.get(index_type)
