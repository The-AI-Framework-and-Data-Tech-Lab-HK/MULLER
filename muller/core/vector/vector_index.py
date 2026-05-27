# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Union, List, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from muller.core.storage import StorageProvider
from muller.core.tensor import Tensor
from .artifact_store import IndexArtifactStore
from . import utils
from .exceptions import (
    IndexNotFoundError,
    IndexMetaError,
    GetIndexError,
    IndexExistsError,
    SearchError,
    IndexAddDataError,
    IndexNotLoadError,
    CreateIndexError,
)

if TYPE_CHECKING:
    import torch
else:
    class _TorchModule:
        Tensor = object

    torch = _TorchModule()


class VectorIndex:
    """
    The Adaptor of different vector index lib
    """

    def __init__(
        self,
        artifact_store: IndexArtifactStore,
        tensor_name: str,
        index_name: str,
        device: str = "cpu",
    ):
        self._index = None
        self.index_name: str = index_name
        self.tensor_name: str = tensor_name
        self.artifact_store = artifact_store
        self._device: str = device
        self._meta: Dict = {}
        self._local_dir: tempfile.TemporaryDirectory[str] | None = None

        # preload index meta
        if self.artifact_store.exists(self.meta_key):
            self._meta = self._load_meta()

    @property
    def index(self):
        """
        Return the index object.
        Returns:
            The index object.
        """
        if self._index is None:
            raise GetIndexError(
                "Get None index, cause index is not created or not load."
            )
        return self._index

    @property
    def index_prefix(self):
        """
        The storage prefix for this vector index.
        Returns:
            A storage key prefix.
        """
        return f"{self.tensor_name}/{self.index_name}"

    @property
    def meta_key(self):
        """
        The storage key for metadata.
        Returns:
            A storage key.
        """
        return f"{self.index_prefix}/meta.json"

    @property
    def artifact_prefix(self):
        """
        The storage prefix for the current index artifact files.
        Returns:
            A storage key prefix.
        """
        return self._meta.get("artifact_prefix", f"{self.index_prefix}/artifacts")

    @property
    def is_exist(self):
        """
        Whether the index is existed.
        Returns:
            True if the index exist, otherwise False.
        """
        return self.artifact_store.exists(self.meta_key)

    @property
    def is_loaded(self):
        """
        Whether the index is loaded in memory.
        Returns:
            True if the index is loaded in memory, otherwise False.
        """
        return self._index is not None

    @property
    def metric(self):
        """
        The metric to measure the distance between vectors.
        Returns:
            A string represent the distance metric.
        """
        try:
            return self._meta["metric"]
        except KeyError as err:
            logging.error(f"Not found 'metric' in self._meta.")
            raise IndexMetaError(
                f"not found 'metric' in self._meta: {self._meta}"
            ) from err

    @property
    def index_type(self):
        """
        The index type of the vector index.
        Returns:
            A string represent the index type.
        """
        try:
            return self._meta["index_type"]
        except KeyError as err:
            logging.error(f"Not found 'index_type' in self._meta.")
            raise IndexMetaError(
                f"not found 'index_type' in self._meta: {self._meta}"
            ) from err

    @property
    def commit_id(self):
        """
        The commit id of the vector index.
        Returns:
            A string represent the commit id.
        """
        try:
            return self._meta["commit_id"]
        except KeyError as err:
            logging.error(f"Not found 'commit_id' in self._meta.")
            raise IndexMetaError(
                f"not found 'commit_id' in self._meta: {self._meta}"
            ) from err

    @commit_id.setter
    def commit_id(self, commit_id: str):
        self._meta["commit_id"] = commit_id
        self._save_meta(self._meta, overwrite=True)

    @index_type.setter
    def index_type(self, index_type: str):
        self._meta["index_type"] = index_type

    def load(self, **kwargs):
        """
        Load the index object into memory.
        Args:
            **kwargs: parameters to load index.
        """
        if self.is_exist:
            self._meta = self._load_meta()
            self._index = self._load_index(**kwargs)
        else:
            raise IndexNotFoundError(self.tensor_name, self.index_name)

    def unload(self):
        """
        Unload the index object in memory.
        Args:
            **kwargs: parameters to unload index.
        """
        self._index = None
        self._meta = None

    def build_index(
        self,
        vector_array: Union[NDArray, torch.Tensor],
        id_array: Union[NDArray, torch.Tensor],
        index_type: str = "FLAT",
        metric: str = "l2",
        **param,
    ):
        """
        Build the vector index.
        Args:
            vector_array: 1-d ndarray to build vector index.
            id_array: 1-d ndarray of the vectors.
            index_type: the index type of vector index.
            metric: the metric type to measure the distance between vectors.
            **param: the create index parameters for specific vector index type.
        """
        logging.info("Start building vector index...")
        self.artifact_store.storage.check_readonly()
        if len(vector_array.shape) > 2:
            raise CreateIndexError(
                f"unexpect vector_array format with shape {vector_array.shape}"
            )
        if index_type not in utils.index_pkg:
            raise CreateIndexError(f"unexpect index_type value with {index_type}")

        commit_id = param.get("commit_id")
        if commit_id is None:
            raise CreateIndexError("commit_id cannot be None.")
        vector_array = np.ascontiguousarray(vector_array, dtype=np.float32)
        id_array = np.ascontiguousarray(id_array, dtype=np.int64)
        dimension = vector_array.shape[1]
        index_creator = utils.load_algo(index_type)
        if index_type == "DISKANN":
            self._reset_local_dir()
            param.update({"path": str(self.local_path)})
        old_artifact_prefix = self._meta.get("artifact_prefix")
        self._index, parameter = index_creator.create(
            vector_array=vector_array,
            id_array=id_array,
            dimension=dimension,
            metric=metric,
            **param,
        )
        parameter = dict(parameter)
        parameter.pop("path", None)
        self._meta = {
            "index_name": self.index_name,
            "index_type": index_type,
            "dimension": dimension,
            "metric": metric,
            "parameter": parameter,
            "commit_id": commit_id,
        }
        self._save_index_and_meta(self._index, self._meta, old_artifact_prefix)
        logging.info("Finish building vector index.")

    def search(
        self, query_array: Union[NDArray, torch.Tensor], **search_param
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint64]]:
        """
        Doing KNN search.
        Args:
            query_array: 2-d ndarray to do KNN search.
            **search_param: the search parameters for specific vector index.

        Returns:
            (id_array, dist_array), A tuple of the result id array and distance array.
        """
        if self.index_type is not None:
            index_algo = utils.load_algo(self.index_type)
            refine_factor = search_param.get("refine_factor", 1)
            topk = search_param.get("topk", 1)
            search_param["topk"] = int(topk * refine_factor)
            dist_list, id_list = index_algo.search(
                self._index, query_array, self._meta["metric"], **search_param
            )
            return dist_list, id_list
        raise IndexMetaError(
            "index meta['index_type'] is None, maybe you should load index or create index first."
        )

    def drop(self):
        """
        Drop the vector index in disk.
        """
        self.artifact_store.storage.check_readonly()
        self.artifact_store.delete_prefix(self.index_prefix)
        self.artifact_store.storage.flush()

    def add_data(self, input_array: NDArray[np.float32], id_array: NDArray[np.int64]):
        """
        Add data into the vector index.
        Args:
            input_array: 1-d ndarray of vector to append into index.
            id_array: 1-d ndarray of ids for the appended vectors.
        """
        if len(input_array.shape) > 2:
            raise CreateIndexError(
                f"unexpect input_array format with shape {input_array.shape}"
            )

        input_array = np.ascontiguousarray(input_array, dtype=np.float32)
        id_array = np.ascontiguousarray(id_array, dtype=np.int64)

        if input_array.shape[1] == self._meta["dimension"]:
            self._index.add_with_ids(input_array, id_array)
        else:
            raise IndexAddDataError(
                f"Add data to index error, cause input array with {input_array.shape[1]}, "
                f"but index with {self._meta['dimension']}"
            )

    @property
    def local_path(self) -> Path:
        if self._local_dir is None:
            self._reset_local_dir()
        return Path(self._local_dir.name)

    def _reset_local_dir(self):
        if self._local_dir is not None:
            self._local_dir.cleanup()
        self._local_dir = tempfile.TemporaryDirectory(
            prefix=f"muller-vector-index-{self.tensor_name}-{self.index_name}-"
        )

    def _load_meta(self):
        try:
            return self.artifact_store.load_json(self.meta_key)
        except KeyError as err:
            raise IndexError("index meta file not found.") from err

    def _load_index(self, **kwargs):
        index = utils.load_algo(self.index_type)
        if hasattr(index, "loads"):
            artifact_name = self._meta.get("artifact", f"{self.index_name}.index")
            data = self.artifact_store.read_bytes(self.artifact_prefix, artifact_name)
            kwargs.setdefault("device", self._device)
            return index.loads(data, **kwargs)

        self._reset_local_dir()
        self.artifact_store.materialize_dir(self.local_path, self.artifact_prefix)
        kwargs.update(
            {
                "path": self.local_path,
                "index_name": self.index_name,
            }
        )
        return index.load(**kwargs)

    def _save_meta(self, vector_index_meta: Dict, overwrite: bool = True):
        self.artifact_store.save_json(vector_index_meta, self.meta_key)
        self.artifact_store.storage.flush()

    def _save_index_and_meta(
        self,
        vector_index,
        vector_index_meta: Dict,
        old_artifact_prefix: str | None = None,
    ):
        index_algo = utils.load_algo(self.index_type)
        if old_artifact_prefix is None:
            old_artifact_prefix = self._meta.get("artifact_prefix")
        new_artifact_prefix = f"{self.index_prefix}/artifacts-{uuid.uuid4().hex}"
        manifest = []

        if hasattr(index_algo, "dumps"):
            artifact_name = f"{self.index_name}.index"
            data = index_algo.dumps(vector_index)
            self.artifact_store.write_bytes(data, new_artifact_prefix, artifact_name)
            manifest.append({
                "path": artifact_name,
                "key": self.artifact_store.key(new_artifact_prefix, artifact_name),
                "size": len(data),
            })
            vector_index_meta["artifact"] = artifact_name
        else:
            manifest = self.artifact_store.publish_dir(self.local_path, new_artifact_prefix)

        vector_index_meta["artifact_prefix"] = new_artifact_prefix
        vector_index_meta["manifest"] = manifest
        self._save_meta(vector_index_meta)

        if old_artifact_prefix and old_artifact_prefix != new_artifact_prefix:
            self.artifact_store.delete_prefix(old_artifact_prefix)
            self.artifact_store.storage.flush()


class TensorVectorIndex:
    """
    TensorVectorIndex manages the index_map from tensor to indexes and work as an adaptor of VectorIndex API and Dataset
    API.
    """

    def __init__(self, storage: StorageProvider, branch_name: str):
        self.storage = storage
        self.artifact_store = IndexArtifactStore(storage, f"_vector_index/{branch_name}")
        self.branch_name = branch_name

        self._index_map: Dict[str, Dict[str, VectorIndex]] = {}
        # init indexed tensor, but not actually load index file
        self._init_tensor_index()

    @property
    def indexed_tensors(self) -> List[str]:
        """
        Find the names of tensors which has been created vector index.
        Returns:
            List of tensor names which has been created vector index.
        """
        return sorted(self._index_map.keys())

    @staticmethod
    def _uuid_to_id(
        uuid_list: NDArray[np.uint64], tensor: Tensor
    ) -> NDArray[np.uint64]:
        tensor_uuid_list = tensor._sample_id_tensor.numpy().flatten()
        uuid_to_pos = {str(uuid): pos for pos, uuid in enumerate(tensor_uuid_list)}
        return np.array([uuid_to_pos[str(uuid)] for uuid in uuid_list], dtype=np.int64)

    @staticmethod
    def _refine_result(
        tensor: Tensor,
        query_vectors: NDArray[np.float32],
        id_list: NDArray[np.uint64],
        topk: int,
        metric_type: str,
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint64]]:
        topk_id_list = []
        topk_dist_list = []
        # calculate real top-k nearest distance list for each query
        for i in range(query_vectors.shape[0]):
            # get origin vectors from id list
            candidate_ids = id_list[i]
            vectors: NDArray[np.uint32, np.float32] = tensor[candidate_ids.tolist()].numpy()
            # calculate the distance list between query_vector and origin vectors
            dist_list: NDArray[np.float32] = utils.cal_distance(
                metric_type, query_vectors[i], vectors
            )
            # select the top-k nearest subscript.
            if len(dist_list) <= topk:
                nn_dist_list = np.argsort(dist_list)
            else:
                nn_dist_list = np.argpartition(a=dist_list, kth=topk - 1)[:topk]
                nn_dist_list = nn_dist_list[np.argsort(dist_list[nn_dist_list])]
            # map the top-k nearest subscript to original id
            topk_ids = candidate_ids[nn_dist_list]
            topk_id_list.append(topk_ids)
            topk_dist_list.append(dist_list[nn_dist_list])

        return np.array(topk_dist_list), np.array(topk_id_list)

    def create_vector_index(
        self,
        tensor: Tensor,
        index_name: str,
        index_type: str,
        metric: str,
        **create_param,
    ):
        """
        Create vector index by tensor and index name.
        Args:
            tensor: The tensor to create index.
            index_name: The name of the vector index.
            index_type: The type of vector index.
            metric: The metric that measure the distance between vectors.
            **create_param: Extra parameters.

        Returns:

        """
        overwrite = create_param.get("overwrite", False)
        vector_index = VectorIndex(self.artifact_store, tensor.key, index_name)
        if overwrite or not vector_index.is_exist:
            vector_array = tensor.numpy()
            vector_index.build_index(
                vector_array=vector_array,
                id_array=np.arange(vector_array.shape[0], dtype=np.int64),
                index_type=index_type,
                metric=metric,
                **create_param,
            )
            self._cache_vector_index(tensor, index_name, vector_index)
        else:
            raise IndexExistsError(tensor_name=tensor.key, index_name=index_name)

    def get_vector_index(self, tensor: Tensor, index_name: str) -> VectorIndex:
        """
        Return the VectorIndex object by tensor and index name.
        Args:
            tensor: The vector tensor.
            index_name: The name of the index.

        Returns:

        """
        if self._index_exists(tensor, index_name):
            tensor_indexes = self._index_map.get(tensor.key, {})
            if index_name in tensor_indexes:
                return tensor_indexes.get(index_name)
        raise IndexNotFoundError(tensor_name=tensor.key, index_name=index_name)

    def load_vector_index(self, tensor: Tensor, index_name: str, **kwargs):
        """
        Load a vector index by muller.core.Tensor and index name.
        Args:
            tensor: The vector tensor.
            index_name: The index name of the vector index.
            **kwargs: Extra parameters.
        """
        vector_index = self.get_vector_index(tensor, index_name)
        if not vector_index.is_loaded:
            vector_index.load(**kwargs)

    def unload_vector_index(self, tensor: Tensor, index_name: str):
        """
        Unload a vector index by muller.core.Tensor and index name.
        Args:
            tensor: The vector tensor.
            index_name: The index name.
        """
        vector_index = self.get_vector_index(tensor, index_name)
        if vector_index.is_loaded:
            vector_index.unload()

    def vector_search(
        self,
        tensor: Tensor,
        index_name: str,
        query_vector: Union[NDArray, torch.Tensor],
        **search_param,
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint64]]:
        """
        Doing vector search on the tensor by query vector array.
        Args:
            tensor: The vector tensor to search.
            index_name: The index name of the index created in the tensor.
            query_vector: A 2-d ndarray, representing a bunch of query vectors.
            **search_param: Extra parameters.

        Returns:
            A Tuple of id list and distance list.
        """
        vector_index = self.get_vector_index(tensor, index_name)
        if vector_index.is_loaded:
            dist_list, id_list = vector_index.search(query_vector, **search_param)
            if search_param.get("refine_factor", 1) <= 1:
                return dist_list, id_list
            return self._refine_result(
                tensor=tensor,
                query_vectors=query_vector,
                id_list=id_list,
                topk=search_param.get("topk", 1),
                metric_type=vector_index.metric,
            )
        raise SearchError(f"index {index_name} is not load.")

    def drop_vector_index(self, tensor: Tensor, index_name: str):
        """
        Drop the vector index by tensor and index name.
        Args:
            tensor: The vector tensor.
            index_name: The name of the index.
        """
        vector_index = self.get_vector_index(tensor, index_name)
        vector_index.unload()
        vector_index.drop()
        tensor_indexes = self._index_map.get(tensor.key, {})
        tensor_indexes.pop(index_name, None)
        # if drop the last index of tensor, remove tensor level directory
        if len(tensor_indexes) == 0:
            self._index_map.pop(tensor.key, None)

    def update_index(
        self,
        tensor_changes: Dict[str, object],
        tensor: Tensor,
        index_name: str,
        new_commit_id: str,
    ):
        """
        Update index when dataset add a new commit.
            Tips: now only support update index for added data, because of the id issue.
        Args:
            tensor_changes (Dict[str, object]):
            tensor:
            index_name:
            new_commit_id:
        """
        vector_index = self.get_vector_index(tensor, index_name)
        if not vector_index.is_loaded:
            raise IndexNotLoadError(tensor_name=tensor.key, index_name=index_name)
        added_items = list(tensor_changes["added"].items())
        if not added_items:
            vector_index.commit_id = new_commit_id
            return
        data_array: NDArray[np.float32] = np.array(
            [value for _, value in added_items], dtype=np.float32
        )
        current_uuids = tensor._sample_id_tensor.numpy().flatten()
        uuid_to_pos = {str(uuid): pos for pos, uuid in enumerate(current_uuids)}
        id_array = np.array([uuid_to_pos[str(uuid)] for uuid, _ in added_items], dtype=np.int64)
        vector_index.add_data(data_array, id_array)
        vector_index._meta["commit_id"] = new_commit_id
        vector_index._save_index_and_meta(vector_index.index, vector_index._meta)

    def _init_tensor_index(self):
        root = self.artifact_store.root.rstrip("/")
        prefix = f"{root}/" if root else ""
        for key in self.artifact_store.list_prefix():
            if not key.endswith("/meta.json"):
                continue
            relative_key = key[len(prefix):] if prefix and key.startswith(prefix) else key
            parts = relative_key.split("/")
            if len(parts) != 3:
                continue
            tensor_name, index_name, meta_file = parts
            if meta_file != "meta.json":
                continue
            self._index_map.setdefault(tensor_name, {})
            self._index_map[tensor_name][index_name] = VectorIndex(
                self.artifact_store,
                tensor_name,
                index_name,
            )

    def _cache_vector_index(
        self, tensor: Tensor, index_name: str, vector_index: VectorIndex
    ):
        indexes = self._index_map.get(tensor.key)
        if indexes is None:
            self._index_map.update({tensor.key: {index_name: vector_index}})
        else:
            indexes.update({index_name: vector_index})

    def _index_exists(self, tensor: Tensor, index_name: str):
        return tensor.key in self._index_map and index_name in self._index_map.get(
            tensor.key
        )
