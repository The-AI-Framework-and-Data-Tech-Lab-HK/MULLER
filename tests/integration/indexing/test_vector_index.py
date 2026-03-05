# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import gc
import logging
import os
import shutil
import time

import numpy as np
import pytest
from numpy._typing import NDArray

import muller
from muller.core.vector.exceptions import SearchError
from tests.constants import SMALL_TEST_PATH
from tests.utils import official_path, check_skip_vector_index_test


@pytest.mark.skipif(
    check_skip_vector_index_test(), reason="It should be skipped if not in local"
)
class TestCpuVectorIndex:
    ds = None

    def setup_method(self, storage):
        logging.info("Create dataset.")
        if os.path.exists(SMALL_TEST_PATH):
            shutil.rmtree(SMALL_TEST_PATH)
        self.ds = muller.dataset(
            path=official_path(storage, SMALL_TEST_PATH), reset=True
        )

    def teardown_method(self):
        logging.info("Drop dataset.")
        self.ds.delete(large_ok=True)

    @pytest.mark.parametrize(
        "index_name,index_type,query_vector",
        [
            ["flat", "FLAT", np.random.rand(1, 3).astype(np.float32)],
            ["hnsw", "HNSWFLAT", np.random.rand(1, 3).astype(np.float32)],
            ["disk_ann", "DISKANN", np.random.rand(3).astype(np.float32)],
        ],
    )
    def test_simple(
        self, index_name: str, index_type: str, query_vector: NDArray[np.float32]
    ):
        logging.info(f"Test Case: simple test {index_name}")
        with self.ds:
            self._add_simple_data()
            vectors = self.ds.vectors
            logging.info(f"dataset vector: {vectors.numpy()}")
            logging.info(f"query vector:{query_vector}")
            self.ds.create_vector_index(
                tensor_name="vectors",
                index_name=index_name,
                index_type=index_type,
                metric="l2",
            )
            dist_list, id_list = self.ds.vector_search(
                query_vector=query_vector,
                tensor_name="vectors",
                index_name=index_name,
                topk=1,
            )
            logging.info(f"id_list: {id_list}")
            logging.info(f"dist_list{dist_list}")
            self.ds.drop_vector_index(tensor_name="vectors", index_name=index_name)

    @pytest.mark.parametrize(
        "index_name,index_type,query_vector",
        [
            ["flat", "FLAT", np.random.rand(1, 3).astype(np.float32)],
            ["hnsw", "HNSWFLAT", np.random.rand(1, 3).astype(np.float32)],
            ["disk_ann", "DISKANN", np.random.rand(3).astype(np.float32)],
        ],
    )
    def test_unload(
        self, index_name: str, index_type: str, query_vector: NDArray[np.float32]
    ):
        logging.info(f"Test Case: simple test {index_name}")
        with self.ds:
            self._add_simple_data()
            vectors = self.ds.vectors
            logging.info(f"dataset vector: {vectors.numpy()}")
            self.ds.create_vector_index(
                tensor_name="vectors",
                index_name=index_name,
                index_type=index_type,
                metric="l2",
            )
            logging.info(f"query vector:{query_vector}")
            self.ds.unload_vector_index(tensor_name="vectors", index_name=index_name)
            with pytest.raises(SearchError):
                self.ds.vector_search(
                    query_vector=query_vector,
                    tensor_name="vectors",
                    index_name=index_name,
                    topk=1,
                )
            self.ds.load_vector_index(tensor_name="vectors", index_name=index_name)
            dist_list, id_list = self.ds.vector_search(
                query_vector=query_vector,
                tensor_name="vectors",
                index_name=index_name,
                topk=1,
            )
            logging.info(f"id_list: {id_list}")
            logging.info(f"dist_list{dist_list}")
            self.ds.drop_vector_index(tensor_name="vectors", index_name=index_name)

    def test_ivfpq_gist(self):
        with self.ds:
            self._add_gist_data()
            self.ds.create_vector_index(
                tensor_name="gist",
                index_name="ivfpq",
                index_type="IVFPQ",
                metric="l2",
                nlist=1000,
                m=96,
            )
            q = np.random.rand(1000, 960)
            start = time.time()
            res = self.ds.vector_search(
                query_vector=q,
                tensor_name="gist",
                index_name="ivfpq",
                topk=10,
                nprobe=10,
                refine_factor=2,
            )
            end = time.time()
            logging.info(f"cost time: {end - start}s result: {res}")

    def test_incremental_vector_index(self):
        q = np.zeros(shape=(1, 3))
        logging.info(f"query vector:{q}")
        with self.ds:
            if "vectors" not in self.ds.tensors:
                self.ds.create_tensor(
                    name="vectors", htype="vector", dtype="float32", dimension=3
                )
            vectors = self.ds.vectors
            logging.info(vectors.key)

            vectors.append(np.array([0.2, 0.2, 0.2]))
            vectors.append(np.array([0.3, 0.3, 0.3]))
            logging.info(f"dataset vector:\n{vectors.numpy()}")
            self.ds.commit()
            self.ds.create_vector_index(
                tensor_name="vectors", index_name="flat", index_type="FLAT", metric="l2"
            )
            dist_list, id_list = self.ds.vector_search(
                query_vector=q, tensor_name="vectors", index_name="flat", topk=1
            )
            logging.info(id_list)
            logging.info(dist_list)

            self.ds.unload_vector_index(tensor_name="vectors", index_name="flat")
            self.ds.load_vector_index(
                tensor_name="vectors", index_name="flat", device="cpu"
            )

            vectors.append(np.array([0.0, 0.0, 0.0]))
            vectors.append(np.array([0.1, 0.1, 0.1]))
            logging.info(f"dataset vector:\n{vectors.numpy()}")
            self.ds.commit()
            self.ds.update_vector_index(tensor_name="vectors", index_name="flat")
            dist_list, id_list = self.ds.vector_search(
                query_vector=q, tensor_name="vectors", index_name="flat", topk=1
            )

            logging.info(id_list)
            logging.info(dist_list)
            self.ds.drop_vector_index(tensor_name="vectors", index_name="flat")

    @pytest.mark.parametrize(
        "index_name,index_type,query_vector,shard_num",
        [
            ["flat", "FLAT", np.random.rand(1, 3).astype(np.float32), 10],
        ],
    )
    def test_shard(self, index_name, index_type, query_vector, shard_num):
        logging.info(f"Test Case: simple test {index_name}")
        with self.ds:
            if "vectors" in self.ds.tensors.keys():
                self.ds.delete_tensor(name="vectors", large_ok=True)
            self.ds.create_tensor(
                name="vectors", htype="vector", dtype="float32", dimension=3
            )
            vectors = self.ds.vectors
            data = np.random.rand(30000).reshape(10000, 3).astype(np.float32)
            vectors.extend(data)
            self.ds.commit()
            vectors = self.ds.vectors
            logging.info(f"dataset vector: {vectors.numpy()}")
            logging.info(f"query vector:{query_vector}")
            self.ds.create_vector_index(
                tensor_name="vectors",
                index_name=index_name,
                index_type=index_type,
                metric="l2",
            )
            dist_list, id_list = self.ds.vector_search(
                query_vector=query_vector,
                tensor_name="vectors",
                index_name=index_name,
                topk=1,
            )
            logging.info(f"id_list: {id_list}")
            logging.info(f"dist_list{dist_list}")
            self.ds.drop_vector_index(tensor_name="vectors", index_name=index_name)

    def _add_gist_data(self):
        logging.info("Loading GIST dataset.")
        if "gist" not in self.ds.tensors.keys():
            self.ds.create_tensor(
                name="gist", htype="vector", dtype="float32", dimension=960
            )
            gist = self.ds.gist
            gist_data = np.random.rand(1000000, 960).astype(np.float32)
            gist.extend(gist_data)
            del gist_data
            gc.collect()
            self.ds.commit()

    def _add_simple_data(self):
        if "vectors" in self.ds.tensors.keys():
            self.ds.delete_tensor(name="vectors", large_ok=True)
        self.ds.create_tensor(
            name="vectors", htype="vector", dtype="float32", dimension=3
        )
        vectors = self.ds.vectors
        vectors.append(np.array([0.0, 0.0, 0.0]))
        vectors.append(np.array([0.1, 0.1, 0.1]))
        vectors.append(np.array([0.2, 0.2, 0.2]))
        vectors.append(np.array([0.3, 0.3, 0.3]))
        self.ds.commit()


if __name__ == "__main__":
    pytest.main(["-s", "test_vector_index.py"])
