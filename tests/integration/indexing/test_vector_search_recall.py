# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import logging
import os
import shutil
import time
from typing import List, Dict

import numpy as np
import pytest

import muller
from tests.constants import SMALL_TEST_PATH
from tests.utils import official_path, check_skip_vector_index_test


@pytest.mark.skipif(
    check_skip_vector_index_test(), reason="It should be skipped if not in local"
)
class TestVectorSearchRecall:
    ds = None

    @staticmethod
    def _calculate_recall(ground_truth, search_result):
        total_recall = []
        for sr, gt in zip(search_result, ground_truth):
            total = len(sr)
            for sr_q, gt_q in zip(sr, gt):
                ac = 0
                for _id in sr_q:
                    if _id in gt_q:
                        ac += 1
                total_recall.append(ac / total)
        mean_recall = sum(total_recall) / len(total_recall)
        logging.info(f"recall: {mean_recall}")

    def setup_method(self, storage):
        logging.info("Create dataset.")
        if os.path.exists(SMALL_TEST_PATH):
            shutil.rmtree(SMALL_TEST_PATH)
        self.ds = muller.dataset(
            path=official_path(storage, SMALL_TEST_PATH), reset=True
        )

    @pytest.mark.parametrize(
        "build_index_param,search_param,search_repeat",
        [
            [
                {
                    "nlist": 256,
                    "m": 120,
                },
                {"nprobe": 10, "refine_factor": 4, "topk": 10},
                1,
            ],
            [
                {
                    "nlist": 256,
                    "m": 120,
                },
                {"nprobe": 10, "refine_factor": 1, "topk": 10},
                1,
            ],
            [
                {
                    "nlist": 256,
                    "m": 120,
                },
                {"nprobe": 10, "refine_factor": 1, "topk": 40},
                1,
            ],
        ],
    )
    def test_ivfpq_gist_recall(
        self, build_index_param: Dict, search_param: Dict, search_repeat: int
    ):
        topk = search_param["topk"]
        logging.info(f"Top k: {topk}")

        with self.ds:
            self._add_gist_data()
            self._launch_index(build_index_param=build_index_param)
            ground_truth, search_result = self._do_search(
                repeat=search_repeat, search_param=search_param
            )
            self._calculate_recall(ground_truth, search_result)

    def _add_gist_data(self):
        if "gist" not in self.ds.tensors.keys():
            self.ds.create_tensor(
                name="gist", htype="vector", dtype="float32", dimension=960
            )
            gist = self.ds.gist
            gist_data = np.random.rand(100000, 960)
            gist.extend(gist_data)
            self.ds.commit()
        logging.info("Finish add gist data.")

    def _cal_ground_truth(self, query, topk) -> List[int]:
        _, ground_truth_id_list = self.ds.vector_search(
            query_vector=query, tensor_name="gist", index_name="flat", topk=topk
        )
        return ground_truth_id_list

    def _cal_search_result(self, query, topk, nprobe, refine_factor):
        _, id_list = self.ds.vector_search(
            query_vector=query,
            tensor_name="gist",
            index_name="ivfpq",
            topk=topk,
            nprobe=nprobe,
            refine_factor=refine_factor,
        )
        return id_list

    def _launch_index(self, build_index_param: Dict):
        nlist = build_index_param["nlist"]
        m = build_index_param["m"]
        self.ds.create_vector_index(
            tensor_name="gist", index_name="flat", index_type="FLAT", metric="l2"
        )
        self.ds.create_vector_index(
            tensor_name="gist",
            index_name="ivfpq",
            index_type="IVFPQ",
            metric="l2",
            nlist=nlist,
            m=m,
        )

    def _do_search(self, repeat: int, search_param: Dict):
        ground_truth = []
        search_result = []
        gt_time = 0
        search_time = 0
        topk = search_param["topk"]
        nprobe = search_param["nprobe"]
        refine_factor = search_param["refine_factor"]
        for _ in range(repeat):
            query = np.random.rand(1000, 960)
            t1 = time.time()
            ground_truth.append(self._cal_ground_truth(query, topk))
            t2 = time.time()
            search_result.append(
                self._cal_search_result(query, topk, nprobe, refine_factor)
            )
            t3 = time.time()
            gt_time += t2 - t1
            search_time += t3 - t2
        logging.info(f"Ground truth time: {gt_time / repeat}")
        logging.info(f"Search result time: {search_time / repeat}")
        logging.info(f"(ground truth / search result) time: {gt_time / search_time}")
        return ground_truth, search_result
