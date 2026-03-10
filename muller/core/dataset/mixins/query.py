# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""Query and indexing mixin for Dataset class."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Union

from muller.constants import DATASET_UUID_NAME, FIRST_COMMIT_ID, INVERTED_INDEX_BATCH_SIZE
from muller.core.auth.permission.index_permission_check import index_permission_check
from muller.core.auth.permission.invalid_view_op import invalid_view_op
from muller.core.auth.permission.user_permission_check import user_permission_check
from muller.core.dataset.uuid.shard_hash import divide_to_shard, load_all_shards


class QueryMixin:
    """Mixin providing query and indexing operations for Dataset."""

    def create_index(self, columns, use_uuid: bool = False, batch_size: int = INVERTED_INDEX_BATCH_SIZE):
        """Creates inverted index for target columns."""
        from muller.core.query import create_index
        create_index(self, columns, use_uuid, batch_size)

    def create_index_vectorized(self,
                                tensor_column: str,
                                index_type: str = "fuzzy_match",
                                use_uuid: bool = False,
                                force_create: bool = False,
                                delete_old_index: bool = True,
                                use_cpp: bool = False,
                                **kwargs):
        """Creates inverted index (vectorized) for the target tensor column of the dataset."""
        from muller.core.query import create_index_vectorized
        create_index_vectorized(self, tensor_column, index_type, use_uuid, force_create,
                                                           delete_old_index, use_cpp, **kwargs)

    def optimize_index(self, tensor, use_uuid=None,
                       optimize_mode="create", max_workers=16, delete_old_index=True, use_cpp=False):
        """Optimize inverted index (vectorized) for the target tensor column of the dataset."""
        inverted_index = self.get_inverted_index(tensor, use_uuid, vectorized=True)
        inverted_index.optimize_index(optimize_mode=optimize_mode,
                                      max_workers=max_workers,
                                      delete_old_index=delete_old_index,
                                      use_cpp=use_cpp)

    def reshard_index(self, tensor, old_shard_num: int, new_shard_num: int, max_workers: int = 16, use_uuid=None):
        """Reshard the inverted index (vectorized) for the target tensor column of the dataset."""
        inverted_index = self.get_inverted_index(tensor, use_uuid, vectorized=True)
        inverted_index.reshard_index(old_shard_num=old_shard_num,
                                     new_shard_num=new_shard_num,
                                     max_workers=max_workers)

    def create_hot_shard_index(self, tensor, use_uuid=None, max_workers: int = 16, n=100000):
        """Create a hot shard for the inverted index (vectorized) for the target tensor column of the dataset."""
        inverted_index = self.get_inverted_index(tensor, use_uuid, vectorized=True)
        inverted_index.add_hot_shard(max_workers=max_workers, n=n)

    @index_permission_check
    def get_inverted_index(self, tensor, use_uuid: bool = False, vectorized: bool = False):
        """
        Load inverted index from storage
        注意：如果要使用老版的倒排索引，需要把vectorized改为False
        """
        branch = self.version_state.get("branch", "main")
        optimize = self[tensor].htype == "generic"
        if vectorized:
            from muller.core.query.inverted_index_vectorized import InvertedIndexVectorized
            return InvertedIndexVectorized(self, self.storage, branch, column_name=tensor, use_uuid=use_uuid)

        from muller.core.query.inverted_index_muller import InvertedIndex
        return InvertedIndex(self.storage, tensor, branch, use_uuid, optimize)

    def query(self, tensor_name, query):
        """Query the target tensor column based on inverted index."""
        inverted_index = self.get_inverted_index(tensor_name)
        ids = inverted_index.search(query)
        if inverted_index.use_uuid:
            # map uuids to global idx
            commit_node = self.version_state.get("commit_node")
            commit_id = commit_node.parent.commit_id
            uuids = self.get_tensor_uuids(tensor_name, commit_id)
            index = set()
            for idx, tmp_uuid in enumerate(uuids):
                if str(tmp_uuid) in ids:
                    index.add(idx)
            ids = index
        return ids

    def filter(
            self,
            function: Optional[Union[Callable, str]] = None,
            index_query: Optional[str] = None,
            connector: Optional[str] = "AND",
            offset: Optional[int] = 0,
            limit: Optional[int] = None,
            **kwargs
    ):
        """Filter the dataset with specified function."""
        compute_future = kwargs.get("compute_future", True)

        def function_contents(func):
            try:
                func.__closure__
            except Exception as e:
                func = func.func
                raise Exception from e
            finally:
                closure = tuple(cell.cell_contents for cell in func.__closure__) if func.__closure__ else ()
                tmp_list = [func.__name__, func.__defaults__, func.__kwdefaults__, closure, func.__code__.co_code,
                        func.__code__.co_consts]
                return tmp_list

        function_key = hash(function) if function is None or isinstance(function, str) else hash(
            tuple(function_contents(function)))
        key = str(index_query).strip() + str(function_key) + connector

        if "filter" not in self.storage.upper_cache:
            self.storage.upper_cache["filter"] = {}
        cache_key = key + str(offset)
        if compute_future and cache_key in self.storage.upper_cache["filter"]:
            result = self.storage.upper_cache["filter"].pop(cache_key).result(timeout=None)  # index
            if (len(result.filtered_index) > 0 and
                    result.filtered_index[-1] != len(self) - 1):  # not last index, have next
                self.filter_next(cache_key, function, index_query, connector, offset=result.filtered_index[-1] + 1,
                                 limit=limit)
            return result
        if bool(self.storage.upper_cache["filter"]) and cache_key not in self.storage.upper_cache[
            "filter"]:  # key doesn't match
            self.storage.upper_cache["filter"] = {}  # clear cache

        ids = []
        if index_query is not None:
            from muller.core.query.safe_evaluator import SafeQueryEvaluator
            evaluator = SafeQueryEvaluator(index_query)
            ids_fuzzy_matching = evaluator.evaluate({'query': self.query_string})
            ids = list(ids_fuzzy_matching)
            ids.sort()
        if offset > 0:
            ids = [i for i in ids if i >= offset]
        if function is None:  # fuzzy matching only
            ids = ids[:limit]
            ret = self[ids]
            ret.filtered_index = ids
            return ret

        ds, ids = self._get_filter_res_from_conditions(function, connector, offset, limit, ids, index_query)

        kwargs["index_query"] = index_query
        kwargs["connector"] = connector
        kwargs["ids"] = ids
        kwargs["key"] = key
        from muller.core.query import filter_dataset, query_dataset
        fn = query_dataset if isinstance(function, str) else filter_dataset
        ret = self._process_filter(fn, ds, function, offset, limit, kwargs)
        return ret

    def filter_next(self, key, function, index_query, connector, offset, limit):
        """Filter the dataset with specified function (in advance and save the results to upper cache)"""
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.filter, function=function, index_query=index_query, connector=connector,
                                     offset=offset, limit=limit)
            self.storage.upper_cache["filter"].update({key: future})

    def aggregate(
            self,
            group_by_tensors: List[str],
            selected_tensors: List[str],
            order_by_tensors: Optional[list] = None,
            aggregate_tensors: Optional[list] = None,
            function: Optional[Callable] = None,
            order_direction: str = 'DESC',
            num_workers: int = 0,
            scheduler: str = "processed",
            progressbar: bool = True,
            method: str = "count",
    ):
        """Conduct aggregation query on the dataset."""
        from muller.core.query import aggregate_dataset
        return aggregate_dataset(
            self,
            function,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
            group_by_tensors=group_by_tensors,
            selected_tensors=selected_tensors,
            order_by_tensors=order_by_tensors,
            aggregate_tensors=aggregate_tensors,
            order_direction=order_direction,
            method=method
        )

    def aggregate_vectorized(
            self,
            group_by_tensors: List[str],
            selected_tensors: List[str],
            order_by_tensors: Optional[list] = None,
            aggregate_tensors: Optional[list] = None,
            order_direction: Optional[str] = 'DESC',
            method: str = 'count'
    ):
        """A vectorized aggregate function accelerated by the parallel computing supported by numpy."""
        from muller.core.query import aggregate_vectorized_dataset
        return aggregate_vectorized_dataset(
            dataset=self,
            group_by_tensors=group_by_tensors,
            selected_tensors=selected_tensors,
            order_by_tensors=order_by_tensors,
            aggregate_tensors=aggregate_tensors,
            order_direction=order_direction,
            method=method
        )

    def filter_vectorized(
            self,
            condition_list,
            connector_list: Optional[List] = None,
            offset: Optional[int] = 0,
            limit: Optional[int] = None,
            compute_future: Optional[bool] = True,
            use_local_index: bool = True,
            max_workers: Optional[int] = 16,
            show_progress: bool = False,
    ):
        """A vectorized filtering function accelerated by the parallel computing supported by numpy."""
        from muller.core.query import filter_vectorized_dataset
        return filter_vectorized_dataset(
            self,
            condition_list=condition_list,
            connector_list=connector_list,
            offset=offset,
            limit=limit,
            compute_future=compute_future,
            use_local_index=use_local_index,
            max_workers=max_workers,
            show_progress=show_progress,
        )

    def create_uuid_index(self):
        """Create uuid and index pair and stored in the disk."""
        try:
            current_id = self.version_state['commit_id']
        except KeyError as e:
            raise KeyError from e
        if current_id != FIRST_COMMIT_ID:
            raise ValueError
        uuids = self.get_tensor_uuids(DATASET_UUID_NAME, current_id)
        divide_to_shard(path=os.path.join(self.path, DATASET_UUID_NAME), uuids=uuids)

    def load_uuid_index(self):
        """Load all uuid indexes from shards."""
        try:
            current_id = self.version_state['commit_id']
        except KeyError as e:
            raise KeyError from e
        if current_id != FIRST_COMMIT_ID:
            raise ValueError
        return load_all_shards(path=os.path.join(self.path, DATASET_UUID_NAME))

    @invalid_view_op
    @user_permission_check
    def create_vector_index(self, tensor_name: str, index_name: str, index_type: str = 'FLAT', metric: str = 'l2',
                            **kwargs: Union[int, float, str]):
        """Create index for tensor in vector type."""
        from muller.core.query import create_vector_index
        create_vector_index(self, tensor_name, index_name, index_type, metric, **kwargs)
