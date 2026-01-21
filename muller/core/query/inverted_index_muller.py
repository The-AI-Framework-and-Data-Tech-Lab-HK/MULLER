# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import json
import os
import pickle
import re
import uuid
from collections import defaultdict
from multiprocessing import Pool

import jieba
import numpy as np
from pathos.pools import ProcessPool

from muller.constants import MAX_WORKERS_FOR_INVERTED_INDEX_SEARCH
from muller.util.exceptions import InvertedIndexNotExistsError

STOP_WORDS = frozenset(('a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can',
                        'for', 'from', 'have', 'if', 'in', 'is', 'it', 'may',
                        'not', 'of', 'on', 'or', 'tbd', 'that', 'the', 'this',
                        'to', 'us', 'we', 'when', 'will', 'with', 'yet',
                        'you', 'your', "，", "；", "？",))


class InvertedIndex(object):
    def __init__(
        self, storage,
        column_name: str,
        branch: str,
        use_uuid: bool = False,
        optimize: bool = False
    ):
        self.inverted_index = defaultdict(set)
        self.column_name = column_name
        self.index_folder = os.path.join("inverted_index_dir", branch, column_name)
        self.use_uuid = use_uuid
        self.storage = storage
        self.file_list_path = os.path.join(self.index_folder, "file_list.json")
        self.optimize = optimize
        self.file_dict = self._get_file_dict()

        # Whether we need index optimization? If it is generic type, then it can be optimized.
        if self.optimize:
            self.optimize = True
        else:
            self.optimize = False

    @staticmethod
    def _naive_tokenize(text):
        pattern = re.compile(r'([a-zA-Z]+)|([\u4e00-\u9fff])|([。，！？])')
        doc_tmp = pattern.sub(r' \1\2\3 ', text)
        words = re.sub(r'\s+', ' ', doc_tmp).split()
        return words

    @staticmethod
    def _jieba_tokenize(text):
        words = jieba.lcut(text, cut_all=True)
        return words

    @staticmethod
    def _divide_into_batches(all_keys: list, all_docs: list, batch_size: int):
        uuid_groups, doc_groups = [], []
        for i in range(0, len(all_keys), batch_size):
            uuid_groups.append(all_keys[i:i+batch_size])
            doc_groups.append(all_docs[i:i+batch_size])
        return uuid_groups, doc_groups

    @staticmethod
    def _merge_dict(source_dict, target_dict):
        for source_key, source_value in source_dict.items():
            target_dict[source_key].update(source_value)
        return target_dict

    def create_index(self, doc_dicts, batch_size: int, save_to_next_storage=True):
        """Create inverted index.
        """
        # Need to clear the inde folder if it is not empty
        if self.column_name in self.file_dict:
            for file_name in self.file_dict[self.column_name]:
                del self.storage[os.path.join(self.index_folder, file_name)]
            self.file_dict = {}


        # Create index via multi-processing
        # Note: Since the memory of processes is not shared, so each process work independently
        # with a piece of self.inverted_index returned as batch
        results = []
        # Divide into batches according to batch_size
        uuids, docs = self._divide_into_batches(list(doc_dicts.keys()), list(doc_dicts.values()), batch_size)
        pool = Pool(len(uuids))
        for uuid_list, doc_list in zip(uuids, docs):
            results.append(pool.apply_async(func=self._create_index_subtask,
                                            args=(uuid_list, doc_list, save_to_next_storage)))

        pool.close()
        pool.join()  # Note: Wait until all the process in the pool finish the execution
        pool.terminate()

        self.file_dict[self.column_name] = {}
        for result in results:
            res = result.get()
            self.file_dict[self.column_name].update({res: []})

        # Update the filt dict，to record whether uuid is used
        self.file_dict["use_uuid"] = self.use_uuid
        self.storage[self.file_list_path] = json.dumps(self.file_dict).encode('utf-8')
        self.storage.flush()

        # Index optimization
        if self.optimize:
            self.optimize_index()

    def update_index(self, diff, batch_size: int, save_to_next_storage=True):
        """Update inverted index, only considers appending new data.
        """
        results = []
        doc_dicts = diff['added']
        uuids, docs = self._divide_into_batches(list(doc_dicts.keys()), list(doc_dicts.values()), batch_size)
        pool = Pool(len(uuids))
        for uuid_list, doc_list in zip(uuids, docs):
            results.append(pool.apply_async(func=self._create_index_subtask,
                                            args=(uuid_list, doc_list, save_to_next_storage)))
        pool.close()
        pool.join()  # Note: Wait until all the process in the pool finish the execution
        pool.terminate()

        for result in results:
            res = result.get()
            self.file_dict[self.column_name].update({res: []})
        self.storage[self.file_list_path] = json.dumps(self.file_dict).encode('utf-8')
        self.storage.flush()

    def optimize_index(self):
        """
        Load the existing index shard files into a huge file, optimize the index,
        then delete the old index files and rewrite the new ones.
        关键：相似与相近的index放在相近的位置
        """
        # Load the file_dict file
        try:
            file_dict = self.file_dict[self.column_name]
        except KeyError as e:
            raise InvertedIndexNotExistsError(self.column_name) from e

        # Load the index file via multi-threading
        def _load_single_batch(file_name):
            batch = self._load_index(file_name)
            # Delete tthe index file in the folder
            del self.storage[os.path.join(self.index_folder, file_name)]
            return batch

        pool = ProcessPool(len(file_dict))
        results = pool.map(_load_single_batch, file_dict)
        pool.close()
        pool.join()  # Note: Wait until all the process in the pool finish the execution
        pool.clear()

        # Record the kv pairs of all the global index in a super_dict
        # TODO: Can be optimize via multi-threading
        super_dict = defaultdict(set)
        for batch in results:
            for k, v in batch.items():
                super_dict[k] = super_dict[k].union(v)

        # Reorder and dunmp to storage
        all_keys = list(super_dict.keys())
        all_keys.sort(reverse=False)  # 升序！
        step = max(1, int(len(all_keys)/len(file_dict)))
        key_groups = [all_keys[i: i+step] for i in range(0, len(all_keys), step)]

        file_name_dict = {}
        for key_group in key_groups:
            new_dict = dict((key, value) for key, value in super_dict.items() if key in key_group)
            new_file_name = str(uuid.uuid4().hex) + ".json"
            self._save_index(new_file_name, new_dict)
            file_name_dict[new_file_name] = key_group

        # Update file_dict
        self.file_dict[self.column_name] = file_name_dict
        self.storage[self.file_list_path] = json.dumps(self.file_dict).encode('utf-8')
        self.storage.flush()

    def search(self, query, search_type="fuzzy_match"):
        """Search keyword.
        """
        # Tokenization of query
        if search_type == "fuzzy_match":
            query_words = self._jieba_tokenize(query)
        elif search_type == "exact_match":
            query_words = [query]
        else:  # search_type=="range_match"
            query_words = (query[0], query[1])

        def _search_single_batch(file_name):
            batch = self._load_index(file_name)
            _res_doc_ids = set()
            if search_type == "fuzzy_match":
                _res_doc_ids = batch[query_words[0]]
                for word in query_words[1:]:
                    tmp_doc_ids = batch[word]
                    # Merge the search results (use the intersection so that each result is involved)
                    _res_doc_ids = _res_doc_ids & tmp_doc_ids

            elif search_type == "exact_match":
                target_query = query_words[0]
                if target_query in batch.keys():  # It is possible that the matched key is empty
                    _res_doc_ids = batch[target_query]

            else:  # search_type=="range_match"
                batch_keys = np.array(list(batch.keys()))
                # Fetch matched keys
                match_keys = batch_keys[np.logical_and(batch_keys >= query_words[0], batch_keys <= query_words[1])]
                if len(match_keys):  # It is possible that the matched key is empty
                    # Fetch the values of the matched keys
                    _res_doc_ids = batch[match_keys[0]]
                    for key in match_keys[1:]:
                        tmp_doc_ids = batch[key]
                        _res_doc_ids = _res_doc_ids | tmp_doc_ids

            return _res_doc_ids

        try:
            file_dict = self.file_dict[self.column_name]
        except KeyError as e:
            raise InvertedIndexNotExistsError(self.column_name) from e

        file_list = self._optimize_search(search_type, query_words, file_dict)
        if len(file_list) == 0:  # The key to be queried is not existed, return []
            return []

        try:
            self.use_uuid = self.file_dict["use_uuid"]
        except KeyError:
            pass

        num_process = min(max(1, len(file_list)), MAX_WORKERS_FOR_INVERTED_INDEX_SEARCH) # Minimal is 1, Maximum is 50
        pool = ProcessPool(num_process)
        results = pool.map(_search_single_batch, file_list)
        pool.close()
        pool.join()  # Note: Wait until all the process in the pool finish the execution
        pool.clear()

        # Merge the results from each batch
        res_doc_ids = results[0] if len(results) > 0 else []

        for j in range(1, len(results)):
            result = results[j]
            res_doc_ids = res_doc_ids | result
        return res_doc_ids

    def _optimize_search(self, search_type, query_words, file_dict):
        """ If it is optimized index sequence, then we can skip the traversal of some meta files and the loading
        of some index files."""
        if self.optimize:
            file_list = []
            # Note: we need to determine the range of files based on exact match and range match
            if search_type != "range_match":
                target = query_words[0]
                for file_name, indexes in file_dict.items():
                    if file_list and target < indexes[0]:
                        # Since the index is already sorted in ascending order, if the value being searched for is
                        # already smaller than the smallest value in the index, it means there will be no matching
                        # records afterward, and the query can be terminated.
                        break
                    if target in indexes:
                        file_list.append(file_name)

            else:
                for file_name, indexes in file_dict.items():
                    if file_list and query_words[1] < indexes[0]:
                        # Since the index is already sorted in ascending order, if the upper bound of the search range
                        # is already smaller than the smallest value in the index, it means there will be no matching
                        # records afterward, and the query can be terminated.
                        break
                    np_index = np.array(indexes)
                    match_index = np_index[np.logical_and(np_index >= query_words[0], np_index <= query_words[1])]
                    if len(match_index):
                        file_list.append(file_name)

        else:
            file_list = list(file_dict.keys())
        return file_list

    def _get_file_dict(self):
        try:
            file_dict = json.loads(self.storage[self.file_list_path].decode('utf-8'))
        except KeyError:
            file_dict = {}
        return file_dict

    def _save_index(self, file_name: str, data: dict):
        file_path = os.path.join(self.index_folder, file_name)
        self.storage[file_path] = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
        self.storage.flush()

    def _load_index(self, file_name):
        return pickle.loads(self.storage[os.path.join(self.index_folder, file_name)])

    def _create_index_subtask(self, uuid_list: list, doc_list: list, save_to_next_storage: bool):
        inverted_index = defaultdict(set)
        # Note: This is a compute-intensive task, and experiments have shown that Python multithreading is ineffective
        # in this case; therefore, the multithreading approach has been abandoned.
        for single_uuid, single_doc in zip(uuid_list, doc_list):
            if isinstance(single_doc, list):
                single_doc = single_doc[0]
            if isinstance(single_doc, str):
                words = self._jieba_tokenize(single_doc)
                for word in words:
                    if word not in STOP_WORDS:
                        inverted_index[word].add(single_uuid)
            else:
                inverted_index[single_doc].add(single_uuid)
        file_name = str(uuid.uuid4().hex) + ".json"
        if save_to_next_storage:
            # Save the index file
            # (Note: Due to the lack of proper index partitioning, the contents of different files may overlap!)
            self._save_index(file_name, inverted_index)
        return file_name
