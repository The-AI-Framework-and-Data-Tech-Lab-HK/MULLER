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

        # 是否需要优化索引？暂定如果为generic类型的列，则可以优化
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
        # 如果索引文件夹不为空，则需要先清空
        if self.column_name in self.file_dict:
            for file_name in self.file_dict[self.column_name]:
                del self.storage[os.path.join(self.index_folder, file_name)]
            self.file_dict = {}


        # 多进程创建索引
        # 注：因为进程内存不共享，所以这里的进程是各干各的，各自有一份self.inverted_index作为batch返回了
        results = []
        # 按batch_size分批
        uuids, docs = self._divide_into_batches(list(doc_dicts.keys()), list(doc_dicts.values()), batch_size)
        pool = Pool(len(uuids))
        for uuid_list, doc_list in zip(uuids, docs):
            results.append(pool.apply_async(func=self._create_index_subtask,
                                            args=(uuid_list, doc_list, save_to_next_storage)))

        pool.close()
        pool.join()  # 注：进程池中进程执行完毕后再关闭。
        pool.terminate()

        self.file_dict[self.column_name] = {}
        for result in results:
            res = result.get()
            self.file_dict[self.column_name].update({res: []})

        # 更新filt dict，记录是否使用uuid
        self.file_dict["use_uuid"] = self.use_uuid
        self.storage[self.file_list_path] = json.dumps(self.file_dict).encode('utf-8')
        self.storage.flush()

        # 索引优化
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
        pool.join()  # 注：进程池中进程执行完毕后再关闭。
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
        # 先把file_dict文件加载上来
        try:
            file_dict = self.file_dict[self.column_name]
        except KeyError as e:
            raise InvertedIndexNotExistsError(self.column_name) from e

        # 再多线程加载各index文件
        def _load_single_batch(file_name):
            batch = self._load_index(file_name)
            # 删了文件夹中的索引文件
            del self.storage[os.path.join(self.index_folder, file_name)]
            return batch

        pool = ProcessPool(len(file_dict))
        results = pool.map(_load_single_batch, file_dict)
        pool.close()
        pool.join()  # 注：进程池中进程执行完毕后再关闭。
        pool.clear()

        # 用super_dict记录一下全局所有索引的kv对【这里可能可以用多线程优化】
        super_dict = defaultdict(set)
        for batch in results:
            for k, v in batch.items():
                super_dict[k] = super_dict[k].union(v)

        # 重新排序并落盘
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

        # 再次更新file_dict
        self.file_dict[self.column_name] = file_name_dict
        self.storage[self.file_list_path] = json.dumps(self.file_dict).encode('utf-8')
        self.storage.flush()

    def search(self, query, search_type="fuzzy_match"):
        """Search keyword.
        """
        # query的分词
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
                    # 检索结果的合并(取交集，需要每个结果都出现)
                    _res_doc_ids = _res_doc_ids & tmp_doc_ids

            elif search_type == "exact_match":
                target_query = query_words[0]
                if target_query in batch.keys():  # 有可能符合的keys为空
                    _res_doc_ids = batch[target_query]

            else:  # search_type=="range_match"
                batch_keys = np.array(list(batch.keys()))
                # 先取出来符合的key
                match_keys = batch_keys[np.logical_and(batch_keys >= query_words[0], batch_keys <= query_words[1])]
                if len(match_keys):  # 有可能符合的keys为空
                    # 再取出来key们对应的值
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
        if len(file_list) == 0:  # 要搜索的key不存在，直接返回空
            return []

        try:
            self.use_uuid = self.file_dict["use_uuid"]
        except KeyError:
            pass

        num_process = min(max(1, len(file_list)), MAX_WORKERS_FOR_INVERTED_INDEX_SEARCH) # 最小为1， 最大为50
        pool = ProcessPool(num_process)
        results = pool.map(_search_single_batch, file_list)
        pool.close()
        pool.join()  # 注：进程池中进程执行完毕后再关闭。
        pool.clear()

        # 合并每个batch的结果
        res_doc_ids = results[0] if len(results) > 0 else []

        for j in range(1, len(results)):
            result = results[j]
            res_doc_ids = res_doc_ids | result
        return res_doc_ids

    def _optimize_search(self, search_type, query_words, file_dict):
        """如果是优化过的索引序列，则可以跳过部分对meta file的遍历以及对部分索引文件的加载"""
        if self.optimize:
            file_list = []
            # 注意：这里需要根据exact match与range match的来确定文件范围
            if search_type != "range_match":
                target = query_words[0]
                for file_name, indexes in file_dict.items():
                    if file_list and target < indexes[0]:
                        # 因为索引已经从小到大排序，如果要搜索的数已经比该索引内最小的数要小了，说明后面也不会再有记录了，可以终止查询
                        break
                    if target in indexes:
                        file_list.append(file_name)

            else:
                for file_name, indexes in file_dict.items():
                    if file_list and query_words[1] < indexes[0]:
                        # 因为索引已经从小到大排序，如果要搜索的上界已经比该索引内最小的数要小了，说明后面也不会再有记录了，可以终止查询
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
        # 注：这是计算密集型任务，经试验证明python多线程在此没有效果，故已经放弃多线程方案
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
            # 保存索引文件 (注：因为没有很好的索引分区，所以这里不同的文件之间的内容可能有重合！)
            self._save_index(file_name, inverted_index)
        return file_name
