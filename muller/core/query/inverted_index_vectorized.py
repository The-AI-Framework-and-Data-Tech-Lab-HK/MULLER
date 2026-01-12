# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import heapq
import json
import logging
import os
import pickle
import re
import shutil
import uuid
import warnings
from collections import defaultdict
import multiprocessing
from typing import Optional, List

import mmh3
import numpy as np

from muller.constants import FIRST_COMMIT_ID, FILTER_LOG
from muller.util.exceptions import InvertedIndexNotExistsError, InvertedIndexUnsupportedError, \
    InvertedIndexNotFoundError, ExecuteError, UnsupportedMethod


class InvertedIndexVectorized(object):
    def __init__(self, dataset, storage, branch, column_name: str, use_uuid=False):
        self.dataset = dataset
        self.storage = storage
        self.branch = branch
        self.column_name = column_name
        self.index_folder = os.path.join("inverted_index_dir_vec", branch, column_name)
        self.use_uuid = use_uuid
        # meta.json：记录了在哪一列的哪个版本完成了倒排索引的建立, 以及详细的元信息【注：与每列的log.json不一样】
        self.meta = os.path.join("inverted_index_dir_vec", branch, "meta.json")
        # col_log_folder: 每列都有的log文件夹，主要记录在此列内已处理过的batch
        self.col_log_folder = "create_index_record"
        # col_log_file: 每列都有的log文件，主要记录在此列内处理的参数
        self.col_log_file = "log.json"
        # logger：记录
        self.logger = self._set_logger(self.dataset.path + os.sep + FILTER_LOG)
        self.hot_shard_data = None

    @property
    def commit_id(self):
        """Function to compute commit id."""
        if self.dataset.commit_id == FIRST_COMMIT_ID:
            return ""
        return self.dataset.version_state['commit_id']

    @staticmethod
    def _set_logger(path):
        # 创建日志记录器
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if not logger.handlers:  # 避免重复添加 handler
            # 创建文件处理器
            file_handler = logging.FileHandler(path, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)

            # 创建控制台处理器
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
            stream_handler.setFormatter(stream_formatter)

            # 添加处理器到日志记录器
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
        return logger

    @staticmethod
    def _jieba_tokenize(text, full_stop_words, compulsory_words,
                        tokenizer="jieba", cut_all=False, case_sensitive=False):
        import jieba

        if not case_sensitive:
            text = text.lower()

        if compulsory_words:
            jieba.load_userdict(compulsory_words)

        words = jieba.lcut(text, cut_all=cut_all)

        final_words = []
        for word in words:
            if word not in full_stop_words:
                final_words.append(word)
        return final_words

    @staticmethod
    def _jieba_tokenize_complex_search(text, full_stop_words, compulsory_words,
                                    tokenizer="jieba", cut_all=False, case_sensitive=False):
        import jieba

        if not case_sensitive:
            text = text.lower()

        if compulsory_words:
            jieba.load_userdict(compulsory_words)

        result = [s for s in re.split(r"(?:\|\|)", text) if s]
        query_tok_dict = {}
        for r in result:
            words = jieba.lcut(r, cut_all=cut_all)
            for word in words:
                if word not in full_stop_words:
                    query_tok_dict.update({r: words})
        return query_tok_dict

    @staticmethod
    def _obtain_stop_words(stop_words_list):
        final_stop_words = set()
        if stop_words_list:
            for file_path in stop_words_list:
                final_stop_words.update([line.strip() for line in open(file_path, 'r').readlines()])
        return final_stop_words

    @staticmethod
    def _byte_to_int64(byte_data, num_of_shard):
        # 哈希为int64（有符号）
        hash_signed = mmh3.hash64(byte_data)[0]
        # 哈希取模计算这个数字的shard
        shard_id = hash_signed % num_of_shard
        return hash_signed, shard_id

    @staticmethod
    def _num_to_shard(num, num_of_shard):
        # 比较简单的方法，直接根据值取模。可能会导致负载均衡问题！
        shard_id = num % num_of_shard
        return int(shard_id)

    @staticmethod
    def _split_data(start, end, num_of_batches, to_remove, cpp_use=False):
        dataset_length = end - start if cpp_use else end - start + 1
        chunk_size = dataset_length // num_of_batches
        remainder = dataset_length % num_of_batches
        # 构造每个子数组的大小
        sizes = np.full(num_of_batches, chunk_size)
        sizes[:remainder] += 1  # 前 remainder 个子数组多一个元素

        # 计算起始索引
        starts = np.cumsum([start] + list(sizes[:-1]))
        ends = starts + sizes - 1

        # shuffle
        shuffled_starts = starts[np.random.permutation(len(starts))]

        # 创建布尔掩码（保留不在 to_remove 中的元素）
        mask = ~np.isin(shuffled_starts, np.array(to_remove))
        # 应用掩码，生成新数组
        filtered_starts = starts[np.random.permutation(len(starts))][mask]
        filtered_ends = ends[np.random.permutation(len(starts))][mask]

        return filtered_starts, filtered_ends

    def create_index(self,
                     index_type: str = "fuzzy_match",
                     num_of_shards: int = 1,
                     uuids=None,
                     max_workers: int = 16,
                     num_of_batches: int = 1,
                     tokenizer: str = "jieba",
                     cut_all: bool = False,
                     stop_words_list: Optional[List[str]] = None,
                     compulsory_words: Optional[str] = None,
                     case_sensitive: bool = False,
                     force_create: bool = False,
                     use_cpp: bool = False,
                     ):
        # 检查是否存在已有的索引。如已有完整索引，则根据force_create的值决定是否需删除再重建。否则将继续创建。
        skip, settings = self._check_existing_indexes(force_create)
        if skip:
            return None
        if settings:
            [num_of_batches, num_of_shards, use_uuids] = settings
            num_of_batches = int(num_of_batches)
            num_of_shards = int(num_of_shards)

        # 定义新的临时索引文件夹，然后把以前的临时文件夹（如有）删了，以免造成混用
        tmp_path = os.path.join(self.dataset.path, self.index_folder + "_tmp")
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)

        # 创建一个log file来记录本次创建的meta信息。注意这只是用于暂时的log记录！
        log_path = os.path.join(self.dataset.path, self.index_folder + "_tmp", self.col_log_folder)
        if not os.path.exists(log_path):
            use_uuids = bool(uuids)
            os.makedirs(log_path)
            tmp_meta = [num_of_batches, num_of_shards, use_uuids]
            with open(os.path.join(log_path, self.col_log_file), "w") as f:
                json.dump(tmp_meta, f)

        # 读取停用词列表并存储到列表内
        if index_type == "fuzzy_match":
            full_stop_words = {"", " ", "  ", '\n', '\t'}
            stop_words = self._obtain_stop_words(stop_words_list)
            if stop_words:
                full_stop_words.update(stop_words)
        else:
            full_stop_words = set()

        # 多进程创建索引
        if use_cpp:
            filtered_starts, filtered_ends = self._split_data(0, len(self.dataset),
                                                              num_of_batches, self._obtain_existing_batches(),
                                                              True)
            num_process = min(len(filtered_starts), max_workers)
            com_words = "" if compulsory_words is None else compulsory_words
            from muller.util.sparsehash.build.custom_hash_map import IndexProcessor
            import muller.util.sparsehash.build.custom_hash_map as cm
            cm.init_logger(os.path.join(self.dataset.path, FILTER_LOG), cm.LogLevel.INFO)
            try:
                IndexProcessor.process_index_parallel(self.dataset.path, self.column_name,
                                                      self.index_folder + "_tmp", self.col_log_folder,
                                                      filtered_starts, filtered_ends,
                                                      num_process, num_of_shards, cut_all,
                                                      case_sensitive, full_stop_words,
                                                      self.commit_id, com_words)
            except Exception as e:
                raise ExecuteError from e
        else:
            tokenizer_params = {
            'tokenizer': tokenizer,
            'cut_all': cut_all,
            'stop_words_list': stop_words_list,
            'compulsory_words': compulsory_words,
            'case_sensitive': case_sensitive
        }
            batch_params = self._setup_batch_params(settings, num_of_batches, num_of_shards, uuids)
            self._create_python_index(
                batch_params, full_stop_words, index_type,
                tokenizer_params['tokenizer'], tokenizer_params['cut_all'],
                tokenizer_params['compulsory_words'], tokenizer_params['case_sensitive'],
                max_workers, uuids
            )

        # 创建完一次索引之后，需要检查是否所有batch的索引都生成完毕，如果有缺少的话，可重新生成！
        unfinished_batches = self.check_index_completeness(self.index_folder + "_tmp", num_of_batches)
        # 日后可考虑采用递归调用unfinished_batches = self.check_create_index_completeness(...) 来保证创建的完整性.
        # 不过需要确保workers意外中断之后可以自动释放内存。

        if not unfinished_batches:
            self.logger.info(f"Creating index of {self.column_name} successfully.")
            return True

        self.logger.info("Creating index fails. There are unfinished batches. "
                         "You may use ds.create_index_vectorized(...) again to finish the creation.")
        return False


    def optimize_index(self,
                       optimize_mode: str = "create",
                       max_workers: int = 16,
                       delete_old_index: bool = False,
                       use_cpp: bool = True):
        """将同一个shard id下的shard文件合并为一个文件"""
        num_of_shards = self._obtain_meta(["num_of_shards"])[0] # 是shard的数量，不是shard文件的数量

        # 如果没有已经生成的临时索引文件文件夹，直接报错返回
        tmp_index_path = os.path.join(self.dataset.path, self.index_folder + "_tmp")
        if not os.path.exists(tmp_index_path):
            raise InvertedIndexNotExistsError(self.column_name)

        # 正式建立索引。注：这里新建了一个（column_name）_optimized索引文件夹，所有索引都写在里面了
        optimized_index_path = os.path.join(self.dataset.path, self.index_folder + f"_optimized")
        if not os.path.exists(optimized_index_path):
            os.makedirs(optimized_index_path)  # 一定要在主进程里新建。如果在子进程里新建的话会打架！
        # merge的时候需要判断一下，这时候是create时的merge呢？还是update时的merge呢？
        # create时的merge：在merge时可以不管已存在的（column_name）索引文件
        # update时的merge：在merge时需要把（column_name）文件夹里的内容也一并合入了
        num_process = min(num_of_shards, max_workers)
        if use_cpp:
            from muller.util.sparsehash.build.custom_hash_map import merge_index_files
            try:
                merge_index_files(tmp_index_path,
                                  optimized_index_path,
                                  current_index_folder=os.path.join(self.dataset.path, self.index_folder),
                                  optimize_mode=optimize_mode,
                                  num_shards=num_of_shards,
                                  num_threads=num_process)
            except Exception as e:
                raise ExecuteError from e
        else:
            pool = multiprocessing.Pool(num_process)
            for i in range(num_of_shards):
                pool.apply_async(func=self._merge_shards,
                                args=(optimize_mode, i,))
            # 注：进程池中进程执行完毕后再关闭。
            pool.close()
            pool.join()

        # 4. 将原索引文件夹（如有）命名为col_[uuid]文件夹。然后将col_optimized文件夹正式命名为col文件夹，并删除col_tmp文件夹
        official_index_path = os.path.join(self.dataset.path, self.index_folder)
        old_index_path = official_index_path + "_" + uuid.uuid4().hex
        if os.path.exists(old_index_path):
            shutil.rmtree(old_index_path)
        if os.path.exists(official_index_path):
            os.rename(official_index_path, old_index_path)
            self.logger.info(f"Rename the old index folder as {old_index_path}")

        os.rename(official_index_path + f"_optimized", official_index_path)
        self.logger.info(f"Generate new index folder of {self.column_name} successfully.")

        if os.path.exists(tmp_index_path):
            shutil.rmtree(tmp_index_path)
            self.logger.info(f"Successfully delete {tmp_index_path} (the unoptimized index).")

        if delete_old_index and os.path.exists(old_index_path):
            shutil.rmtree(old_index_path)
            self.logger.info(f"Successfully delete {old_index_path} (the old index)!")


    def update_index(self,
                     start_index: int,
                     end_index: int,
                     index_type: str = "fuzzy_match",
                     num_of_shards: int = 1,
                     uuids=None,
                     max_workers: int = 16,
                     num_of_batches: int = 1,
                     tokenizer_params: dict = None,
                     use_cpp: bool = True,
                     ):
        """Function to update index based on the original index."""

        default_params = {
            'tokenizer': "jieba",
            'cut_all': False,
            'stop_words_list': [],
            'compulsory_words': None,
            'case_sensitive': False
        }
        if tokenizer_params:
            default_params.update({k: v for k, v in tokenizer_params.items()
                                   if k in default_params and v is not None})
        tokenizer_params = default_params

        # 初始化检查和设置
        settings = self._load_and_validate_settings(use_cpp, index_type)

        # 停用词处理
        full_stop_words = self._get_stop_words(index_type, tokenizer_params['stop_words_list'])

        # 索引更新
        if settings['use_cpp']:
            self._update_with_cpp(
                start_index, end_index, num_of_batches,
                num_of_shards, max_workers, full_stop_words,
                "" if tokenizer_params['compulsory_words'] is None else tokenizer_params['compulsory_words'],
                tokenizer_params['case_sensitive'],
                tokenizer_params['cut_all']
            )
        else:
            self._update_with_python(
                start_index, end_index, num_of_batches,
                num_of_shards, max_workers,
                index_type, tokenizer_params, uuids
            )

        # 完整性检查
        return self._check_update_completion(num_of_batches)


    def check_index_completeness(self, folder: str, num_of_batches: int):
        """Function to check created index's completeness."""
        path = os.path.join(self.dataset.path, folder, self.col_log_folder)
        current_batches = set()
        for file_name in os.listdir(path):
            if file_name.find(self.col_log_file) == -1:
                current_batches.add(int(file_name))
        unfinished_batches = num_of_batches - len(current_batches)
        return unfinished_batches


    def reshard_index(self, old_shard_num: int, new_shard_num: int, max_workers: int = 16):
        """Function to re-shard index."""
        with multiprocessing.Pool(min(old_shard_num, max_workers)) as pool:
            optimize_batch = [pool.apply_async(func=self._reshard_single,
                                               args=(i, old_shard_num, new_shard_num))
                              for i in range(old_shard_num)]
            # 等待上述任务全部完成
            for res in optimize_batch:
                res.wait()

    def add_hot_shard(self, max_workers: int = 16, n: int = 100000):
        """从现有的shard里选出n个出现频率最高的词，写进我们的hot shard！"""
        num_of_shards = self._obtain_meta(["num_of_shards"])[0]
        with multiprocessing.Pool(min(num_of_shards, max_workers)) as pool:
            results = [pool.apply_async(func=self._obtain_hot_data_from_single_shard,
                                               args=(i, ))
                              for i in range(num_of_shards)]
            # 等待上述任务全部完成
            for res in results:
                res.wait()

            results = [res.get() for res in results]

        top_n = heapq.nlargest(n, (num for lst in results for num in lst))

        with multiprocessing.Pool(min(len(top_n), max_workers)) as pool:
            results = [pool.apply_async(func=self._obtain_set_of_key,
                                        args=(num,))
                       for num in top_n]
            # 等待上述任务全部完成
            for res in results:
                res.wait()

            results = [res.get() for res in results]

        final_dict = defaultdict(set)
        for i, num in enumerate(top_n):
            final_dict[num] = results[i]

        # 落盘这个hot shard
        file_name = "hot_shard"
        self._dump_index(self.index_folder, file_name, final_dict)
        self.logger.info(f"dump hot shard")

    def load_hot_shard(self):
        """Function to load hot shard."""
        self.hot_shard_data = self._load_index(os.path.join(self.index_folder, "hot_shard"))


    def search_cpp(self,
                   query: [str, int, bool, float],
                   search_type="fuzzy_match",
                   max_workers: int = 16):
        """Function to search the enter query in cpp engine."""
        [num_buckets, _, cut_all, stop_words_list, compulsory_words, case_sensitive] = (
            self._obtain_meta(["num_of_shards", "tokenizer", "cut_all",
                               "stop_words_list", "compulsory_words", "case_sensitive"]))
        if search_type == "exact_match":
            if isinstance(query, str):
                cpp_query = query.encode("utf-8").decode('latin-1')
            elif isinstance(query, bool):
                cpp_query = np.array(int(query)).tobytes().decode('latin-1')
            elif isinstance(query, float):
                cpp_query = np.array(query, dtype='<f8').tobytes()
            else:
                cpp_query = np.array(query).tobytes().decode('latin-1')
            full_stop_words = set()
        else:
            cpp_query = query
            full_stop_words = {"", " ", "  ", '\n', '\t'}
            if self._obtain_stop_words(stop_words_list):
                full_stop_words.update(self._obtain_stop_words(stop_words_list))
        from muller.util.sparsehash.build.custom_hash_map import search_idx
        try:
            results = search_idx(cpp_query,
                                 os.path.join(self.dataset.path, self.index_folder),
                                 search_type,
                                 num_buckets,
                                 cut_all,
                                 full_stop_words,
                                 case_sensitive,
                                 max_workers,
                                 "" if compulsory_words is None else compulsory_words)
        except Exception as e:
            raise ExecuteError from e
        return results


    def search(self,
               query: [str, int, bool, float],
               search_type="fuzzy_match",
               max_workers: int = 16):
        """Searches the index for query matches (fuzzy/exact) using parallel processing."""

        meta_data = (self._obtain_meta(["num_of_shards", "tokenizer", "cut_all",
                               "stop_words_list", "compulsory_words", "case_sensitive"]))

        shard_data = self._process_query(
            query, search_type, meta_data
        )

        search_results = self._parallel_search(
            search_type, shard_data['shard_list'],
            shard_data['shard_word_dict'], max_workers
        )

        return self._merge_results(search_results)


    def complex_search(self, query: str, max_workers: int = 16, use_cpp: bool = False):
        """Function to search the type of complex_fuzzy_match."""
        meta_data = self._obtain_meta([
        "num_of_shards", "tokenizer", "cut_all",
        "stop_words_list", "compulsory_words", "case_sensitive"
    ])

        if use_cpp:
            return self._cpp_complex_search(query, meta_data, max_workers)

        # 处理Python搜索
        return self._python_complex_search(query, meta_data, max_workers)


    def _cpp_complex_search(self, query, meta_data, max_workers):
        """使用CPP实现的复杂搜索"""
        from muller.util.sparsehash.build.custom_hash_map import search_idx
        try:
            return search_idx(
                query,
                os.path.join(self.dataset.path, self.index_folder),
                "complex_fuzzy_match",
                meta_data[0],  # num_buckets
                meta_data[2],  # cut_all
                self._get_stop_words(meta_data[3], "fuzzy_match"),  # stop_words
                meta_data[5],  # case_sensitive
                max_workers,
                "" if meta_data[4] is None else meta_data[4]  # compulsory_words
            )
        except Exception as e:
            raise ExecuteError from e


    def _python_complex_search(self, query, meta_data, max_workers):
        """使用Python实现的复杂搜索"""
        # 获取停用词和分词结果
        stop_words = self._get_stop_words(meta_data[3], "fuzzy_match")
        query_tok_dict = self._jieba_tokenize_complex_search(
            query, stop_words, meta_data[4],  # compulsory_words
            meta_data[1], meta_data[2], meta_data[5]  # tokenizer, cut_all, case_sensitive
        )

        if not query_tok_dict:
            return set()

        # 处理shard映射
        shard_data = self._process_complex_query_shards(query_tok_dict, meta_data[0])
        if not shard_data['shard_list']:
            return set()

        # 并行搜索
        search_results = self._parallel_complex_search(
            shard_data['shard_list'],
            shard_data['shard_word_dict'],
            max_workers
        )

        # 合并结果
        return self._merge_complex_results(search_results, shard_data['value_tok_dict'])


    def _process_complex_query_shards(self, query_tok_dict, num_buckets):
        """处理复杂查询的shard映射"""
        shard_word_dict = {}
        value_tok_dict = {}

        for sub_query, words in query_tok_dict.items():
            hash_id_list = []
            for word in words:
                hash_signed, shard_id = self._byte_to_int64(word.encode("utf-8"), num_buckets)
                if shard_id not in shard_word_dict:
                    shard_word_dict[shard_id] = [hash_signed]
                else:
                    shard_word_dict[shard_id].append(hash_signed)
                hash_id_list.append(hash_signed)
            value_tok_dict[sub_query] = hash_id_list

        return {
            'shard_list': list(shard_word_dict.keys()),
            'shard_word_dict': shard_word_dict,
            'value_tok_dict': value_tok_dict
        }


    def _parallel_complex_search(self, shard_list, shard_word_dict, max_workers):
        """并行复杂搜索"""
        num_process = min(len(shard_list), max_workers)
        pool = multiprocessing.Pool(num_process)

        results = []

        for shard_id in shard_list:
            results.append(pool.apply_async(func=self._search_single_shard_for_complex_query,
                                            args=(shard_id,
                                                  shard_word_dict.get(shard_id, None),
                                                  )))
        pool.close()
        pool.join()

        return [res.get() for res in results]


    def _merge_complex_results(self, search_results, value_tok_dict):
        """合并复杂搜索结果"""
        final_res = {}
        for res in search_results:
            for word, doc_ids in res.items():
                final_res[word] = doc_ids

        final_ids = set()
        for _, words in value_tok_dict.items():
            res_doc_ids = final_res.get(words[0], set())
            for word in words:
                res_doc_ids &= final_res.get(word, set())  # 对于一个sub query来说，里面的每个词都需要出现，所以是交集
            final_ids |= res_doc_ids  # 对于不同的subquery，因为是OR的连接关系，所以只需要并集即可
        return final_ids


    def _get_stop_words(self, index_type, stop_words_list):
        """获取停用词集合"""
        if index_type != "fuzzy_match":
            return set()

        stop_words = {"", " ", "  ", '\n', '\t'}
        extra_stop_words = self._obtain_stop_words(stop_words_list)
        if extra_stop_words:
            stop_words.update(extra_stop_words)
        return stop_words


    def _setup_batch_params(self, settings, num_of_batches, num_of_shards, uuids):
        """处理批次参数"""
        params = {
            'num_of_batches': num_of_batches,
            'num_of_shards': num_of_shards,
            'use_uuids': bool(uuids)
        }
        if settings:
            params.update(zip(['num_of_batches', 'num_of_shards', 'use_uuids'],
                              map(int, settings[:3])))
        return params


    def _setup_paths(self, batch_params):
        """处理路径相关操作"""
        tmp_path = os.path.join(self.dataset.path, self.index_folder + "_tmp")
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)

        log_path = os.path.join(tmp_path, self.col_log_folder)
        os.makedirs(log_path, exist_ok=True)

        with open(os.path.join(log_path, self.col_log_file), "w") as f:
            json.dump([
                batch_params['num_of_batches'],
                batch_params['num_of_shards'],
                batch_params['use_uuids']
            ], f)

    def _create_cpp_index(self, batch_params, cut_all, stop_words, compulsory_words, case_sensitive, max_workers):
        """CPP版本索引创建"""
        from muller.util.sparsehash.build.custom_hash_map import IndexProcessor
        import muller.util.sparsehash.build.custom_hash_map as cm

        com_words = "" if compulsory_words is None else compulsory_words

        starts, ends = self._split_data(
            0, len(self.dataset),
            batch_params['num_of_batches'],
            self._obtain_existing_batches(),
            True
        )

        cm.init_logger(os.path.join(self.dataset.path, FILTER_LOG), cm.LogLevel.INFO)
        try:
            IndexProcessor.process_index_parallel(
                self.dataset.path, self.column_name,
                self.index_folder + "_tmp", self.col_log_folder,
                starts, ends,
                min(len(starts), max_workers),
                batch_params['num_of_shards'],
                cut_all,
                case_sensitive,
                stop_words,
                self.commit_id,
                com_words
            )
        except Exception as e:
            raise ExecuteError from e


    def _create_python_index(self, batch_params, stop_words, index_type, tokenizer,
                             cut_all, compulsory_words, case_sensitive, max_workers, uuids):
        """Python版本索引创建"""
        if uuids:
            raise InvertedIndexUnsupportedError("Not support for using uuid")

        ranges = self._split_data( # starts, ends
            0, len(self.dataset),
            batch_params['num_of_batches'],
            self._obtain_existing_batches()
        )

        pool = multiprocessing.Pool(
            processes=min(len(ranges[0]), max_workers),
            maxtasksperchild=1
        )

        tokenizer_params = {
            'tokenizer': tokenizer,
            'cut_all': cut_all,
            'stop_words_list': stop_words,
            'compulsory_words': compulsory_words,
            'case_sensitive': case_sensitive,
            'num_shards': batch_params['num_of_shards']
        }

        for i, start in enumerate(ranges[0]):
            pool.apply_async(
                func=self._process_index,
                args=(
                    i,
                    int(start),
                    int(ranges[1][i] + 1),
                    index_type,
                    tokenizer_params
                )
            )

        pool.close()
        pool.join()


    def _check_index_completion(self, num_of_batches):
        """检查索引完整性"""
        unfinished = self.check_index_completeness(self.index_folder + "_tmp", num_of_batches)
        if not unfinished:
            self.logger.info(f"Creating index of {self.column_name} successfully.")
            return True

        self.logger.info("Creating index fails. There are unfinished batches. "
                         "You may use ds.create_index_vectorized(...) again to finish the creation.")
        return False


    def _load_and_validate_settings(self, use_cpp, index_type):
        """加载并验证索引设置"""
        try:
            meta_json = json.loads(self.storage[self.meta].decode('utf-8'))
            settings = meta_json[self.column_name]
        except KeyError as e:
            raise ValueError("There is no existing index, please create first.") from e

        try:
            before_cpp = settings['use_cpp']
        except KeyError as e:
            raise ValueError("The meta of inverted_index is invalid.") from e

        if use_cpp != before_cpp:
            warnings.warn(
                f"`use_cpp` parameter does not match the original setting ({before_cpp}). "
                f"Using original value instead of {use_cpp}."
            )
            use_cpp = before_cpp

        if use_cpp and index_type == "exact_match":
            raise UnsupportedMethod(
                "Exact match not supported in C++ version. Set `use_cpp=False`."
            )

        return {'use_cpp': use_cpp}


    def _update_with_cpp(self, start_index, end_index, num_of_batches,
                         num_of_shards, max_workers, stop_words,
                         compulsory_words, case_sensitive, cut_all):
        """使用C++更新索引"""
        from muller.util.sparsehash.build.custom_hash_map import IndexProcessor
        import muller.util.sparsehash.build.custom_hash_map as cm

        starts, ends = self._split_data(
            start_index, end_index,
            num_of_batches,
            self._obtain_existing_batches(),
            True
        )

        cm.init_logger(os.path.join(self.dataset.path, FILTER_LOG), cm.LogLevel.INFO)
        try:
            IndexProcessor.process_index_parallel(
                self.dataset.path, self.column_name,
                self.index_folder + "_tmp", self.col_log_folder,
                starts, ends,
                min(len(starts), max_workers),
                num_of_shards,
                cut_all,
                case_sensitive,
                stop_words,
                self.commit_id,
                compulsory_words or ""
            )
        except Exception as e:
            raise ExecuteError from e

    def _update_with_python(self, start_index, end_index, num_of_batches,
                            num_of_shards, max_workers,
                            index_type, tokenizer_params, uuids):
        """使用Python更新索引"""
        if uuids:
            raise InvertedIndexUnsupportedError("UUIDs not supported")

            # 获取处理范围
        ranges = self._split_data( # starts, ends
            start_index, end_index,
            num_of_batches,
            self._obtain_existing_batches()
        )

        # 创建进程池
        pool = multiprocessing.Pool(
            processes=min(len(ranges[0]), max_workers),
            maxtasksperchild=1
        )

        tokenizer_params['num_shards'] = num_of_shards

        # 提交任务
        for i, start in enumerate(ranges[0]):
            pool.apply_async(
                func=self._process_index,
                args=(
                    i,
                    int(start),
                    int(ranges[1][i]) + 1, # range = (start, end) end = ends[i] + 1
                    index_type,
                    tokenizer_params
                )
            )

        pool.close()
        pool.join()


    def _check_update_completion(self, num_of_batches):
        """检查更新完整性"""
        unfinished = self.check_index_completeness(
            self.index_folder + "_tmp", num_of_batches
        )

        if not unfinished:
            self.logger.info("Index updated successfully.")
            return True

        self.logger.info("Index update failed with unfinished batches.")
        return False


    def _process_query(self, query, search_type, meta_data):
        """处理查询并返回shard映射数据"""
        num_buckets = meta_data[0]
        shard_word_dict = {}

        if search_type == "exact_match":
            hash_data = self._get_hash_for_query(query, num_buckets)
            shard_word_dict[hash_data['shard_id']] = [hash_data['hash_signed']]
        else:
            full_stop_words = self._get_stop_words("fuzzy_match", meta_data[3])
            query_words = self._jieba_tokenize(
                query, full_stop_words, meta_data[4],
                meta_data[1], meta_data[2], meta_data[5]
            )
            if not query_words:
                return {'shard_list': [], 'shard_word_dict': {}}

            for word in query_words:
                hash_data = self._get_hash_for_query(word, num_buckets)
                shard_word_dict.setdefault(hash_data['shard_id'], []).append(hash_data['hash_signed'])

        return {
            'shard_list': list(shard_word_dict.keys()),
            'shard_word_dict': shard_word_dict
        }


    def _parallel_search(self, search_type, shard_list, shard_word_dict, max_workers):
        """并行搜索处理"""
        if not shard_list:
            return []

        num_process = min(len(shard_list), max_workers)
        pool = multiprocessing.Pool(num_process)

        results = [
            pool.apply_async(
                func=self._search_single_shard,
                args=(search_type, shard_id, shard_word_dict.get(shard_id))
            )
            for shard_id in shard_list
        ]

        pool.close()
        pool.join()

        return [res.get() for res in results]


    def _merge_results(self, search_results):
        """合并搜索结果"""
        if not search_results:
            return {}

        merged = search_results[0]
        for res in search_results[1:]:
            merged &= res
        return merged


    def _get_hash_for_query(self, query, num_buckets):
        """获取查询的哈希和shard ID"""
        if isinstance(query, str):
            bytes_data = query.encode("utf-8")
        elif isinstance(query, bool):
            bytes_data = np.array(int(query)).tobytes()
        else:
            bytes_data = np.array(query).tobytes()

        hash_signed, shard_id = self._byte_to_int64(bytes_data, num_buckets)
        return {'hash_signed': hash_signed, 'shard_id': shard_id}


    def _process_index(self, batch_count, start: int, end: int, index_type: str,
                       tokenizer_params):
        # 1. 先把该处理的所有行（句子）加载到内存
        shards = [defaultdict(set) for _ in range(tokenizer_params['num_shards'])]
        try:
            # 2. 处理数据集并构建索引
            for i, sample in enumerate(self.dataset[start: end]):
                sample_data = sample[self.column_name].tobytes()

                if index_type == "fuzzy_match":
                    words = self._jieba_tokenize(
                        str(sample_data, "utf-8"),
                        tokenizer_params['stop_words_list'],
                        tokenizer_params['compulsory_words'],
                        tokenizer_params['tokenizer'],
                        tokenizer_params['cut_all'],
                        tokenizer_params['case_sensitive'],
                    )
                    for word in words:
                        self._add_to_shard(shards, word.encode("utf-8"), i + start, tokenizer_params['num_shards'])
                else:
                    self._add_to_shard(shards, sample_data, i + start, tokenizer_params['num_shards'])

            # 3. 保存分片数据
            for shard_info in enumerate(shards): # shard_id, shard
                self._dump_index(self.index_folder + "_tmp", f"{shard_info[0]}/{start}", shard_info[1])

            # 4. 记录处理完成
            self._log_completion(self.index_folder + "_tmp", batch_count, start)

        except Exception as e:
            self.logger.info(f"{batch_count} creation fails because of {e}")


    def _add_to_shard(self, shards, data, line_num, num_of_shards):
        """将数据添加到对应的shard中"""
        hash_signed, shard_id = self._byte_to_int64(data, num_of_shards)
        if hash_signed not in shards[shard_id]:
            shards[shard_id][hash_signed] = set()
        shards[shard_id][hash_signed].add(line_num)


    def _log_completion(self, folder, batch_count, start):
        """记录处理完成的日志"""
        self.logger.info(f"batch {batch_count} (starting with {start}) is finished")
        file_path = os.path.join(folder, self.col_log_folder, str(start))
        self.storage[file_path] = b""
        self.storage.flush()


    def _process_index_cpp(self, batch_count, start, end, num_of_shards,
                           cut_all, case_sensitive, full_stop_words, compulsory_words):
        import muller.util.sparsehash.build.custom_hash_map as cm
        cm.init_logger(os.path.join(self.dataset.path, FILTER_LOG), cm.LogLevel.INFO)
        from muller.util.sparsehash.build.custom_hash_map import IndexProcessor
        return IndexProcessor.process_index_single(
            self.dataset.path, self.column_name, self.index_folder + "_tmp",
            self.col_log_folder, batch_count, start, end,
            num_of_shards, cut_all, case_sensitive, full_stop_words,
            compulsory_words
        )

    def _dump_index(self, path, file_name, data):
        file_path = os.path.join(path, file_name)
        self.storage[file_path] = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
        self.storage.flush()

    def _merge_shards(self,
                      optimize_mode: str,
                      shard_id: int,
                      ):
        try:
            # 0. 查看在optimized文件夹里是否已有目标索引文件
            optimized_index_path = os.path.join(self.dataset.path, self.index_folder + f"_optimized")
            if os.path.exists(optimized_index_path) and str(shard_id) in os.listdir(optimized_index_path):
                self.logger.info(f"Already exists {shard_id}. Skip!")
                return

            merged = defaultdict(set)
            # 注：如果从头到尾只有一个file且当前是create index，其实不需要读上来，直接复制到目标地址即可
            file_list = os.listdir(os.path.join(self.dataset.path, self.index_folder + "_tmp", str(shard_id)))
            current_index_folder = os.path.join(self.dataset.path, self.index_folder)
            if len(file_list) == 1 and optimize_mode != "update":
                shutil.copy(os.path.join(self.dataset.path, self.index_folder + "_tmp", str(shard_id), "0"),
                            os.path.join(self.dataset.path, self.index_folder + "_optimized", str(shard_id)))

            else:
                # 将当前shard_id文件夹下的所有键值合并起来
                for file in file_list:
                    tmp_dict = pickle.loads(self.storage[os.path.join(self.index_folder + "_tmp", str(shard_id), file)])
                    for word, pos_set in tmp_dict.items():
                        merged[word].update(pos_set)

                # 查看一下是否有已有的索引文件夹，如有且当前是update index，则一并将对应的shard_id合并了。
                if os.path.exists(current_index_folder) and optimize_mode == "update":
                    tmp_dict = pickle.loads(self.storage[os.path.join(self.index_folder,
                                                                      str(shard_id))])
                    for word, pos_set in tmp_dict.items():
                        merged[word].update(pos_set)

                # 2. 保存这个新的索引文件
                new_file = str(shard_id)  # 注意：现在这个版本，optimize size一定是1了
                self._dump_index(self.index_folder + f"_optimized", new_file, merged)
            self.logger.info(f"merged shards: {shard_id}")

        except Exception as e:
            self.logger.info(f"{shard_id} merge fails because of {e}")


    def _load_index(self, shard_id):
        return pickle.loads(self.storage[os.path.join(self.index_folder, shard_id)])

    def _search_single_shard(self, search_type, shard_id, word_list):
        try:
            batch = self._load_index(str(shard_id))
        except Exception as e:
            raise InvertedIndexNotFoundError(self.column_name) from e

        _res_doc_ids = set()
        if word_list:
            if search_type == "fuzzy_match":
                _res_doc_ids = batch[word_list[0]]
                for word in word_list[1:]:
                    # 检索结果的合并(取交集，需要每个结果都出现)
                    _res_doc_ids = _res_doc_ids & batch[word]

            elif search_type == "exact_match":
                target_query = word_list[0]
                if target_query in batch.keys():  # 有可能符合的keys为空
                    _res_doc_ids = batch[target_query]

            else:  # search_type=="range_match"
                batch_keys = np.array(list(batch.keys()))
                # 先取出来符合的key
                match_keys = batch_keys[np.logical_and(batch_keys >= word_list[0], batch_keys <= word_list[1])]
                if len(match_keys):  # 有可能符合的keys为空
                    # 再取出来key们对应的值
                    _res_doc_ids = batch[match_keys[0]]
                    for key in match_keys[1:]:
                        tmp_doc_ids = batch[key]
                        _res_doc_ids = _res_doc_ids | tmp_doc_ids

        return _res_doc_ids

    def _search_single_shard_for_complex_query(self, shard_id, word_list):
        try:
            batch = self._load_index(str(shard_id))
        except Exception as e:
            raise InvertedIndexNotFoundError(self.column_name) from e
        _word_doc_dict = {}
        if word_list:
            for word in word_list:
                _word_doc_dict[word] = batch[word] # 这是一个set

        return _word_doc_dict

    def _obtain_meta(self, key_list: list):
        try:
            meta_json = json.loads(self.storage[self.meta].decode('utf-8'))
            meta_list = []
            for key in key_list:
                meta_list.append(meta_json.get(self.column_name).get(key, None))
            return meta_list
        except KeyError as e:
            raise InvertedIndexNotExistsError(self.column_name) from e

    def _reshard_single(self, shard_id: int, new_shard_num: int):
        new_shards = [defaultdict(set) for _ in range(new_shard_num)]
        for file in os.listdir(self.dataset.path + "/" + self.index_folder):
            if file.startswith(str(shard_id) + "_"):
                with open(self.dataset.path + "/" + self.index_folder + "/" + file, 'rb') as f:
                    tmp_dict = pickle.load(f)
                    for word, pos_set in tmp_dict.items():
                        new_shard_id = word % new_shard_num

                        if word not in new_shards[new_shard_id]:
                            new_shards[new_shard_id][word] = pos_set
                        else:
                            new_shards[new_shard_id][word] |= pos_set

        # 4. 每个shard落盘
        count = 0
        for shard in new_shards:
            file_name = str(count) + "_" + str(uuid.uuid4().hex)
            self._dump_index(os.path.join("inverted_index_dir_vec",
                                          self.branch, self.column_name + f"_reshard_{new_shard_num}"),
                             file_name, shard)
            self.logger.info(f"dump index, old shard id: {shard_id}, new file_name: {file_name}")
            count += 1

    def _obtain_hot_data_from_single_shard(self, shard_id, n=1000):
        shard_name_list = self._obtain_shard_name_from_shard_id(shard_id)
        if not shard_name_list:
            return []

        single_data = self._load_index(shard_name_list[0])

        # 使用 heapq.nlargest 找出 set 大小最大的前 n 个 key
        top_n_keys = heapq.nlargest(n, single_data.keys(), key=lambda k: len(single_data[k]))

        return top_n_keys

    def _obtain_shard_name_from_shard_id(self, shard_id):
        shard_name_list = []
        for file in os.listdir(self.dataset.path + "/" + self.index_folder):
            if file.startswith(str(shard_id) + "_"):
                shard_name_list.append(file)
        return shard_name_list

    def _obtain_set_of_key(self, num):
        """ 给出一个num（int64类型），基于现有的索引，直接输出他对应的set """
        num_of_shards = self._obtain_meta(["num_of_shards"])[0]
        shard_id = num % num_of_shards
        shard_name_list = self._obtain_shard_name_from_shard_id(shard_id)
        if not shard_name_list:
            return set()

        single_data = self._load_index(shard_name_list[0])
        return set(single_data.get(num, set()))

    def _obtain_existing_batches(self):
        existing_batches = []
        log_path = os.path.join(self.dataset.path, self.index_folder, self.col_log_folder)
        if os.path.exists(log_path):
            for file in os.listdir(log_path):
                if file.find("log") == -1:
                    existing_batches.append(int(file))
        self.logger.info(f"The following batches are already constructed: {existing_batches}")
        return existing_batches

    def _check_existing_indexes(self, force_create: bool):
        skip = False
        settings = None
        # 首先读一下meta是否存在。
        # 如果meta不存在，说明没有完整的索引。

        if force_create:
            warnings.warn(f"We are going to create a new index to replace the current index.\n"
                          f"Note that the current index still works before we finish the creation and optimization "
                          f"of the new index.")
            return skip, settings

        try:
            meta_json = json.loads(self.storage[self.meta].decode('utf-8'))
        except KeyError:
            # 如果存在上回索引建立的痕迹，说明上回索引建立不全，继续使用上回的默认设置从中断继续恢复即可。
            log_path = os.path.join(self.dataset.path, self.index_folder, self.col_log_folder, self.col_log_file)
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    settings = json.load(f)
                self.logger.info(f"We did not finish the construction of the original indexes. Now we can continue. "
                             f"Note that we will use the original number of batches.")
                self.logger.info(f"Original settings: {settings}")
            # 如果不存在痕迹，则说明未有索引，直接建立即可。
            else:
                self.logger.info(f"There is no existing indexes. Start to create index...")
            meta_json = {}

        # 如果meta存在，说明上回索引建立是完整的。
        if meta_json and meta_json.get(self.column_name):
            warnings.warn("There is already an existing index. Please specify force_create=True when using"
                          "ds.create_index_vectorized() and we will clean the existing index.")
            skip = True
        return skip, settings
