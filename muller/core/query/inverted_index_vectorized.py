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
from muller.core.storage.cache_utils import get_base_storage
from muller.util.exceptions import InvertedIndexNotExistsError, InvertedIndexUnsupportedError, \
    InvertedIndexNotFoundError, ExecuteError, UnsupportedMethod
from muller.util.path import is_remote_path


class InvertedIndexVectorized(object):
    def __init__(self, dataset, storage, branch, column_name: str, use_uuid=False):
        self.dataset = dataset
        self.storage = storage
        self.branch = branch
        self.column_name = column_name
        self.index_folder = os.path.join("inverted_index_dir_vec", branch, column_name)
        self.use_uuid = use_uuid
        # meta.json: Records which column and version the inverted index was built for, along with detailed metadata.
        # [Note: This is different from the log.json file associated with each individual column.]
        self.meta = os.path.join("inverted_index_dir_vec", branch, "meta.json")
        # col_log_folder: A log folder for each column, primarily recording the batches that have already been processed
        # within that column.
        self.col_log_folder = "create_index_record"
        # col_log_file: A log file for each column, primarily recording the parameters used during processing
        # of that column.
        self.col_log_file = "log.json"
        # logger: FILTER_LOG is plain on-disk log. On remote-backed datasets
        # (s3://, huawei-obs://, roma://, mem://, ...) we cannot open it as a
        # local file, so skip the FileHandler and keep only console logging.
        log_path = (
            None
            if is_remote_path(self.dataset.path)
            else self.dataset.path + os.sep + FILTER_LOG
        )
        self.logger = self._set_logger(log_path)
        self.hot_shard_data = None

    @property
    def commit_id(self):
        """Function to compute commit id."""
        if self.dataset.commit_id == FIRST_COMMIT_ID:
            return ""
        return self.dataset.version_state['commit_id']

    def _ensure_local_for_cpp(self):
        """Fail fast if a C++ index path is requested on a remote dataset.

        The C++ engine reads dataset chunks via ``muller::Reader`` (plain
        ``std::ifstream``) and writes shard files via ``fopen`` /
        ``saveToFileNoCompression``. Both bypass MULLER's storage
        abstraction, so the C++ path only works when ``dataset.path``
        resolves to a local directory. We refuse to start instead of
        silently producing a broken or wrong-location index.
        """
        if is_remote_path(self.dataset.path):
            raise UnsupportedMethod(
                f"The C++ inverted-index engine reads and writes the local "
                f"filesystem directly and cannot operate on a remote dataset "
                f"(path='{self.dataset.path}'). Pass use_cpp=False, or run "
                f"this operation against a locally-backed dataset."
            )

    @staticmethod
    def _set_logger(path: Optional[str]):
        # ``logging.getLogger`` returns a process-wide singleton per name, so
        # the name must encode the log destination. The legacy shared name
        # ('my_logger') combined with the ``if not logger.handlers`` guard
        # meant the FileHandler registered by the FIRST instance in the
        # process was silently reused by every later instance, mixing logs
        # across datasets (and, if the first instance was remote-backed,
        # dropping file logging for all later local datasets). Instances that
        # target the same log file share one logger, so handlers are still
        # registered only once per destination. Remote-backed datasets
        # (path=None) share a single console-only logger.
        suffix = re.sub(r"[^0-9A-Za-z]+", "_", path) if path is not None else "console"
        logger = logging.getLogger(f"muller.inverted_index.{suffix}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if not logger.handlers:
            # File logging is only meaningful for a local path; skip it for
            # remote-backed datasets so __init__ does not crash trying to
            # open a non-local file.
            if path is not None:
                file_handler = logging.FileHandler(path, mode='a')
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

            # Console logging
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
            stream_handler.setFormatter(stream_formatter)
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
        # Hash into int64
        hash_signed = mmh3.hash64(byte_data)[0]
        # Compute the shard
        shard_id = hash_signed % num_of_shard
        return hash_signed, shard_id

    @staticmethod
    def _num_to_shard(num, num_of_shard):
        # A relatively simple approach: directly take the modulo of the value. This may lead to load balancing issues!
        shard_id = num % num_of_shard
        return int(shard_id)

    @staticmethod
    def _split_data(start, end, num_of_batches, to_remove, cpp_use=False):
        dataset_length = end - start if cpp_use else end - start + 1
        chunk_size = dataset_length // num_of_batches
        remainder = dataset_length % num_of_batches
        # Compute the size of each sub-array
        sizes = np.full(num_of_batches, chunk_size)
        sizes[:remainder] += 1

        # Compute the start index
        starts = np.cumsum([start] + list(sizes[:-1]))
        ends = starts + sizes - 1

        # shuffle
        shuffled_starts = starts[np.random.permutation(len(starts))]

        # Create a boolean mask (keeping elements not in to_remove)
        mask = ~np.isin(shuffled_starts, np.array(to_remove))
        # Apply the mask, and generate a new array.
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
        # Check if an existing index is present. If a complete index already exists, decide whether to delete and
        # rebuild it based on the value of force_create. Otherwise, proceed with creating the index.
        skip, settings = self._check_existing_indexes(force_create)
        if skip:
            return None
        if settings:
            [num_of_batches, num_of_shards, use_uuids] = settings
            num_of_batches = int(num_of_batches)
            num_of_shards = int(num_of_shards)

        # Define a new temporary index prefix, then delete leftovers from the
        # previous attempt. Python indexing stores these artifacts through
        # MULLER storage; the C++ path still needs a local directory because
        # the native code writes files by path.
        use_uuids = bool(uuids)
        tmp_meta = [num_of_batches, num_of_shards, use_uuids]
        if use_cpp:
            self._ensure_local_for_cpp()
            tmp_path = os.path.join(self.dataset.path, self.index_folder + "_tmp")
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)

            log_path = os.path.join(self.dataset.path, self.index_folder + "_tmp", self.col_log_folder)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
                with open(os.path.join(log_path, self.col_log_file), "w") as f:
                    json.dump(tmp_meta, f)
        else:
            self._delete_storage_prefix(self.index_folder + "_tmp")
            self._save_run_settings(self.index_folder + "_tmp", tmp_meta)

        # Read the stopword list and store it in a list.
        if index_type == "fuzzy_match":
            full_stop_words = {"", " ", "  ", '\n', '\t'}
            stop_words = self._obtain_stop_words(stop_words_list)
            if stop_words:
                full_stop_words.update(stop_words)
        else:
            full_stop_words = set()

        # Create index via multi-processinng
        if use_cpp:
            filtered_starts, filtered_ends = self._split_data(0, len(self.dataset),
                                                              num_of_batches,
                                                              self._obtain_existing_batches(use_cpp=True),
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

        # After creating an index, you need to verify whether the indexes for all batches have been successfully
        # generated. If any are missing, they can be regenerated!
        unfinished_batches = self.check_index_completeness(self.index_folder + "_tmp", num_of_batches,
                                                           use_cpp=use_cpp)
        # In the future, we could consider using a recursive call like
        # unfinished_batches = self.check_create_index_completeness(...) to ensure the completeness of index creation.
        # However, we must also ensure that memory is automatically released if workers are unexpectedly interrupted.

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
        """Merge all shard files under the same shard ID into a single file."""
        if use_cpp:
            self._ensure_local_for_cpp()

        num_of_shards = self._obtain_meta(["num_of_shards"])[0] # Number of shards, not number of shard files

        # If no temporary index file folder has already been generated, raise an error and return immediately.
        tmp_index_path = os.path.join(self.dataset.path, self.index_folder + "_tmp")
        if use_cpp and not os.path.exists(tmp_index_path):
            raise InvertedIndexNotExistsError(self.column_name)
        if not use_cpp and not self._storage_prefix_exists(self.index_folder + "_tmp"):
            raise InvertedIndexNotExistsError(self.column_name)

        # Formally build the index. Note: A new (column_name)_optimized index folder is created here,
        # and all indexes are written into it.
        optimized_index_path = os.path.join(self.dataset.path, self.index_folder + f"_optimized")
        if use_cpp and not os.path.exists(optimized_index_path):
            # It must be created in the main process. If created in a child process, conflicts will occur!
            os.makedirs(optimized_index_path)
        # During merging, you need to determine whether this is a merge during index creation or during an update:
        # Merge during creation: Existing (column_name) index files can be ignored; only the new shards are merged.
        # Merge during update: The contents of the existing (column_name) folder must also be included and
        # merged together.
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
            mp_context = multiprocessing.get_context("fork") if hasattr(os, "fork") else multiprocessing
            pool = mp_context.Pool(num_process)
            # Collect AsyncResults so worker exceptions propagate to the
            # main process via ``.get()`` below, instead of being silently
            # swallowed by ``close() + join()``.
            results = [
                pool.apply_async(func=self._merge_shards,
                                 args=(optimize_mode, i,))
                for i in range(num_of_shards)
            ]
            pool.close()
            pool.join()
            for res in results:
                res.get()

        if use_cpp:
            # Rename the original index folder (if it exists) to col_[uuid].
            # Then, rename the col_optimized folder to col, and delete the col_tmp folder.
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
        else:
            self._promote_storage_prefix(self.index_folder + "_optimized", self.index_folder)
            self._delete_storage_prefix(self.index_folder + "_tmp")
            self._delete_storage_prefix(self.index_folder + "_optimized")
            self.logger.info(f"Generate new index prefix of {self.column_name} successfully.")


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

        # Initialization checks and setup
        settings = self._load_and_validate_settings(use_cpp, index_type)

        # Get stop words
        full_stop_words = self._get_stop_words(index_type, tokenizer_params['stop_words_list'])

        # Update index
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

        # Check whether update is complete
        return self._check_update_completion(num_of_batches, use_cpp=settings['use_cpp'])


    def check_index_completeness(self, folder: str, num_of_batches: int, *, use_cpp: bool = False):
        """Check how many per-batch completion markers are still missing.

        For ``use_cpp=True`` the markers are written by the native engine
        via ``std::ofstream`` directly to the local filesystem, so the
        storage abstraction never sees them and we must inspect the local
        FS. For the Python path the markers are written through
        ``self.storage[...]`` and the local FS is irrelevant (and wrong on
        remote-backed datasets).
        """
        current_batches = set()
        if use_cpp:
            path = os.path.join(self.dataset.path, folder, self.col_log_folder)
            if os.path.exists(path):
                for file_name in os.listdir(path):
                    if file_name.find(self.col_log_file) == -1:
                        current_batches.add(int(file_name))
        else:
            current_batches = set(self._list_completed_batches(folder))
        unfinished_batches = num_of_batches - len(current_batches)
        return unfinished_batches

    def _storage_keys_under(self, prefix: str):
        prefix = prefix.rstrip("/")
        prefix_with_slash = f"{prefix}/"
        keys = set(self.storage._all_keys())
        try:
            keys.update(get_base_storage(self.storage)._all_keys(refresh_tag=True))
        except TypeError:
            keys.update(get_base_storage(self.storage)._all_keys())
        except Exception:
            pass
        return sorted(
            key for key in keys
            if key == prefix or key.startswith(prefix_with_slash)
        )

    def _storage_prefix_exists(self, prefix: str):
        return bool(self._storage_keys_under(prefix))

    def _delete_storage_prefix(self, prefix: str):
        for key in self._storage_keys_under(prefix):
            del self.storage[key]
        self.storage.flush()

    def _promote_storage_prefix(self, source_prefix: str, target_prefix: str):
        self._delete_storage_prefix(target_prefix)
        source_prefix = source_prefix.rstrip("/")
        source_prefix_with_slash = f"{source_prefix}/"
        for source_key in self._storage_keys_under(source_prefix):
            relative_key = (
                source_key[len(source_prefix_with_slash):]
                if source_key.startswith(source_prefix_with_slash)
                else source_key.rsplit("/", 1)[-1]
            )
            self.storage[os.path.join(target_prefix, relative_key)] = self.storage[source_key]
        self.storage.flush()

    def _run_settings_key(self, folder: str):
        return os.path.join(folder, self.col_log_folder, self.col_log_file)

    def _save_run_settings(self, folder: str, settings: list):
        self.storage[self._run_settings_key(folder)] = json.dumps(settings).encode("utf-8")
        self.storage.flush()

    def _load_run_settings(self, folder: str):
        return json.loads(self.storage[self._run_settings_key(folder)].decode("utf-8"))

    def _list_completed_batches(self, folder: str):
        log_prefix = os.path.join(folder, self.col_log_folder)
        completed_batches = []
        for key in self._storage_keys_under(log_prefix):
            file_name = key.rsplit("/", 1)[-1]
            if file_name.find(self.col_log_file) == -1:
                completed_batches.append(int(file_name))
        return completed_batches


    def reshard_index(self, old_shard_num: int, new_shard_num: int, max_workers: int = 16):
        """Function to re-shard index."""
        with multiprocessing.Pool(min(old_shard_num, max_workers)) as pool:
            optimize_batch = [pool.apply_async(func=self._reshard_single,
                                               args=(i, old_shard_num, new_shard_num))
                              for i in range(old_shard_num)]
            # Use ``.get()`` rather than ``.wait()`` so worker exceptions
            # propagate to the main process instead of being silently dropped.
            for res in optimize_batch:
                res.get()

    def add_hot_shard(self, max_workers: int = 16, n: int = 100000):
        """Select the top n most frequently occurring terms from the existing shards
        and write them into our hot shard!"""
        num_of_shards = self._obtain_meta(["num_of_shards"])[0]
        with multiprocessing.Pool(min(num_of_shards, max_workers)) as pool:
            results = [pool.apply_async(func=self._obtain_hot_data_from_single_shard,
                                               args=(i, ))
                              for i in range(num_of_shards)]
            # Wait for all the above tasks to be finished
            for res in results:
                res.wait()

            results = [res.get() for res in results]

        top_n = heapq.nlargest(n, (num for lst in results for num in lst))

        with multiprocessing.Pool(min(len(top_n), max_workers)) as pool:
            results = [pool.apply_async(func=self._obtain_set_of_key,
                                        args=(num,))
                       for num in top_n]
            # Wait for all the above tasks to be finished
            for res in results:
                res.wait()

            results = [res.get() for res in results]

        final_dict = defaultdict(set)
        for i, num in enumerate(top_n):
            final_dict[num] = results[i]

        # Dump to storage
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
        self._ensure_local_for_cpp()
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

        # Python search
        return self._python_complex_search(query, meta_data, max_workers)


    def _cpp_complex_search(self, query, meta_data, max_workers):
        """CPP complex search"""
        self._ensure_local_for_cpp()
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
        """Python complex search"""
        # Retrieve the stopword list and tokenization results.
        stop_words = self._get_stop_words(meta_data[3], "fuzzy_match")
        query_tok_dict = self._jieba_tokenize_complex_search(
            query, stop_words, meta_data[4],  # compulsory_words
            meta_data[1], meta_data[2], meta_data[5]  # tokenizer, cut_all, case_sensitive
        )

        if not query_tok_dict:
            return set()

        # Process shard mappinng
        shard_data = self._process_complex_query_shards(query_tok_dict, meta_data[0])
        if not shard_data['shard_list']:
            return set()

        # Parallel complex search
        search_results = self._parallel_complex_search(
            shard_data['shard_list'],
            shard_data['shard_word_dict'],
            max_workers
        )

        # Merge results
        return self._merge_complex_results(search_results, shard_data['value_tok_dict'])


    def _process_complex_query_shards(self, query_tok_dict, num_buckets):
        """Process shard mapping"""
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
        """Parallel complex search"""
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
        """Merge search results"""
        final_res = {}
        for res in search_results:
            for word, doc_ids in res.items():
                final_res[word] = doc_ids

        final_ids = set()
        for _, words in value_tok_dict.items():
            res_doc_ids = final_res.get(words[0], set())
            for word in words:
                # For a sub-query, every term within it must appear, so the result is the intersection
                # of the postings for those terms.
                res_doc_ids &= final_res.get(word, set())
            # For different subqueries, since they are connected by an OR relationship,
            # a union of their results is sufficient.
            final_ids |= res_doc_ids
        return final_ids


    def _get_stop_words(self, index_type, stop_words_list):
        """Get stop words"""
        if index_type != "fuzzy_match":
            return set()

        stop_words = {"", " ", "  ", '\n', '\t'}
        extra_stop_words = self._obtain_stop_words(stop_words_list)
        if extra_stop_words:
            stop_words.update(extra_stop_words)
        return stop_words


    def _setup_batch_params(self, settings, num_of_batches, num_of_shards, uuids):
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
        self._delete_storage_prefix(self.index_folder + "_tmp")
        self._save_run_settings(
            self.index_folder + "_tmp",
            [
                batch_params['num_of_batches'],
                batch_params['num_of_shards'],
                batch_params['use_uuids']
            ],
        )

    def _create_cpp_index(self, batch_params, cut_all, stop_words, compulsory_words, case_sensitive, max_workers):
        """Create cpp index"""
        self._ensure_local_for_cpp()
        from muller.util.sparsehash.build.custom_hash_map import IndexProcessor
        import muller.util.sparsehash.build.custom_hash_map as cm

        com_words = "" if compulsory_words is None else compulsory_words

        starts, ends = self._split_data(
            0, len(self.dataset),
            batch_params['num_of_batches'],
            self._obtain_existing_batches(use_cpp=True),
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
        """Create python index"""
        if uuids:
            raise InvertedIndexUnsupportedError("Not support for using uuid")

        ranges = self._split_data( # starts, ends
            0, len(self.dataset),
            batch_params['num_of_batches'],
            self._obtain_existing_batches(use_cpp=False)
        )

        mp_context = multiprocessing.get_context("fork") if hasattr(os, "fork") else multiprocessing
        pool = mp_context.Pool(
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

        # Collect AsyncResults so worker exceptions propagate to the main
        # process via ``.get()`` below. Previously this loop was fire-and-
        # forget, which made every worker-side failure (unpicklable args,
        # ``spawn`` re-importing an unfindable ``__main__`` under hosts like
        # ``streamlit run`` or ``jupyter``, read-only storage, etc.) invisible
        # — the only trace was a missing batch-completion marker that later
        # made ``check_index_completeness`` return "unfinished".
        results = [
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
            for i, start in enumerate(ranges[0])
        ]

        pool.close()
        pool.join()
        for res in results:
            res.get()


    def _check_index_completion(self, num_of_batches, *, use_cpp: bool = False):
        unfinished = self.check_index_completeness(self.index_folder + "_tmp", num_of_batches, use_cpp=use_cpp)
        if not unfinished:
            self.logger.info(f"Creating index of {self.column_name} successfully.")
            return True

        self.logger.info("Creating index fails. There are unfinished batches. "
                         "You may use ds.create_index_vectorized(...) again to finish the creation.")
        return False


    def _load_and_validate_settings(self, use_cpp, index_type):
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
        """Update index with c++"""
        self._ensure_local_for_cpp()
        from muller.util.sparsehash.build.custom_hash_map import IndexProcessor
        import muller.util.sparsehash.build.custom_hash_map as cm

        starts, ends = self._split_data(
            start_index, end_index,
            num_of_batches,
            self._obtain_existing_batches(use_cpp=True),
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
        """Update index with Python"""
        if uuids:
            raise InvertedIndexUnsupportedError("UUIDs not supported")

            # Get the data range
        ranges = self._split_data( # starts, ends
            start_index, end_index,
            num_of_batches,
            self._obtain_existing_batches(use_cpp=False)
        )

        # Initialize process pool
        mp_context = multiprocessing.get_context("fork") if hasattr(os, "fork") else multiprocessing
        pool = mp_context.Pool(
            processes=min(len(ranges[0]), max_workers),
            maxtasksperchild=1
        )

        tokenizer_params['num_shards'] = num_of_shards

        # Submit the tasks. Collect AsyncResults so worker exceptions
        # propagate via ``.get()`` below instead of being silently dropped
        # (see analogous comment in ``_create_python_index``).
        results = [
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
            for i, start in enumerate(ranges[0])
        ]

        pool.close()
        pool.join()
        for res in results:
            res.get()


    def _check_update_completion(self, num_of_batches, *, use_cpp: bool = False):
        unfinished = self.check_index_completeness(
            self.index_folder + "_tmp", num_of_batches, use_cpp=use_cpp
        )

        if not unfinished:
            self.logger.info("Index updated successfully.")
            return True

        self.logger.info("Index update failed with unfinished batches.")
        return False


    def _process_query(self, query, search_type, meta_data):
        """Process the query and return shard mapping data."""
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
        """Parallel search"""
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
        """Merge search results"""
        if not search_results:
            return {}

        merged = search_results[0]
        for res in search_results[1:]:
            merged &= res
        return merged


    def _get_hash_for_query(self, query, num_buckets):
        """Obtain the query's hash and shard ID."""
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
        # 1. First, load all the rows (sentences) to be processed into memory.
        shards = [defaultdict(set) for _ in range(tokenizer_params['num_shards'])]
        try:
            # 2. Process the dataset and build the index.
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

            # 3. Save the sharded data.
            for shard_info in enumerate(shards): # shard_id, shard
                self._dump_index(self.index_folder + "_tmp", f"{shard_info[0]}/{start}", shard_info[1])

            # 4. Log that processing is complete.
            self._log_completion(self.index_folder + "_tmp", batch_count, start)

        except Exception as e:
            # Log then re-raise so the caller's ``AsyncResult.get()`` surfaces
            # the real error in the main process instead of silently leaving a
            # missing batch-completion marker behind.
            self.logger.info(f"{batch_count} creation fails because of {e}")
            raise


    def _add_to_shard(self, shards, data, line_num, num_of_shards):
        hash_signed, shard_id = self._byte_to_int64(data, num_of_shards)
        if hash_signed not in shards[shard_id]:
            shards[shard_id][hash_signed] = set()
        shards[shard_id][hash_signed].add(line_num)


    def _log_completion(self, folder, batch_count, start):
        self.logger.info(f"batch {batch_count} (starting with {start}) is finished")
        file_path = os.path.join(folder, self.col_log_folder, str(start))
        self.storage[file_path] = b""
        self.storage.flush()


    def _process_index_cpp(self, batch_count, start, end, num_of_shards,
                           cut_all, case_sensitive, full_stop_words, compulsory_words):
        self._ensure_local_for_cpp()
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
            # 0. Check whether the target index file already exists in the optimized folder.
            optimized_index_prefix = self.index_folder + f"_optimized"
            if self._storage_prefix_exists(os.path.join(optimized_index_prefix, str(shard_id))):
                self.logger.info(f"Already exists {shard_id}. Skip!")
                return

            merged = defaultdict(set)
            # Note: If there is only a single file from start to finish and the current operation is creating an index,
            # there's no need to read it into memory—just copy it directly to the target location.
            tmp_shard_prefix = os.path.join(self.index_folder + "_tmp", str(shard_id))
            file_list = [
                key.rsplit("/", 1)[-1]
                for key in self._storage_keys_under(tmp_shard_prefix)
            ]
            if len(file_list) == 1 and optimize_mode != "update":
                self.storage[os.path.join(optimized_index_prefix, str(shard_id))] = self.storage[
                    os.path.join(tmp_shard_prefix, file_list[0])
                ]
                self.storage.flush()

            else:
                # Merge all key-value pairs under the current shard_id folder.
                for file in file_list:
                    tmp_dict = pickle.loads(self.storage[os.path.join(self.index_folder + "_tmp", str(shard_id), file)])
                    for word, pos_set in tmp_dict.items():
                        merged[word].update(pos_set)

                # Check if an existing index folder is present;
                # if so, and the current operation is an index update, merge the corresponding shard_id as well.
                if self._storage_prefix_exists(self.index_folder) and optimize_mode == "update":
                    tmp_dict = pickle.loads(self.storage[os.path.join(self.index_folder,
                                                                      str(shard_id))])
                    for word, pos_set in tmp_dict.items():
                        merged[word].update(pos_set)

                # 2. Save this new index file.
                new_file = str(shard_id)  # Note: In this version, the optimize size is always 1.
                self._dump_index(self.index_folder + f"_optimized", new_file, merged)
            self.logger.info(f"merged shards: {shard_id}")

        except Exception as e:
            # Log then re-raise so the caller's ``AsyncResult.get()`` surfaces
            # the real error in the main process instead of silently leaving a
            # half-merged optimized folder behind.
            self.logger.info(f"{shard_id} merge fails because of {e}")
            raise


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
                    # Merge search results (take the intersection — each term must appear in the results).
                    _res_doc_ids = _res_doc_ids & batch[word]

            elif search_type == "exact_match":
                target_query = word_list[0]
                if target_query in batch.keys():  # It's possible that the set of matching keys is empty.
                    _res_doc_ids = batch[target_query]

            else:  # search_type=="range_match"
                batch_keys = np.array(list(batch.keys()))
                # First, extract the matching keys.
                match_keys = batch_keys[np.logical_and(batch_keys >= word_list[0], batch_keys <= word_list[1])]
                if len(match_keys):  # It's possible that the set of matching keys is empty.
                    # Then, retrieve the values corresponding to those keys.
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
                _word_doc_dict[word] = batch[word] # This is a set

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
        # Iterate shards through the storage abstraction so this works on
        # any backend (local/S3/OBS/...). Preserves the legacy
        # ``<shard_id>_<uuid>`` file-naming convention used by reshard
        # output; behavior on files that don't match the pattern is
        # unchanged.
        prefix = self.index_folder.rstrip("/")
        name_prefix = f"{str(shard_id)}_"
        for key in self._storage_keys_under(prefix):
            file_name = key.rsplit("/", 1)[-1]
            if file_name.startswith(name_prefix):
                tmp_dict = pickle.loads(self.storage[key])
                for word, pos_set in tmp_dict.items():
                    new_shard_id = word % new_shard_num

                    if word not in new_shards[new_shard_id]:
                        new_shards[new_shard_id][word] = pos_set
                    else:
                        new_shards[new_shard_id][word] |= pos_set

        # 4. Dump each shard to storage
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

        # Use heapq.nlargest to find the top n keys with the largest set sizes.
        top_n_keys = heapq.nlargest(n, single_data.keys(), key=lambda k: len(single_data[k]))

        return top_n_keys

    def _obtain_shard_name_from_shard_id(self, shard_id):
        """Enumerate per-shard reshard-output filenames via storage.

        Reshard output (see ``_reshard_single``) names files as
        ``<shard_id>_<uuid>``. We list them through ``self.storage`` so the
        lookup is backend-agnostic.
        """
        prefix = self.index_folder.rstrip("/")
        name_prefix = f"{str(shard_id)}_"
        shard_name_list = []
        for key in self._storage_keys_under(prefix):
            file_name = key.rsplit("/", 1)[-1]
            if file_name.startswith(name_prefix):
                shard_name_list.append(file_name)
        return shard_name_list

    def _obtain_set_of_key(self, num):
        """ Given a num (of type int64), directly output its corresponding set based on the existing index."""
        num_of_shards = self._obtain_meta(["num_of_shards"])[0]
        shard_id = num % num_of_shards
        shard_name_list = self._obtain_shard_name_from_shard_id(shard_id)
        if not shard_name_list:
            return set()

        single_data = self._load_index(shard_name_list[0])
        return set(single_data.get(num, set()))

    def _obtain_existing_batches(self, *, use_cpp: bool = False):
        """Return the list of batch start-indexes whose creation markers exist.

        For ``use_cpp=True`` the markers are local FS files (written by the
        native engine). For the Python path they live exclusively under
        ``self.storage``.
        """
        existing_batches = []
        if use_cpp:
            log_path = os.path.join(self.dataset.path, self.index_folder, self.col_log_folder)
            if os.path.exists(log_path):
                for file in os.listdir(log_path):
                    if file.find("log") == -1:
                        existing_batches.append(int(file))
        else:
            existing_batches = self._list_completed_batches(self.index_folder)
        self.logger.info(f"The following batches are already constructed: {existing_batches}")
        return existing_batches

    def _check_existing_indexes(self, force_create: bool):
        skip = False
        settings = None
        # First, check whether the meta file exists. If it doesn't, it means a complete index has not been built.

        if force_create:
            warnings.warn(f"We are going to create a new index to replace the current index.\n"
                          f"Note that the current index still works before we finish the creation and optimization "
                          f"of the new index.")
            return skip, settings

        try:
            meta_json = json.loads(self.storage[self.meta].decode('utf-8'))
        except KeyError:
            # If traces of a previous indexing attempt exist (run-settings
            # blob in storage), it indicates that the prior index build was
            # incomplete; resume from the interruption using its parameters.
            # We deliberately consult only the storage abstraction here: a
            # legacy local-FS fallback at ``self.dataset.path/.../log.json``
            # was previously checked first, but (a) it bypasses the storage
            # layer entirely (incorrect for remote-backed datasets) and (b)
            # it looked at ``self.index_folder`` while the write side puts
            # the settings under ``self.index_folder + "_tmp"`` -- so it
            # could never actually match on either backend. Dropped.
            if self._storage_prefix_exists(self._run_settings_key(self.index_folder)):
                settings = self._load_run_settings(self.index_folder)
                self.logger.info(f"We did not finish the construction of the original indexes. Now we can continue. "
                             f"Note that we will use the original number of batches.")
                self.logger.info(f"Original settings: {settings}")
            # If no traces exist, it means no index has been created yet, so proceed to build one directly.
            else:
                self.logger.info(f"There is no existing indexes. Start to create index...")
            meta_json = {}

        # If the meta file exists, it indicates that the previous index build was complete.
        if meta_json and meta_json.get(self.column_name):
            warnings.warn("There is already an existing index. Please specify force_create=True when using"
                          "ds.create_index_vectorized() and we will clean the existing index.")
            skip = True
        return skip, settings
