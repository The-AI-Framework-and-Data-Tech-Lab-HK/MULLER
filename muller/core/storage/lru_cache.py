# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/storage/lru_cache.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import dataclasses
import json
import sys
from collections import OrderedDict
from typing import Dict, Optional, Union, Set, Any, List, Iterable
from typing import Tuple, Type, Callable, Sequence

from muller.constants import VERSION_CONTROL_INFO_FILENAME, DATASET_META_FILENAME
from muller.core.chunk.base_chunk import BaseChunk
from muller.core.meta.tensor_meta import TensorMeta
from muller.core.partial_reader import PartialReader
from muller.core.storage.muller_memory_object import MULLERMemoryObject
from muller.core.storage.memory import MemoryProvider
from muller.core.storage.provider import StorageProvider
from muller.core.storage.util import _get_nbytes, identity
from muller.util.exceptions import ReadOnlyModeError
from muller.util.json import HubJsonDecoder, HubJsonEncoder


def obj_to_bytes(obj):
    """Object to bytes"""
    if isinstance(obj, MULLERMemoryObject):
        obj = obj.tobytes()
    if isinstance(obj, memoryview):
        obj = bytes(obj)
    if isinstance(obj, dict):  # tensor meta
        temp_dict = {}
        for key, value in obj.items():
            d = {str(k): v for k, v in value.__getstate__().items()}
            temp_dict[key] = d
        temp_bytes = bytes(json.dumps(temp_dict, sort_keys=True, indent=4, cls=HubJsonEncoder), "utf-8")
        obj = temp_bytes
    return obj


@dataclasses.dataclass
class LRUCacheProcessFuncs:
    """过个规范而已啦...
        Args:
            read_process_f: called for every reading operation's output after reading to do some processing
            (e.g. deserialization).

            write_cache_process_f: called for every writing into cache_storage operation's input before writing to
            do some processing (e.g. deserialization).

            write_next_process_f: called for every writing into next_storage operation's input before writing to
            do some processing (e.g. deserialization).
    """
    read_process_f: Callable = identity
    write_cache_process_f: Callable = identity
    write_next_process_f: Callable = identity


class LRUCache(StorageProvider):
    """LRU Cache that uses StorageProvider for caching"""

    def __init__(
            self,
            cache_storage: StorageProvider,
            next_storage: Optional[StorageProvider],
            cache_size: int,
    ):
        """Initializes the LRUCache. It can be chained with other LRUCache objects to create multilayer caches.

        Args:
            cache_storage (StorageProvider): The storage being used as the caching layer of the cache.
                This should be a base provider such as MemoryProvider, LocalProvider or RomaProvider,
                but not another LRUCache.
            next_storage (StorageProvider): The next storage layer of the cache.
                This can either be a base provider (i.e. it is the final storage) or another LRUCache
                (i.e. in case of chained cache).
                While reading data, all misses from cache would be retrieved from here.
                While writing data, the data will be written to the next_storage when cache_storage is full or
                flush is called.
            cache_size (int): The total space that can be used from the cache_storage in bytes.
                This number may be less than the actual space available on the cache_storage.
                Setting it to a higher value than actually available space may lead to unexpected behaviors.
        """
        self.next_storage = next_storage
        self.cache_storage = cache_storage
        self.cache_size = cache_size

        # tracks keys in lru order, stores size of value, only keys present in this exist in cache
        self.lru_sizes: OrderedDict[str, int] = OrderedDict()

        self.dirty_keys: Dict[str, None] = (
            OrderedDict() if sys.version_info < (3, 7) else {}  # type: ignore
        )  # keys present in cache but not next_storage. Using a dict instead of set to preserve order.

        self.cache_used = 0
        self.muller_objects: Dict[str, MULLERMemoryObject] = {}
        # Sherry: path for dataset_meta.json, version_control... and their values
        # Sherry: BRING THIS BACK AFTER ASYNC IS FIXED
        self.upper_cache = {}  # for uuids, future filter results, etc.
        self.upper_cache_merge = {}
        # track the version states to record the commit users for authentication checking
        self.version_state = None  # will be updated by checkout/commit/auto_commit... in version control
        self.version_state_storage = None  # fetch and record the version state from storage

    def __getitem__(self, key: str):
        """If item is in cache_storage, retrieves from there and returns.
        If item isn't in cache_storage, retrieves from next storage, stores in cache_storage (if possible) and returns.

        Args:
            key (str): The path relative to the root of the underlying storage.

        Raises:
            KeyError: if an object is not found at the path.

        Returns:
            bytes: The bytes of the object present at the path.
        """
        result = self.get_item_from_cache(key)
        if result:
            return result

        if self.next_storage is not None:
            # fetch from storage, may throw KeyError
            result = self.next_storage[key]

            if _get_nbytes(result) <= self.cache_size:  # insert in cache if it fits
                self.insert_in_cache(key, result)
            return result
        raise KeyError(key)

    def __setitem__(self, key: str, value: Union[bytes, MULLERMemoryObject, dict]):
        """Puts the item in the cache_storage (if possible), else writes to next_storage.

            Args:
                key (str): the path relative to the root of the underlying storage.
                value (bytes): the value to be assigned at the path.

            Raises:
                ReadOnlyError: If the provider is in read-only mode.
        """
        # Check the state of provider
        assert key != '_uuid'
        self.check_readonly()
        if key in self.muller_objects:
            if isinstance(self.muller_objects[key], dict):
                for temp in self.muller_objects[key].values():
                    temp.is_dirty = False
            else:
                self.muller_objects[key].is_dirty = False

        if key in self.lru_sizes:
            size = self.lru_sizes.pop(key)
            self.cache_used -= size

        if _get_nbytes(value) <= self.cache_size:
            self.insert_in_cache(key, value)
            self.dirty_keys[key] = None
            if isinstance(value, MULLERMemoryObject):
                value.is_dirty = False
        else:  # larger than cache, directly send to next layer
            self._forward_value(key, value)
        self.maybe_flush()

    def __delitem__(self, key: str):
        # Check the state of provider
        self.check_readonly()

        # Conduct the delete operation
        deleted_from_cache = False

        if key in self.muller_objects:
            self.remove_muller_object(key)
            deleted_from_cache = True

        if key in self.lru_sizes:
            size = self.lru_sizes.pop(key)
            self.cache_used -= size
            del self.cache_storage[key]
            self.dirty_keys.pop(key, None)
            deleted_from_cache = True

        try:
            if self.next_storage is not None:
                del self.next_storage[key]
            else:
                raise KeyError(key)
        except KeyError:
            if not deleted_from_cache:
                raise

    def __len__(self):
        """Returns the number of files present in the cache and the underlying storage.

        Returns:
            int: the number of files present inside the root.
        """
        return len(self._all_keys())

    # Sherry: Need to be supplemented

    def __iter__(self):
        """Generator function that iterates over the keys of the cache and the underlying storage.

        Yields:
            str: the key of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self._all_keys()

    # Sherry: Need to be supplemented

    def __getstate__(self) -> Dict[str, Any]:
        """Returns the state of the cache, for pickling"""

        # flushes the cache before pickling
        self._flush_if_not_read_only()

        return {
            "next_storage": self.next_storage,
            "cache_storage": MemoryProvider(),
            "cache_size": self.cache_size,
        }

    def __setstate__(self, state: Dict[str, Any]):
        """Recreates a cache with the same configuration as the state.

        Args:
            state (dict): The state to be used to recreate the cache.

        Note:
            While restoring the cache, we reset its contents.
            In case the cache storage was local/s3 and is still accessible when unpickled
            (if same machine/s3 creds present respectively), the earlier cache contents are no longer accessible.
        """
        # Sherry: We might want to change this behaviour in the future by having a separate file that keeps a track of
        #  the lru order for restoring the cache.
        # This would also allow the cache to persist across different different Dataset objects pointing to
        # the same dataset.
        self.next_storage = state["next_storage"]
        self.cache_storage = state["cache_storage"]
        self.cache_size = state["cache_size"]
        self.lru_sizes = OrderedDict()
        self.dirty_keys = OrderedDict()
        self.cache_used = 0
        self.muller_objects = {}
        self.upper_cache = {}
        self.upper_cache_merge = {}
        self.version_state = None
        self.version_state_storage = None

    def get_muller_object(
            self,
            key: str,
            expected_class,
            meta: Optional[List] = None,
            url=False,
            partial_bytes: int = 0,
    ):
        """Get the MULLER object. """
        if partial_bytes != 0:
            assert issubclass(expected_class, BaseChunk)
            if key in self.lru_sizes:
                return self[key]
            buff = self.get_bytes(key, 0, partial_bytes)
            obj = expected_class.frombuffer(buff, meta, partial=True)
            obj.data_bytes = PartialReader(self, key, header_offset=obj.header_bytes)
            if obj.nbytes <= self.cache_size:
                self.insert_in_cache(key, obj)
            return obj
        if url:
            from muller.util.remove_cache import get_base_storage

            item = get_base_storage(self).get_presigned_url(key).encode("utf-8")
            if issubclass(expected_class, BaseChunk):
                obj = expected_class.frombuffer(item, meta, url=True)
                return obj

            raise ValueError(
                "Expected class should be subclass of BaseChunk when url is True."
            )

        item = self[key]

        if isinstance(item, MULLERMemoryObject):
            if not isinstance(item, expected_class):
                raise ValueError(
                    f"'{key}' was expected to have the class '{expected_class.__name__}'. Instead got: '{type(item)}'."
                )
            return item

        if isinstance(item, (bytes, memoryview)):
            if expected_class == dict:
                obj = json.loads(item, cls=HubJsonDecoder)
                total_bytes = 0
                for k, v in obj.items():  # for each tensor meta
                    temp_obj = TensorMeta()
                    temp_obj.__setstate__(v)
                    temp_obj.is_dirty = False
                    obj[k] = temp_obj
                    total_bytes += temp_obj.nbytes
                if total_bytes <= self.cache_size:
                    self.insert_in_cache(key, obj)
                return obj

            obj = (
                expected_class.frombuffer(obj_to_bytes(item))
                if meta is None
                else expected_class.frombuffer(obj_to_bytes(item), meta)
            )
            if obj.nbytes <= self.cache_size:
                self.insert_in_cache(key, obj)

            return obj

        if isinstance(item, dict):  # tensor_meta.json
            return item

        raise ValueError(f"Item at '{key}' got an invalid type: '{type(item)}'.")

    def bytes_to_muller_object(self,
                             byts: bytes,
                             expected_class: Type,
                             meta: Optional[Dict] = None,
                             ) -> Union[bytes, MULLERMemoryObject]:
        """Deserialize a bytes object into a MULLERMemoryObject calling MULLERMemoryObject.frombuffer.
            This function handles several situations in the following cases:
            1) If expected_class is not a subclass of MULLERMemoryObject, then raise ValueError.
            2) If expected_class is bytes, then just return byts.
            3) Call expected_class.frombuffer(byts, (meta)).

            Args:
                byts: bytes

                expected_class: Must be bytes or a subclass of MULLERMemoryObject.

                meta: Second argument of expected_class.from_buffer, None means no second argument.

            returns:
                byts or expected_class.from_buffer(byts, (meta)).
        """
        if expected_class is bytes:
            return byts
        if not issubclass(expected_class, MULLERMemoryObject):
            raise ValueError("Expected a subclass of MULLERMemoryObject, got {}".format(expected_class))
        obj = (
            expected_class.frombuffer(byts)
            if meta is None
            else expected_class.frombuffer(byts, meta)
        )
        return obj

    def muller_object_to_bytes(self,
                              muller_object: MULLERMemoryObject) -> bytes:
        """serialize a MULLERMemoryObject into bytes by calling MULLERMemoryObject.tobytes().
            This function handles several situation in the following cases:

            Args:
                muller_object (MULLERMemoryObject): MULLERMemoryObject to be serialized.

            returns:
                muller_object.tobytes()
        """
        if not isinstance(muller_object, MULLERMemoryObject):
            raise ValueError(f"Argument muller_object must be an instance of MULLERMemoryObject, got {type(muller_object)}")
        return muller_object.tobytes()

    def process_to_muller_object(self,
                               data: Union[bytes, MULLERMemoryObject],
                               expected_class,
                               meta: Optional[Dict] = None
                               ) -> MULLERMemoryObject:
        """Handle several corner cases to generalize bytes_to_muller_object, to support the API of process_function.
            If data is already a MULLERMemoryObject, then check if isinstance(data, expected_class), since deserialization
            can not be done twice, otherwise call bytes_to_muller_object.
        """
        if expected_class is None:
            return data

        if isinstance(data, MULLERMemoryObject):
            if not isinstance(data, expected_class):
                raise ValueError(f"Argument byts is not bytes, this case argument expected_class is expected to be "
                                 f"byts's class ({type(data)}), but got {expected_class}")
            return data
        if isinstance(data, bytes):
            return self.bytes_to_muller_object(data, expected_class, meta)
        raise ValueError(f'data must be an instance of bytes or MULLERMemoryObject, got {type(data)}')

    def process_to_bytes(self, data: Union[bytes, MULLERMemoryObject]) -> bytes:
        """Handle several corner cases to generalize muller_object_to_bytes, to support the API of process_function.
            If data is already a bytes, then just return bytes, otherwise call muller_object_to_bytes.
        """
        if isinstance(data, bytes):
            return data
        return self.muller_object_to_bytes(data)

    def get_multiple_muller_objects_with_muller_objects(self,
                                                    keys: List[str],
                                                    expected_class: Optional[Type] = None,
                                                    meta: Optional[Any] = None,
                                                    ) -> List[Any]:
        """Check if self.muller_objects has any data required before calling get_multiple_muller_objects.
             This function may be removed and implemented in another subclass in the future.
        """
        # 不管所有muller_objects里面的东西，直接筛掉
        new_paths = []  # 直接把所有muller_objects里的东西筛掉
        for key in keys:
            if key not in self.muller_objects:
                new_paths.append(key)
        # 调用
        new_results = self.get_multiple_muller_objects(keys=new_paths, expected_class=expected_class, meta=meta)
        # 把muller_objects里面的东西再加进来返回
        results = []
        i = 0
        for key in keys:
            if key in self.muller_objects:
                results.append(self.muller_objects[key])
            else:
                results.append(new_results[i])
                i += 1
        return results

    def get_multiple_muller_objects(self,
                                  keys: List[str],
                                  expected_class: Optional[Type] = None,
                                  meta: Optional[Any] = None,
                                  ) -> List[Any]:
        """This function may be renamed as get_multiple_files, while the get_multiple_files function now may be defined
        in a parent class in the future.
            Fetch from cache and next_storage the data and do deserialization or serialization (if data is popped from
        cache). The storage will store the deserialized result if it can do that (namely, if it's MemoryProvider). Any
        data, if serialized, should always be serialized in the same way, otherwise this function raises ValueError.

        Args:
            keys: the key for every reading request, sorted in chronological order.

            expected_class: the class to deserialize into, the class must be a MULLERMemoryObject, None means no
            deserialization is applied.

            metas: The second argument of expected_class.from_buffer, None means no second argument.

        Returns:
            The data as a list with each element corresponding to each path in paths.
        """

        # Define serializing and deserializing functions.
        def deser_f(data: Union[bytes, MULLERMemoryObject]) -> MULLERMemoryObject:
            return self.process_to_muller_object(data, expected_class, meta)

        def ser_f(data: Union[bytes, MULLERMemoryObject]) -> bytes:
            return self.process_to_bytes(data)

        process_fs = LRUCacheProcessFuncs(read_process_f=deser_f,
                                          write_cache_process_f=identity if isinstance(
                                              self.cache_storage,
                                              MemoryProvider) else ser_f,
                                          write_next_process_f=ser_f)
        data = self.get_items_update(keys, process_fs=process_fs)
        return data

    def get_items(self,
                  keys: Set[str],
                  ignore_key_error: bool = False,
                  process_fs: Optional[LRUCacheProcessFuncs] = None,
                  ) -> Dict[str, Any]:
        """适应Provider接口, 实际上调用有序读版本的get_items_update."""
        key_lst = list(keys)
        results = self.get_items_update(keys=key_lst,
                                        ignore_key_error=ignore_key_error,
                                        process_fs=process_fs)
        return dict(zip(key_lst, results))

    def get_items_update(self,
                         keys: Sequence[str],
                         ignore_key_error: bool = False,
                         process_fs: Optional[LRUCacheProcessFuncs] = None,
                         ) -> List[Any]:
        """This function may be removed and implemented in another parent class in the future.
            The cache reads multiple files from both cache_storage and next_storage, and update cache according to the
        order given by keys.
            This function simply adjusts the API and call the more general responds_to_multiple function.

            Args:
                keys: the key for every reading request, sorted in chronological order.

                ignore_key_error: Whether to ignore KeyError. The response to a reading request that raises a KeyError
                will be None.

                process_fs: read_process_f, write_cache_process_f and write_next_process_f.

            Returns:
                The results
        """
        io_types = ['r'] * len(keys)
        contents = [None] * len(keys)
        return self.respond_to_multiple(keys, io_types=io_types,
                                        contents=contents,
                                        ignore_key_error=ignore_key_error,
                                        process_fs=process_fs)

    def respond_to_multiple(self,
                            keys: Sequence[str],
                            io_types: Sequence[str],
                            contents: Sequence[Any],
                            ignore_key_error: bool = False,
                            process_fs: Optional[LRUCacheProcessFuncs] = None,
                            ) -> List[Any]:
        """This function may be removed and implemented in another parent class in the future.
            The cache receives multiple io(reading/writing/deleting) requests, deals with them altogether and responds
            to all of them.
            Compared to responding to every request one by one, this function merges multiple requests and responds to
        them in an efficient way, also keeping the cache's data almost the same (if not better) as the one by one case
        in the end.
            Except for merging duplicate values in keys, this function's performance is directly related to the
        performance of self.cache_storage and self.next_storage's performance in io multiple files, to be exact, the
        performance of get_multiple_files, set_multiple_files and delete_multiple_files of self.cache_storage and
        self.next_storage.

            Args:
                keys: the key for every io request, sorted in chronological order.

                io_types: specifying every request's type, corresponding to keys one by one, with 'r', 'w' and 'd'
                representing reading, writing and deleting respectively. 'r' and 'w' may have no difference in this
                function.

                contents: contents of every writing request, corresponding to keys one by one, only valid when the
                corresponding io_type is 'w', otherwise its values doesn't matter.

                ignore_key_error: Whether to ignore KeyError. The response to a reading request that raises a KeyError
                will be None.

                process_fs: read_process_f, write_cache_process_f and write_next_process_f.

            Returns:
                responses to all the request, where the response to a reading request is its corresponding data, and to
                writing and deleting is None. Note that the data is processed by next_to_cache_f if it's fetched from
                next_storage.

            Note: Only the case where all io_types are 'r' is supported now.
            Note: Actually this function calls get/set/delete_multiple_files_mp now to avoid conflicts with previous
            code.
        """
        self._check_io_types(io_types=io_types)
        if process_fs is None:
            process_fs = LRUCacheProcessFuncs()

        data = {}  # Record all the data needed.

        # 1. Get all the needed data
        keys_st = set(keys)
        data.update(self.get_items_no_update(keys=keys_st,
                                             ignore_key_error=ignore_key_error,
                                             process_f=process_fs.read_process_f))
        # ignore_key_error重整
        keys_filtered = list(data.keys())
        # 检查
        if not ignore_key_error and set(data) != keys_st:
            raise RuntimeError("The data found doesn't match given keys, something went wrong.")

        # 2. Update self.lru_sizes and self.cache_used
        try:  # 这个key 1000%在data里，为什么还需要捕获错误？这种多此一举的流水线要求早就该删掉！！！
            sizes = [self.lru_sizes.get(key, _get_nbytes(data[key]))
                     for key in keys_filtered]
        except KeyError as e:
            raise e
        added, removed = self._update_meta_after_multiple(keys=keys_filtered,
                                                          io_types=['r'] * len(keys_filtered),
                                                          sizes=sizes)

        # 3. Update self.cache_storage and self.next_storage
        # 3.1. Remove the data in self.cache_storage according to removed, write the dirty data among it into
        # next_storage according to self.dirty_keys and update self.dirty_keys
        self._update_data(added=added,
                          removed=removed,
                          data=data,
                          write_cache_process_f=process_fs.write_cache_process_f,
                          write_next_process_f=process_fs.write_next_process_f)

        # 4. Get the responses
        return [data[key] if key in data else None for key in keys]

    def get_items_no_update(self,
                            keys: Set[str],
                            ignore_key_error: bool = False,
                            process_f: Callable = identity) -> Dict[str, Any]:
        """This function may be removed and implemented in another parent class in the future.
            The cache reads multiple files from both cache_storage and next_storage, but never updates anything.
        Note that here no deserialization is applied, the original data in cache_storage or next_storage is provided.
        Since there's only reading data, this function is thread-safe, and can be called in some multithreading tasks.

            Args:
                keys: the key for every reading request

                ignore_key_error: If True, if True, keys that raise KeyError, then the output may not
                include all key in keys.

                process_f: called for every reading request to do some processing (e.g. deserialization).

            Returns:
                {key: content for key in keys}
        """
        st = set(self.lru_sizes)
        keys_in, keys_out = keys.intersection(st), keys.difference(st)
        dic = {}
        dic.update(self.cache_storage.get_items(keys_in, ignore_key_error=False))
        dic.update(self.next_storage.get_items(keys_out, ignore_key_error=ignore_key_error))
        return {key: process_f(value) for (key, value) in dic.items()}

    def set_items(self, contents: Dict[str, Any]):
        """Set items. """
        raise NotImplementedError(f'set_items is not implemented for {self.__class__.__name__}')

    def del_items(self, keys: Set[str]):
        """Delete items. """
        self.check_readonly()
        key_set = set()
        for key in keys:
            if key in self.muller_objects:
                self.remove_muller_object(key)
            if key in self.lru_sizes:
                size = self.lru_sizes.pop(key)
                self.cache_used -= size
                self.dirty_keys.pop(key, None)
                key_set.add(key)

        self.cache_storage.del_items(key_set)
        if self.next_storage is not None:
            self.next_storage.del_items(keys)

    def clear_target_upper_cache(self, filed, commit_id, tensor_name):
        """clear the given filed contains of upper cache."""
        if self.upper_cache.get(filed, {}).get(commit_id, {}).get(tensor_name, {}):
            self.upper_cache['uuids'][commit_id][tensor_name] = None

    def add_records_cache_merge(
            self,
            records
    ):
        """Function to add values of upper_cache_merge."""
        if records.original_commit_id not in self.upper_cache_merge:
            self.upper_cache_merge[records.original_commit_id] = {}
        if records.target_commit_id not in self.upper_cache_merge[records.original_commit_id]:
            self.upper_cache_merge[records.original_commit_id][records.target_commit_id] = {}

        self.upper_cache_merge[records.original_commit_id][records.target_commit_id][records.tensor_name] = {
            'app_ori_idx': records.app_ori_idx, 'app_tar_idx': records.app_tar_idx, 'delete_ori': records.delete_ori,
            'delete_tar': records.delete_tar, 'original_id_to_index_map': records.original_id_to_index_map,
            'target_id_to_index_map': records.target_id_to_index_map, 'updated_indexes': records.updated_indexes,
            'detect_conflicts': records.detect_conflicts}


    def get_records_cache_merge(
            self,
            target_id: str,
            tensor_name:str,
            original_id: str
    ):
        """Function to get the value of upper_cache_merge."""
        return self.upper_cache_merge.get(original_id, {}).get(target_id, {}).get(tensor_name, {})

    def remove_muller_object(self, key: str):
        """Remove the MULLER objects."""
        self.muller_objects.pop(key, None)

    def clear_muller_objects(self):
        """Clear the MULLER objects."""
        self.muller_objects.clear()

    def register_muller_object(self, key: str, obj: MULLERMemoryObject):
        """Registers a new object in the cache."""
        assert key != '_uuid'
        self.muller_objects[key] = obj

    def clear_cache(self):
        """
        Flushes the content of all the cache layers if not in read mode and then deletes contents
           of all the layers of it.
        This doesn't delete data from the actual storage.
        """
        self._flush_if_not_read_only()
        self.clear_cache_without_flush()

    def clear_cache_without_flush(self):
        """Clear the cache without flushing. """
        self.cache_used = 0
        self.lru_sizes.clear()
        self.dirty_keys.clear()
        self.cache_storage.clear()
        self.muller_objects.clear()
        if self.next_storage is not None and hasattr(self.next_storage, "clear_cache"):
            self.next_storage.clear_cache()

    def flush(self, streaming_mode=True):
        # Sherry: Can we accumulate a set of files first before writing to next_storage?
        """Writes data from cache_storage to next_storage. Only the dirty keys are written.
        This is a cascading function and leads to data being written to the final storage in case of a chained cache.
        """
        self.check_readonly()
        initial_autoflush = self.autoflush
        self.autoflush = False

        for key, obj in self.muller_objects.items():
            if isinstance(obj, dict):  # tensor meta
                for _, v in obj.items():
                    if v.is_dirty:
                        self[key] = obj
                        v.is_dirty = False
            elif obj.is_dirty:
                self[key] = obj
                obj.is_dirty = False

        if self.dirty_keys:
            if hasattr(self.next_storage, "set_items"):
                d = {
                    key: obj_to_bytes(self.cache_storage[key])
                    for key in self.dirty_keys
                }
                self.next_storage.set_items(d)
                self.dirty_keys.clear()
            else:
                # Sherry: We should do batch processing here in ROMA
                if streaming_mode:
                    for key in self.dirty_keys.copy():
                        self._forward(key)
                else:
                    full_key_list = []
                    for key in self.dirty_keys.copy():  # why need a .copy() here?
                        full_key_list.append(key)
                    self._forward_list(full_key_list)

                if self.next_storage is not None:
                    self.next_storage.flush()

        self.autoflush = initial_autoflush

    def clear(self, prefix=""):
        """Clear the LRU Cache."""
        # Check the state of provider
        self.check_readonly()

        # Conduct the clear operation
        if prefix:
            rm = [key for key in self.muller_objects if key.startswith(prefix)]
            for key in rm:
                self.remove_muller_object(key)

            rm = [key for key in self.lru_sizes if key.startswith(prefix)]
            for key in rm:
                size = self.lru_sizes.pop(key)
                self.cache_used -= size
                self.dirty_keys.pop(key, None)

        else:
            self.cache_used = 0
            self.lru_sizes.clear()
            self.dirty_keys.clear()
            self.muller_objects.clear()

        self.cache_storage.clear(prefix=prefix)
        if self.next_storage is not None:
            self.next_storage.clear(prefix=prefix)

    def get_object_size(self, key: str) -> int:
        """Obtain the object size. """
        if key in self.muller_objects:
            return self.muller_objects[key].nbytes

        try:
            return self.cache_storage.get_object_size(key)
        except KeyError:
            if self.next_storage is not None:
                return self.next_storage.get_object_size(key)
            raise

    def obtain_target_user_name_from_storage(self, commit_id):
        """Function to obtain target user name or version state."""
        try:
            target_user_name = self.version_state_storage["commits"][commit_id]["commit_user_name"]
        except (TypeError, KeyError):
            # There are two situations that we need to fetch version state from storage
            # 1. TypeError: self.version_state_storage is None.
            # 2. KeyError: the commit_id is not found. We need to fetch the most updated version state from storage.
            self.version_state_storage = json.loads(self.next_storage[VERSION_CONTROL_INFO_FILENAME]
                                                    .decode('utf8').replace("'", '"'))
            existing_commits = self.version_state_storage["commits"]
            if commit_id in existing_commits.keys():
                target_user_name = existing_commits[commit_id]["commit_user_name"]
            else:
                target_user_name = None  # should be a checkpoint instead of a version.
        return target_user_name

    def obtain_dataset_creator_name_from_storage(self):
        """Function to obtain creator name."""
        try:
            dataset_creator_name = self.version_state["meta"].dataset_creator
        except (TypeError, KeyError):
            dataset_meta = json.loads(self.next_storage[DATASET_META_FILENAME]
                                                    .decode('utf8').replace("'", '"'))
            dataset_creator_name = dataset_meta["dataset_creator"]
        return dataset_creator_name

    def check_readonly(self):
        """
        Raises a ReadOnlyModeError exception if the provider is in read-only mode.
        """
        if self.read_only:
            raise ReadOnlyModeError()

    def get_item_from_cache(self, key: str):
        """Get item from the cache. """
        # avoid multi thread pop cache problem using try ... except ...
        try:
            if key in self.muller_objects:
                if key in self.lru_sizes:
                    self.lru_sizes.move_to_end(key)  # refresh position for LRU
                return self.muller_objects[key]
            if key in self.lru_sizes:
                self.lru_sizes.move_to_end(key)  # refresh position for LRU
                return self.cache_storage[key]
            return None
        except Exception:
            return None

    def insert_in_cache(self, key: str, value: Union[bytes, MULLERMemoryObject, dict]):
        """Helper function that adds a key value pair to the cache.

        Args:
            key (str): the path relative to the root of the underlying storage.
            value (bytes): the value to be assigned at the path.

        Raises:
            ReadOnlyError: If the provider is in read-only mode.
        """
        self._free_up_space(_get_nbytes(value))
        self.cache_storage[key] = value  # type: ignore

        self._update_used_cache_for_key(key, _get_nbytes(value))

    def _forward(self, key):
        """Forward the value at a given path to the next storage, and un-marks its key."""
        if self.next_storage is not None:
            self._forward_value(key, self.cache_storage[key])

    def _forward_value(self, key, value):
        if self.next_storage is not None:
            self.dirty_keys.pop(key, None)
            if isinstance(value, MULLERMemoryObject):
                self.next_storage[key] = value.tobytes()  # Sherry: write into the next-storage path
            else:
                self.next_storage[key] = value

    def _forward_list(self, keys: Iterable[str]):
        """The is the batch version of self._forward() and self._forward_value.
        We forward the values of a list of given paths to the next storage, and un-mark their keys.
        """
        content_dict = {}
        for key in keys:
            if self.next_storage is not None:
                self.dirty_keys.pop(key, None)
                value = self.cache_storage[key]
                if isinstance(value, MULLERMemoryObject):
                    content_dict.update({key: value.tobytes()})
                else:
                    content_dict.update({key: value})
        # "Batch mode"
        self.next_storage.set_items(content_dict)

    def _all_keys(self) -> Set[str]:
        next_key_set = set()
        if self.next_storage is not None:
            next_key_set = self.next_storage._all_keys()  # type: ignore
        cache_ket_set = self.cache_storage._all_keys()
        key_set = set().union(next_key_set, cache_ket_set)
        for key, obj in self.muller_objects.items():
            if isinstance(obj, dict):  # tensor meta
                for _, v in obj.items():
                    if v.is_dirty:
                        key_set.add(key)
            elif obj.is_dirty:
                key_set.add(key)
        return key_set

    def _free_up_space(self, extra_size: int):
        """Helper function that frees up space the required space in cache.
            No action is taken if there is sufficient space in the cache.

        Args:
            extra_size (int): the space that needs is required in bytes.
        """
        while self.cache_used > 0 and extra_size + self.cache_used > self.cache_size:
            self._pop_from_cache()

    def _pop_from_cache(self):
        """Helper function that pops the least recently used key, value pair from the cache"""
        key, itemsize = self.lru_sizes.popitem(last=False)
        if key in self.dirty_keys:
            self._forward(key)
        del self.cache_storage[key]
        self.cache_used -= itemsize

    def _check_io_types(self, io_types: Sequence[str]):
        """检查io_types合法性. 注意目前只支持读."""
        for io_type in io_types:
            if not io_type == 'r':
                raise NotImplementedError(f'Only reading requests is supported now')

    def _update_meta_after_multiple(self, keys: Sequence[str], io_types: Sequence[str], sizes: Sequence[int]) -> Tuple[
        Set[str], Set[str]]:
        """This function may be removed and implemented in another parent class in the future.
            Update self.lru_sizes and self.cache_used after responding to multiple io requests.
            Note that this function is not called after actually responding to multiple io requests, instead it pretends
        that the cache has done __getitem__/__setitem__/__delitem__ for these requests in the given order, and update
        self.lru_sizes and self.cache_used to a state that stores the most recently used data.
            This function finds the state that the cache stores the maximum amount of data, while ensuring that
        self.cache_used does not exceed self.cache_size, so it may find a state different from which the cache goes into
        after responding to those requests one by one (actually always more keys in self.lru_sizes and larger
        self.cache_used).
            This function may be used when the cache tries to deal with multiple io requests at the same time (i.e. in
        one call from outside), but also updates self.lru_sizes and self.cache_used according to last recently used
        principles
            Note that keys can have duplicate values.
            Note that the meta dirty_keys is not updated in this method, and must be updated outside.

        Args:
            keys: the key for every io request, sorted in chronological order.

            io_types: specifying every request's type, corresponding to keys one by one, with 'r', 'w' and 'd'
                representing reading, writing and deleting respectively. 'r' and 'w' may have no difference in this
                function.

            sizes: The size of the files (in bytes) corresponding to keys one by one. Note that the size of deleting
                operation is not required, and its value doesn't matter.

        Returns:
            added, namely set(new_lru_sizes).difference(set(old_lru_sizes))

            removed, namely set(old_lru_sizes).difference(set(new_lru_sizes))

            These two combined give the information about the old state, which can be used for updating
        self.cache_storage and self.next_storage in the future.
        """
        # Step 1: add all the future keys
        cur_size = 0
        future_sizes = OrderedDict()  # store the future items to be added later
        deleted = set()
        for i in range(-1, -len(keys) - 1, -1):
            key, size, typ = keys[i], sizes[i], io_types[i]
            if typ != 'd':
                if key not in future_sizes and key not in deleted:
                    if cur_size + size > self.cache_size:
                        break
                    cur_size += size
                    future_sizes[key] = size
            else:
                deleted.add(key)

        # Step 2: update self.lru_sizes and self.cache_used and record the change into added and removed
        added = set()
        removed = set()

        # Step 2.1: remove key from the old self.lru_sizes
        cur_size += self.cache_used
        # Step 2.1.1: remove keys in deleted and future_sizes
        for key in deleted.union(set(future_sizes)):
            if key in self.lru_sizes:
                cur_size -= self.lru_sizes.pop(key)
                removed.add(key)
        # Step 2.1.2: remove keys not used recently
        while cur_size > self.cache_size:
            key, size = self.lru_sizes.popitem(last=False)
            cur_size -= size
            removed.add(key)

        # Step 2.2: add keys in future_sizes
        while len(future_sizes) > 0:
            key, size = future_sizes.popitem(last=True)
            if key not in removed and key not in self.lru_sizes:
                added.add(key)
            self.lru_sizes[key] = size
            removed.discard(key)  # added key now shouldn't be in removed

        # Step 3: self.cache_used not updated before, now update it to the final cur_size
        self.cache_used = cur_size

        return added, removed

    def _update_data(self,
                     added: Set[str],
                     removed: Set[str],
                     data: Optional[Dict],
                     write_cache_process_f: Optional[Callable] = None,
                     write_next_process_f: Optional[Callable] = None,
                     ) -> None:
        """过流水线的，只在respond_to_multiple里面使用.
            根据所给的增量与减量更新cache和storage中的数据以及dirty_keys，data为内存中已有的数据，避免二次读取.
        """
        # 3.1.1. Find out which to read then write and which to delete
        dirty_results = {}  # record the dirty data to write into next_storage
        to_read_from_cache, to_delete_from_cache = set(), set()
        for key in removed:
            if key in self.dirty_keys:
                if key in data:
                    dirty_results[key] = write_next_process_f(data[key])
                else:
                    to_read_from_cache.add(key)
                self.dirty_keys.pop(key)
            else:
                to_delete_from_cache.add(key)
        # 3.1.2. Read and write
        read_from_cache = self.cache_storage.get_items(to_read_from_cache)
        # process
        for key, value in read_from_cache.items():
            read_from_cache[key] = write_next_process_f(value)
        dirty_results.update(read_from_cache)
        if dirty_results:
            self.next_storage.set_items(dirty_results)
        # 3.1.3 Delete
        if to_delete_from_cache:
            self.cache_storage.del_items(to_delete_from_cache)

        # 3.2 Write into self.cache_storage
        # process
        write_into_cache = {}
        for key in added:
            if key not in data:
                raise RuntimeError(f"Cannot find key: {key} in data retrieved, something went wrong.")
            write_into_cache[key] = write_cache_process_f(data[key])
        if write_into_cache:
            self.cache_storage.set_items(write_into_cache)

    def _flush_if_not_read_only(self):
        """Flushes the cache if not in read-only mode."""
        if not self.read_only:
            self.flush()

    def _update_used_cache_for_key(self, key: str, new_size: int):
        if new_size < 0:
            raise ValueError(f"`new_size` must be >= 0. Got: {new_size}")
        if key in self.lru_sizes:
            old_size = self.lru_sizes[key]
            self.cache_used -= old_size
        self.cache_used += new_size
        self.lru_sizes[key] = new_size
