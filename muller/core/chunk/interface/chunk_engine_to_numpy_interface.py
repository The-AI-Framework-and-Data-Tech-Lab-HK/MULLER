# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/chunk_engine.py
#
# Modifications Copyright (c) 2026 Xueling Lin

import bisect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from typing import (
    Any,
    Optional,
    Union,
    List,
    Tuple
)

import numpy as np

from muller.constants import (
    DATASET_UUID_NAME,
    MAX_WORKERS_FOR_CHUNK_ENGINE
)
from muller.core.index.index import Index
from muller.core.meta.encode.chunk_id import ChunkIdEncoder
from muller.util.exceptions import GetChunkError, DynamicTensorNumpyError
from muller.util.exceptions import NumpyDataNotContinuousError
from muller.util.exceptions import ReadSampleFromChunkError
from muller.util.keys import get_chunk_key


def numpy(
        chunk_engine,
        index: Index,
        aslist: bool = False,
        index_list: List = None,
        **kwargs,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Returns as numpy. """

    # Read arguments.
    use_data_cache = kwargs.get('use_data_cache', False)
    fetch_chunks = kwargs.get('fetch_chunks', False)
    max_workers = kwargs.get('max_workers', MAX_WORKERS_FOR_CHUNK_ENGINE)
    continuous = kwargs.get('continuous', False)
    full = kwargs.get('full', False)
    batch_random_access = kwargs.get('batch_random_access', False)
    parallel = kwargs.get('parallel', None)
    fetch_chunks = fetch_chunks or _get_full_chunk(chunk_engine, index)

    return (chunk_engine.sequence_numpy if chunk_engine.is_sequence else chunk_engine.protected_numpy)(
        # Sherry: check how to deal with sequence situation.
        index=index, aslist=aslist, use_data_cache=use_data_cache, fetch_chunks=fetch_chunks,
        max_workers=max_workers, continuous=continuous, full=full, batch_random_access=batch_random_access,
        index_list=index_list, parallel=parallel
    )


def arrow(
        chunk_engine,
        index: Index,
        aslist: bool = False,
        use_data_cache: bool = True,
        fetch_chunks: bool = False,
    ):
    """Returns as arrow."""
    fetch_chunks = fetch_chunks or _get_full_chunk(chunk_engine, index)
    return chunk_engine.protected_numpy(index=index, aslist=aslist, use_data_cache=use_data_cache,
                                        fetch_chunks=fetch_chunks, as_arrow=True)


def protected_numpy(
        chunk_engine,
        index: Index,
        aslist: bool = False,
        index_list=None,
        **kwargs
) -> Union[np.ndarray, List[np.ndarray]]:
    """Reads samples from chunks and returns as a numpy array. If `aslist=True`, returns a sequence of numpy arrays.

    Args:
        chunk_engine (Chunk Engine): the chunk that we fetch the numpy from.
        index (Index): Represents the samples to read from chunks. See `Index` for more information.
        aslist (bool): If True, the samples will be returned as a list of numpy arrays. If False, returns a single
                       numpy array. Defaults to False. For polygons, aslist is always True.
        use_data_cache (bool): If True, the data cache is used to speed up the read if possible.
                               If False, the data cache is ignored. Defaults to True.
        fetch_chunks (bool): If True, full chunks will be retrieved from the storage,
                             otherwise only required bytes will be retrieved.
            This will always be True even if specified as False in the following cases:
            - The tensor is ChunkCompressed
            - The chunk which is being accessed has more than 128 samples.
        max_workers (int): max workers used in thread pool.
        continuous (bool): The data is continuous.

    Raises:
        DynamicTensorNumpyError: If shapes of the samples being read are not all the same.
        GetChunkError: If a chunk cannot be retrieved from the storage.
        ReadSampleFromChunkError: If a sample cannot be read from a chunk.
        GetDataFromLinkError: If data cannot be retrieved from a link.

    Returns:
        Union[np.ndarray, List[np.ndarray]]: Either a list of numpy arrays or a single numpy array
        (depending on the `aslist` argument).
    """

    # Read arguments.
    use_data_cache = kwargs.get('use_data_cache', False)
    fetch_chunks = kwargs.get('fetch_chunks', False)
    streaming_mode = kwargs.get('streaming_mode', True)
    as_arrow = kwargs.get('as_arrow', False)
    max_workers = kwargs.get('max_workers', MAX_WORKERS_FOR_CHUNK_ENGINE)
    continuous = kwargs.get('continuous', False)
    full = kwargs.get('full', False)
    batch_random_access = kwargs.get('batch_random_access', False)
    parallel = kwargs.get('parallel', None)
    fetch_chunks = fetch_chunks or _get_full_chunk(chunk_engine, index)

    if chunk_engine.tensor_meta.htype == "polygon":
        aslist = True
    if use_data_cache and chunk_engine.is_data_cachable and chunk_engine.key != DATASET_UUID_NAME:
        samples = _numpy_from_data_cache(chunk_engine, index, chunk_engine.num_samples, aslist, as_arrow)
    else:

        # Sherry: Here we start to get the samples in a batch mode
        if _validate_batch_samples(chunk_engine, fetch_chunks):
            if continuous and chunk_engine.is_fixed_shape:
                samples = get_samples_continuous(chunk_engine, index, max_workers)
            elif full and chunk_engine.is_fixed_shape:
                samples = get_samples_full(chunk_engine, None, max_workers)
            elif batch_random_access:
                samples = get_samples_batch_random_access(chunk_engine,
                                                          index_list=index_list,
                                                          max_workers=max_workers,
                                                          parallel=parallel)
            else:
                samples = get_samples(chunk_engine, index, aslist, as_arrow, max_workers)
        else:
            # Sherry: We can fetch all the chunks (from remote or local storage to memory) in a batch mode
            if not streaming_mode:
                _protected_numpy_unstreaming_mode(chunk_engine, index)

            # Here we start to get individual samples
            samples = _obtain_individual_samples(chunk_engine, index, fetch_chunks, as_arrow, aslist)

    if aslist and all(map(np.isscalar, samples)):
        samples = list(arr.item() for arr in samples)
    if not index.values[0].subscriptable():
        samples = samples[0]
    if aslist:
        return samples
    if as_arrow:
        import pyarrow as pa
        return pa.array(samples)
    return np.array(samples)


def _validate_batch_samples(chunk_engine, fetch_chunks):
    from muller.core.storage import MemoryProvider  # Sherry: import recursion risk
    return (
                fetch_chunks
                and not chunk_engine.is_video
                and not isinstance(chunk_engine.base_storage, MemoryProvider)
        )


def _obtain_individual_samples(chunk_engine, index, fetch_chunks, as_arrow, aslist):
    samples = []
    last_shape = None
    for global_sample_index in index.values[0].indices(chunk_engine.num_samples):
        try:
            sample = chunk_engine.get_single_sample(
                global_sample_index,
                index,
                fetch_chunks=fetch_chunks,
            )
        except GetChunkError as e:
            raise GetChunkError(
                e.chunk_key, global_sample_index, chunk_engine.name
            ) from e
        except ReadSampleFromChunkError as e:
            raise ReadSampleFromChunkError(
                e.chunk_key, global_sample_index, chunk_engine.name
            ) from e

        check_sample_shape(sample.shape, last_shape, chunk_engine.key, index, aslist)
        last_shape = sample.shape
        if chunk_engine.tensor_meta.htype == "polygon":
            sample = [p.__array__() for p in sample]
        if as_arrow and chunk_engine.tensor_meta.dtype != "List":
            sample = sample[0]
        samples.append(sample)
    return samples


def _protected_numpy_unstreaming_mode(chunk_engine, index):
    if not index.values[0].value.start:
        temp_start = 0
    else:
        temp_start = index.values[0].value.start
    global_sample_index_list = list(range(temp_start, index.values[0].value.stop))
    chunk_engine.get_chunks_for_multi_samples(global_sample_index_list)


def get_samples_continuous(
        chunk_engine,
        index: Index,
        max_workers: int = MAX_WORKERS_FOR_CHUNK_ENGINE):
    """Get samples for the given index, index must be continuous, fetches chunks in parallel.

    Args:
        chunk_engine: The chunk engine to use.
        index (Index): Index applied on the tensor.
        max_workers(int): max workers used in thread pool.

    Returns:
        List of samples.
    """
    idxs = list(index.values[0].indices(chunk_engine.num_samples))
    if idxs != list(range(idxs[0], idxs[-1] + 1)):
        raise NumpyDataNotContinuousError()

    all_chunk_names = [hex(item[0]).split('x')[-1] for item in chunk_engine.chunk_id_encoder.encoded]
    # 通过二分查找，从chunk_index/unshard算出需要哪些chunks，再多进程读这些chunks，最后拼起来
    last_id_list = [line[1] for line in chunk_engine.chunk_id_encoder.encoded]
    first_chunk_index = bisect.bisect_left(last_id_list, idxs[0]) # 从这个chunk开始读
    last_chunk_index = bisect.bisect_left(last_id_list, idxs[-1]) # 读到这个chunk为止就够了
    chunk_names = all_chunk_names[first_chunk_index: last_chunk_index + 1]

    results = []

    # 先计算第一个chunk
    if chunk_engine.translate_to_local_index(idxs[0], first_chunk_index) == 0:  # 第一个chunk要全读
        results.append(_get_chunk_numpy_full(chunk_engine, chunk_names[0]))
    else:
        results.append(_get_chunk_numpy_continuous(chunk_engine,
                                                   chunk_names[0],
                                                   chunk_engine.translate_to_local_index(idxs[0],
                                                   first_chunk_index),
                                                   int(last_id_list[first_chunk_index])))
    # 去掉头尾，中间的chunk
    with ThreadPoolExecutor(min(max_workers, len(chunk_names))) as executor:  # 注：这里用ProcessPool会导致文件锁报错
        for chunk_name in chunk_names[1: -1]:
            results.append(executor.submit(_get_chunk_numpy_full, chunk_engine, chunk_name).result())

    # 最后一个chunk
    if len(chunk_names) >= 2:
        if idxs[-1] == last_id_list[last_chunk_index]:
            results.append(_get_chunk_numpy_full(chunk_engine, chunk_names[-1]))
        else:
            results.append(_get_chunk_numpy_continuous(chunk_engine,
                                                       chunk_names[-1],
                                                       0,
                                                        int(chunk_engine.translate_to_local_index(idxs[-1],
                                                                                                  last_chunk_index))))
    final_results = np.concatenate(results)
    final_results = final_results.reshape(len(final_results), 1)
    return final_results


def get_samples_full(
        chunk_engine,
        all_chunk_names: Optional[List[str]],
        max_workers: int = MAX_WORKERS_FOR_CHUNK_ENGINE,
        strategy: str = "processed"
):
    """Get full samples, fetches chunks in parallel.

    Args:
        chunk_engine: The chunk engine to use.
        all_chunk_names: name of chunks to be loaded.
        max_workers(int): max workers used in thread pool.
        strategy: The strategy to load the chunks.

    Returns:
        List of samples.
    """
    if not all_chunk_names:
        all_chunk_names = [hex(item[0]).split('x')[-1] for item in chunk_engine.chunk_id_encoder.encoded]
    results = []
    if strategy == "processed":
        with ProcessPoolExecutor(min(max_workers, len(all_chunk_names))) as executor:
            for chunk_name in all_chunk_names:
                results.append(executor.submit(_get_chunk_numpy_full, chunk_engine, chunk_name))
    else:
        with ThreadPoolExecutor(min(max_workers, len(all_chunk_names))) as executor:
            for chunk_name in all_chunk_names:
                results.append(executor.submit(_get_chunk_numpy_full, chunk_engine, chunk_name))
    final_results = np.concatenate([result.result() for result in results])
    final_results = final_results.reshape(len(final_results), 1)
    return final_results


def get_samples_batch_random_access(chunk_engine,
                                    index_list: List,
                                    max_workers: int = MAX_WORKERS_FOR_CHUNK_ENGINE,
                                    parallel: Optional[str] = None):
    """【Added by Sherry】
    Get samples for the given index, index must be continuous, fetches chunks in parallel.

    Args:
        chunk_engine: The chunk engine to be used.
        index_list (List): Index applied on the tensor.
        max_workers (int): max workers used in thread pool.
        parallel (Optional[str]): whether using threads for parallel computing or not.

    Returns:
        List of samples.
    """

    load_res = _load_chunk_infos(chunk_engine, list(index_list))  # 注意，这个方法里将index_list重新排序了

    chunk_ids, rows, idxss, _ = zip(*list(load_res))
    chunk_ids, rows, idxss = list(chunk_ids), list(rows), list(idxss)

    samples, chunk_id_local_indexes = _generate_chunk_id_local_indexes(chunk_engine, rows, idxss)

    results = _random_access_return_results(chunk_engine, chunk_ids, parallel, chunk_id_local_indexes, max_workers)

    sorted_results = []
    for idx in index_list:
        sorted_results.append(results[samples.get(idx, None)])  # 这里是按index_list原顺序取回来
    return sorted_results


def _generate_chunk_id_local_indexes(chunk_engine, rows, idxss):
    # Generate all the requests
    samples = {}  # Results takes the form {idx: data}
    # Traverse all the idx, generate the requests and record the place at requests for every idx as where to find
    # data later
    samples_idx = 0
    chunk_id_local_indexes = []
    for row, idxs in zip(rows, idxss):
        idxs = set(idxs)
        local_idxes = []
        for idx in idxs:
            local_idx = chunk_engine.translate_to_local_index(idx, row)
            local_idxes.append(local_idx)
            samples[idx] = samples_idx  # record where to find data later
            samples_idx += 1
        chunk_id_local_indexes.append(local_idxes)
    return samples, chunk_id_local_indexes


def _random_access_process_chunk(tmp_chunk, local_sample_indexes):
    chunk_results = []
    for local_sample_index in local_sample_indexes:
        result = tmp_chunk.read_sample(
            local_sample_index,
            cast=True,
            is_tile=False,
            decompress=True,
        )
        chunk_results.append(result)
    return chunk_results


def _random_access_return_results(chunk_engine, chunk_ids, parallel, chunk_id_local_indexes, max_workers):
    chunks = chunk_engine.get_multiple_chunks_from_chunk_ids(chunk_ids)
    results = []
    if not parallel:
        for i, chunk in enumerate(chunks):
            results.extend(_random_access_process_chunk(chunk, chunk_id_local_indexes[i]))
    else:  # parallel == "threaded"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            single_results = list(executor.map(_random_access_process_chunk, chunks, chunk_id_local_indexes))
        for res in single_results:
            results.extend(res)
    return results


def get_samples(
        chunk_engine,
        index: Index,
        aslist: bool,
        as_arrow: bool = False,
        max_workers: int = MAX_WORKERS_FOR_CHUNK_ENGINE):
    """Get samples for the given index, fetches chunks in parallel.

    Args:
        chunk_engine: The chunk engine to be used.
        index (Index): Index applied on the tensor.
        aslist (bool): Whether to return a list or numpy array.
        as_arrow (bool): Whether return Apache Arrow format.
        max_workers(int): max workers used in thread pool.

    Returns:
        List of samples.
    """
    # Modified by zhouzhenyu at 20240929 for fixing multithread bugs in LRU_Cache.

    is_polygon = chunk_engine.tensor_meta.htype == "polygon"
    # Do not operate self.cache in self.load_chunks, but only compute chunk_info and let cache operate itself
    # Only load chunk_info
    load_res = _load_chunk_infos(chunk_engine, list(index.values[0].indices(chunk_engine.num_samples)))

    # it seems that res_list just do list(load_res)
    request_args = list(load_res)  # new name
    if len(request_args) == 1:
        samples = _get_samples_from_single_request(chunk_engine,
                                                        request_arg=request_args[0],
                                                        index=index,
                                                        is_polygon=is_polygon,
                                                        aslist=aslist,)

    else:
        chunk_ids, rows, idxss, _ = zip(*request_args)
        chunk_ids, rows, idxss = list(chunk_ids), list(rows), list(idxss)
        samples = _get_samples_from_multiple_chunks(chunk_engine,
                                                         chunk_ids=chunk_ids,
                                                         rows=rows,
                                                         idxss=idxss,
                                                         index=index,
                                                         is_polygon=is_polygon,
                                                         aslist=aslist,
                                                         max_workers=max_workers)

    if as_arrow and chunk_engine.tensor_meta.dtype != "List":
        try:
            result = [samples[idx][0] for idx in index.values[0].indices(chunk_engine.num_samples)]
        except KeyError as e:
            raise ReadSampleFromChunkError from e
    else:
        try:
            result = [samples[idx] for idx in index.values[0].indices(chunk_engine.num_samples)]
        except KeyError as e:
            raise ReadSampleFromChunkError from e
    return result


def get_chunks_for_multi_samples(chunk_engine, global_sample_index_list: list):
    """Get chunks for multiple samples. """
    chunk_key_list = []
    for global_sample_index in global_sample_index_list:
        # First, we obtain the chunk ids (the work in def get_chunks_for_sample())
        chunk_id_list = chunk_engine.chunk_id_encoder[global_sample_index]
        # Then, we obtain the chunk names (the work in def get_chunk_for_chunk_id())
        for chunk_id in chunk_id_list:
            chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
            chunk_key = get_chunk_key(chunk_engine.key, chunk_name)
            if chunk_key not in chunk_key_list:
                chunk_key_list.append(chunk_key)
    # Lastly, we obtain all the chunks in the chunk_key_list
    # 原来用的check_filelist 现在改了 ---------------
    chunk_engine.cache.get_items(keys=set(chunk_key_list))


def _get_full_chunk(chunk_engine, index) -> bool:
    """Reads samples from chunks and returns as a boolean that says whether we need to fetch full chunks
        or only specified subset of it.
    Args:
        index (Index): Represents the samples to read from chunks. See `Index` for more information.
    Returns:
        bool: True/False, whether to fetch a full chunk or only a part of it.
    """
    threshold = 10

    if isinstance(index.values[0].value, slice):
        start = index.values[0].value.start or 0
        stop = index.values[0].value.stop or chunk_engine.num_samples
        step = index.values[0].value.step or 1

        if start < 0:
            start = chunk_engine.num_samples + start

        if stop < 0:
            stop = chunk_engine.num_samples + stop

        numpy_array_length = (stop - start) // step
        return numpy_array_length > threshold
    if isinstance(index.values[0].value, tuple):
        return len(index.values[0].value) > threshold
    return False


def _numpy_from_data_cache(chunk_engine, index, length, aslist, as_arrow=False):
    """Returns as numpy from data cache. """
    samples = []
    enc = chunk_engine.chunk_id_encoder
    for global_sample_index in index.values[0].indices(length):
        if (
                chunk_engine.cached_data is None
                or global_sample_index not in chunk_engine.cache_range
        ):
            _obtain_cache_range(chunk_engine, enc, global_sample_index, as_arrow)

        sample = chunk_engine.cached_data[global_sample_index - chunk_engine.cache_range.start]  # type: ignore

        # need to copy if aslist otherwise user might modify the returned data
        # if not aslist, we already do np.array(samples) while formatting which copies
        if not as_arrow:
            sample = sample.copy() if aslist else sample
            sample = sample[tuple(entry.value for entry in index.values[1:])]
        samples.append(sample)
    return samples


def _obtain_cache_range(chunk_engine, enc, global_sample_index, as_arrow):
    dtype = chunk_engine.tensor_meta.dtype
    chunk_arr = enc.array
    if chunk_engine.is_fixed_shape:
        row, chunk_id = chunk_engine.get_fixed_shape_chunk_id(global_sample_index, chunk_arr,
                                                              np.dtype(dtype).itemsize)
        chunk = chunk_engine.get_chunk_from_chunk_id(chunk_id)
    else:
        row = enc.__getitem__(global_sample_index, True)[0][1]
        chunks = chunk_engine.get_chunks_for_sample(global_sample_index)
        assert len(chunks) == 1
        chunk = chunks[0]

    first_sample = int(0 if row == 0 else chunk_arr[row - 1][1] + 1)
    last_sample = int(chunk_engine.chunk_id_encoder.array[row][1])
    num_samples = last_sample - first_sample + 1
    full_shape = (num_samples,) + tuple(chunk_engine.tensor_meta.max_shape)

    data_bytes = bytearray(chunk.data_bytes)

    if as_arrow:
        chunk_engine.cached_data = np.frombuffer(data_bytes, dtype).tolist()
    else:
        chunk_engine.cached_data = np.frombuffer(data_bytes, dtype).reshape(
            full_shape
        )
    chunk_engine.cache_range = range(first_sample, last_sample + 1)


def _get_chunk_numpy_continuous(chunk_engine, chunk_name, start_idx, end_idx):
    sample_dtype = chunk_engine.tensor_meta.dtype
    sample_size = np.array([], dtype=sample_dtype).itemsize
    chunk_key = get_chunk_key(chunk_engine.key, chunk_name)
    chunk = chunk_engine.get_chunk(chunk_key)
    np_bytes = chunk.data_bytes[start_idx*sample_size: end_idx*sample_size + sample_size]
    final_results = np.frombuffer(np_bytes, dtype=sample_dtype)
    return final_results


def _get_chunk_numpy_full(chunk_engine, chunk_name):
    sample_dtype = chunk_engine.tensor_meta.dtype
    chunk_key = get_chunk_key(chunk_engine.key, chunk_name)
    chunk = chunk_engine.get_chunk(chunk_key)
    np_bytes = chunk.data_bytes
    final_results = np.frombuffer(np_bytes, dtype=sample_dtype)
    return final_results


def _load_chunk_infos(chunk_engine,
                     indices: List[int],
                     reverse: bool = False):
    """This function is only temporarily used by self.get_samples, for fixing multithreading bugs in LRU_Cache.
        Only compute chunk_infos.
        Added by zhouzhenyu 20240929.
    """
    chunk_infos = chunk_engine.get_chunk_infos(indices)
    if reverse:
        chunk_infos = reversed(chunk_infos)
    for chunk_info in chunk_infos:
        yield chunk_info


def _get_samples_from_multiple_chunks(chunk_engine,
                                      chunk_ids: List[int],
                                      rows: List[int],
                                      idxss: List[List[int]],
                                      index: Index,  # All with the same index
                                      is_polygon: bool,  # All the same
                                      aslist: bool,  # All the same
                                      max_workers: int = MAX_WORKERS_FOR_CHUNK_ENGINE,
                                      ):
    """This function is only temporarily used by self.get_samples, for fixing multithreading bugs in LRU_Cache.
        Generate all the read requests' arguments and call read_multiple_basic_samples_from_chunk, catch exceptions,
        and do some processing.
        Added by zhouzhenyu at 20240929.
    """
    samples, chunk_id_local_indexes = _generate_chunk_id_local_indexes(chunk_engine, rows, idxss)

    # Get the reading result for all requests and catch exception
    try:
        data = chunk_engine.read_multiple_basic_samples_from_chunk(chunk_ids, chunk_id_local_indexes, index)
    except GetChunkError as e:
        raise GetChunkError(e.chunk_key, tensor_name=chunk_engine.name) from e
    except ReadSampleFromChunkError as e:
        raise ReadSampleFromChunkError(e.chunk_key, tensor_name=chunk_engine.name) from e
    # Process data
    samples = _get_samples_from_multiple_chunks_process_data(chunk_engine, idxss, samples, data, index,
                                                             aslist, is_polygon)
    return samples  # Not return last_shapes because I find it's actually not used in self.get_samples


def _get_samples_from_multiple_chunks_process_data(chunk_engine, idxss, samples, data, index, aslist, is_polygon):
    for idxs in idxss:
        # Through idxss because the _get_samples function seems to do some checking in idxs's order.
        init_last_shape = None
        last_shape = init_last_shape
        for idx in idxs:
            sample = data[samples[idx]]
            # Process sample
            check_sample_shape(sample.shape, last_shape, chunk_engine.key, index, aslist)
            last_shape = sample.shape
            if is_polygon:
                sample = [p.__array__() for p in sample]
            # Put into results.
            samples[idx] = sample
        check_sample_shape(last_shape, init_last_shape, chunk_engine.key, index, aslist)
    return samples


def _get_samples_from_single_request(chunk_engine,
                                     request_arg: Tuple[Any, Any, Any, Any],
                                     index: Index,
                                     is_polygon: bool,
                                     aslist: bool,):
    """仅仅被_get_samples_from_load_res函数调用，仅仅分解一下逻辑，仅仅用来过流水线，没有任何其它的意义。"""
    samples = {}
    chunk_id, row, idxs, _ = request_arg
    samples.update(_get_samples_parallel(chunk_engine, chunk_id=chunk_id,
                                              row=row,
                                              idxs=idxs,
                                              index=index,
                                              is_polygon=is_polygon,
                                              aslist=aslist,
                                              last_shape=None))
    return samples


def _get_samples_parallel(chunk_engine, chunk_id, row, idxs, index, is_polygon, aslist, last_shape):
    read_samples = partial(
        _get_samples,
        chunk_engine,
        index=index,
        is_polygon=is_polygon,
        aslist=aslist,
    )

    chunk_samples, chunk_last_shape = read_samples(chunk_id, row, idxs)
    check_sample_shape(chunk_last_shape, last_shape, chunk_engine.key, index, aslist)
    return chunk_samples


def _get_samples(
        chunk_engine,
        chunk_id: int,
        row: int,
        idxs: List[int],
        index: Index,
        is_polygon: bool,
        aslist: bool,
):
    """Get samples from a chunk.

    Args:
        chunk_id (int): Chunk to read samples from. Can be ``None`` in case of tiles.
        row (int): Row of the chunk in the chunk_id_encoder.
        idxs (List[int]): List of global sample indices to read from this chunk.
        index (Index): Original index applied on the tensor.
        is_polygon (bool): Whether the tensor is a polygon tensor.
        aslist (bool): Whether to return a list or numpy array.

    Raises:
        GetChunkError: If a chunk cannot be retrieved from the storage.
        ReadSampleFromChunkError: If a sample cannot be read from a chunk.

    Returns:
        Dict of samples and shape of the last sample encountered in this chunk.
    """
    samples = {}
    last_shape = None

    for idx in idxs:
        if idx in samples:
            continue
        try:
            if not chunk_engine.is_tiled_sample(idx) and idx < chunk_engine.num_samples:
                local_idx = chunk_engine.translate_to_local_index(idx, row)
                sample = chunk_engine.read_basic_sample_from_chunk(
                    chunk_id, local_idx, index
                )
            else:
                sample = chunk_engine.get_single_sample(idx, index)
        except GetChunkError as e:
            raise GetChunkError(e.chunk_key, idx, chunk_engine.name) from e
        except ReadSampleFromChunkError as e:
            raise ReadSampleFromChunkError(e.chunk_key, idx, chunk_engine.name) from e
        check_sample_shape(sample.shape, last_shape, chunk_engine.key, index, aslist)
        last_shape = sample.shape
        if is_polygon:
            sample = [p.__array__() for p in sample]
        samples[idx] = sample
    return samples, last_shape


def check_sample_shape(shape, last_shape, key, index, aslist):
    """Check the sample shape. """
    if not aslist and last_shape is not None and shape != last_shape:
        raise DynamicTensorNumpyError(key, index, "shape")
