# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/core/dataset/dataset.py
#
# Modifications Copyright (c) 2026 Xueling Lin

import pathlib
import posixpath
import uuid
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

import muller
from muller.constants import (SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE, CREATE_TENSOR_HIDDEN_UUID, DATASET_UUID_NAME)
from muller.core.dataset import Dataset
from muller.core.fast_forwarding import ffw_dataset_meta
from muller.core.lock import unlock_dataset
from muller.core.meta.dataset_meta import DatasetMeta
from muller.core.storage.local import LocalProvider
from muller.core.tensor import Tensor
from muller.htype import (UNSPECIFIED,
                         HTYPE_CONFIGURATIONS,
                         verify_htype_key_value)
from muller.util.exceptions import (RenameError,
                                   RenameStorageError,
                                   TensorDoesNotExistError,
                                   TensorAlreadyExistsError,
                                   TensorTooLargeToDelete,
                                   InvalidTensorNameError,
                                   SampleAppendingError,
                                   SampleAppendError,
                                   SampleExtendError,
                                   DatasetTooLargeToDelete)
from muller.util.muller_keywords import is_muller_keyword
from muller.util.htype import parse_complex_htype
from muller.util.keys import (filter_name,
                             get_sample_shape_tensor_key,
                             get_sample_id_tensor_key,
                             get_sample_info_tensor_key,
                             tensor_exists)
from muller.util.keys import get_downsampled_tensor_key
from muller.util.path import convert_pathlib_to_string_if_needed
from muller.util.version_control import (auto_checkout)

_LOCKABLE_STORAGES = {LocalProvider}


def create_tensor(
        ds,
        name: str,
        htype: str = UNSPECIFIED,
        dtype: Union[str, np.dtype] = UNSPECIFIED,
        sample_compression: Union[str, None] = UNSPECIFIED,
        chunk_compression: str = UNSPECIFIED,
        hidden: bool = False,
        **kwargs,
):
    """
    Create tensors.
    """

    return _create_tensor(
        ds,
        name,
        htype,
        dtype,
        sample_compression,
        chunk_compression,
        hidden,
        **kwargs)


def create_tensor_like(
        ds, name: str, source: "Tensor",
) -> "Tensor":
    """Copies the ``source`` tensor's meta information and creates a new tensor with it. No samples are copied,
    only the meta/info for the tensor is.

    Examples:
        >>> ds.create_tensor_like("cats", ds["images"])

    Args:
        ds (Dataset): The target dataset.
        name (str): Name for the new tensor.
        source (Tensor): Tensor who's meta/info will be copied. May or may not be contained in the same dataset.

    Returns:
        Tensor: New Tensor object.
    """

    info = source.info.__getstate__().copy()
    meta = source.meta.__getstate__().copy()

    del meta["min_shape"]
    del meta["max_shape"]
    del meta["length"]
    del meta["version"]
    del meta["name"]
    meta["dtype"] = np.dtype(meta["typestr"]) if meta["typestr"] else meta["dtype"]

    destination_tensor = _create_tensor(
        ds=ds,
        name=name,
        create_id_tensor=f"_{name}_id" in source.dataset.get_tensors(),
        create_shape_tensor=bool(source.get_sample_shape_tensor()),
        create_sample_info_tensor=bool(source.get_sample_info_tensor()),
        **meta,
    )
    destination_tensor.info.update(info)
    return destination_tensor


def delete_tensor(ds, name: str, large_ok: bool = False):
    """Delete a tensor."""
    return _delete_tensor(ds, name, large_ok)


def extend(
        ds,
        samples: Dict[str, Any],
        skip_ok: bool = False,
        append_empty: bool = False,
        ignore_errors: bool = False,
        progressbar: bool = False,
):
    """ Extend samples to the dataset. """
    extend_flag = False
    if isinstance(samples, Dataset):
        samples = samples.tensors
        extend_flag = True
    elif set(map(type, samples.values())) == {np.ndarray}:
        extend_flag = True
    if not samples:
        return
    n = len(samples[next(iter(samples.keys()))])
    for v in samples.values():
        if len(v) != n:
            sizes = {k: len(v) for (k, v) in samples.items()}
            raise ValueError(
                f"Incoming samples are not of equal lengths. Incoming sample sizes: {sizes}"
            )
    if extend_flag:
        if ignore_errors:
            warnings.warn(
                "`ignore_errors` argument will be ignored while extending with numpy arrays or tensors."
            )
        _append_or_extend(
            ds,
            samples,
            extend_flag=True,
            skip_ok=skip_ok,
            append_empty=append_empty,
        )
    else:
        with ds:
            if progressbar:
                indices = tqdm(range(n))
            else:
                indices = range(n)
            for i in indices:
                try:
                    _append_or_extend(
                        ds,
                        {k: v[i] for k, v in samples.items()},
                        extend_flag=False,
                        skip_ok=skip_ok,
                        append_empty=append_empty,
                    )
                except Exception as e:
                    if ignore_errors:
                        continue

                    if isinstance(e, SampleAppendError):
                        raise SampleExtendError(str(e)) from e.__cause__
                    raise e


def append(
        ds,
        sample: Dict[str, Any],
        skip_ok: bool = False,
        append_empty: bool = False,
):
    """Append samples to the dataset."""
    _append_or_extend(
        ds,
        sample,
        extend_flag=False,
        skip_ok=skip_ok,
        append_empty=append_empty,
    )


def update(ds, sample: Dict[str, Any]):
    """Update samples in the dataset."""
    if len(ds.index) > 1:
        raise ValueError(
            "Cannot make partial updates to samples using `ds.update`. Use `ds.tensor[index] = value` instead."
        )

    with ds:
        saved = defaultdict(list)
        try:
            for k, v in sample.items():
                tensor_meta = ds[k].meta
                sample_compression = tensor_meta.sample_compression
                chunk_compression = tensor_meta.chunk_compression
                engine = ds[k].chunk_engine

                for idx in ds.index.values[0].indices(ds[k].num_samples):
                    if tensor_meta.is_sequence:
                        old_sample = []
                        for i in range(*engine.sequence_encoder[idx]):
                            item = _get_sample_from_engine(
                                ds,
                                engine,
                                i,
                                sample_compression or chunk_compression,
                                tensor_meta.dtype,
                                chunk_compression is not None or engine.is_text_like,
                            )
                            old_sample.append(item)
                    else:
                        old_sample = _get_sample_from_engine(
                            ds,
                            engine,
                            idx,
                            sample_compression or chunk_compression,
                            tensor_meta.dtype,
                            chunk_compression is not None or engine.is_text_like,
                        )

                    saved[k].append(old_sample)
                ds[k] = v

        except Exception as e:
            for k, v in saved.items():
                # squeeze
                if len(v) == 1:
                    v = v[0]
                try:
                    ds[k] = v
                except Exception as e2:
                    raise Exception(
                        "Error while attempting to rollback updates"
                    ) from e2
            raise e


def pop(ds, index: Optional[Union[List, int]] = None, rechunk: bool = False):
    """Pop samples in the dataset."""
    if index is None:
        temp = [ds.max_len - 1]
        index = temp

    if not isinstance(index, list):
        temp = [index]
        index = temp

    if not index:
        return

    if len(set(index)) != len(index):
        raise ValueError("Duplicate indices are not allowed.")

    max_len = ds.max_len
    if max_len == 0:
        raise IndexError("Can't pop from empty dataset.")

    for idx in index:
        if idx < 0:
            raise IndexError("Pop doesn't support negative indices.")
        if idx >= max_len:
            raise IndexError(
                f"Index {idx} is out of range. The longest tensor has {max_len} samples."
            )

    index = sorted(index, reverse=True)
    _pop(ds, index, rechunk)

    ds.append_only = False


def delete(ds, large_ok=False):
    """Delete the dataset."""
    if ds.view_entry is not None:
        ds.view_entry.delete()
        return

    if ds.vds is not None:
        ds.vds.delete(large_ok=large_ok)

    if not large_ok:
        size = ds.size_approx()
        if size > muller.constants.DELETE_SAFETY_SIZE:
            raise DatasetTooLargeToDelete(ds.path)

    unlock_dataset(ds)
    ds.storage.clear()


def rename(ds, path: Union[str, pathlib.Path]):
    """Renames the dataset to `path`.

    Example:

        >>> ds = muller.load("path_to_dataset")
        >>> ds.rename("path_to_renamed_dataset")

    Args:
        path (str, pathlib.Path): New path to the dataset.

    Raises:
        RenameError: If ``path`` points to a different directory.
    """
    # Note: currently we only accept the rename operation in LocalProvider and MemProvider
    if not isinstance(ds.base_storage, LocalProvider):
        raise RenameStorageError()

    new_path = convert_pathlib_to_string_if_needed(path)
    old_path = convert_pathlib_to_string_if_needed(ds.path)

    def formal_path(test_path):
        if test_path[-1] == "/":
            return test_path[:-1]
        return test_path

    new_path = formal_path(new_path)
    old_path = formal_path(old_path)

    if posixpath.split(new_path)[0] != posixpath.split(old_path)[0]:
        raise RenameError()

    ds.base_storage.rename(new_path)
    ds.path = new_path


def handle_rename_tensor(ds, name, new_name):
    """Function to handle rename tensor"""
    tensor = ds[name]
    tensor.meta.name = new_name
    tensor.meta.is_dirty = True
    tensor_names = ds.version_state.get("tensor_names", "")
    if not tensor_names:
        raise ValueError
    key = tensor_names.pop(name)
    meta = ds.meta
    if key not in meta.hidden_tensors:
        tensor_diff = tensor.chunk_engine.commit_diff
        # if tensor was created in this commit, tensor name has to be updated without adding it to diff.
        if not tensor_diff.created:
            ds.get_dataset_diff.tensor_renamed(name, new_name)
    tensor_names[new_name] = key
    ffw_dataset_meta(meta)
    meta.rename_tensor(name, new_name)

    return tensor


def rename_tensor(ds, name: str, new_name: str):
    """Renames tensor with name ``name`` to ``new_name``

    Args:
        ds (Dataset): Dataset to rename.
        name (str): Name of tensor to be renamed.
        new_name (str): New name of tensor.

    Raises:
        TensorDoesNotExistError: If tensor of name ``name`` does not exist in the dataset.
        TensorAlreadyExistsError: Duplicate tensors are not allowed.
        InvalidTensorNameError: If ``new_name`` is in dataset attributes.
        RenameError: If ``new_name`` is `_uuid`.
    """

    auto_checkout(ds)

    if name not in ds.get_tensors():
        raise TensorDoesNotExistError(name)

    name = filter_name(name)
    new_name = filter_name(new_name)

    if not CREATE_TENSOR_HIDDEN_UUID and name == "_uuid":
        raise RenameError("Cannot rename _uuid tensor.")

    if new_name in ds.version_state.get("tensor_names", ""):
        raise TensorAlreadyExistsError(new_name)

    new_tensor_name = posixpath.split(new_name)[1]
    if not new_tensor_name or new_tensor_name in dir(ds):
        raise InvalidTensorNameError(new_name)

    tensor = ds.handle_rename_tensor(name, new_name)

    ds.storage.maybe_flush()
    return tensor


def create_uuid_tensor(ds):
    """Create uuid tensor column. """
    return _create_tensor(
        ds=ds,
        name=DATASET_UUID_NAME,
        hidden=True,
        create_id_tensor=False,
        create_sample_info_tensor=False,
        create_shape_tensor=False,
        split_tensor_meta=ds.split_tensor_meta,
    )


def _create_tensor(
        ds,
        name: str,
        htype: Union[str, None] = UNSPECIFIED,
        dtype: Union[str, np.dtype] = UNSPECIFIED,
        sample_compression: Union[str, None] = UNSPECIFIED,
        chunk_compression: Union[str, None] = UNSPECIFIED,
        hidden: bool = False,
        **kwargs):

    exist_ok = kwargs.pop("exist_ok", False)
    hidden_kwargs = {"create_sample_info_tensor": kwargs.pop("create_sample_info_tensor", False),
                     "create_shape_tensor": kwargs.pop("create_shape_tensor", False),
                     "create_id_tensor": kwargs.pop("create_id_tensor", False),
                     "downsampling": kwargs.pop("downsampling", None),
                     "hidden": hidden}

    # Validate
    auto_checkout(ds)
    name = filter_name(name)  # tensor name
    key = ds.version_state["tensor_names"].get(name)
    is_sequence, _, htype = parse_complex_htype(htype)
    if key:
        return _validate_existing_tensor(name, key, exist_ok, ds, htype, dtype, sample_compression, chunk_compression,
                                         hidden, is_sequence)
    if name in ds.version_state["full_tensors"]:
        key = f"{name}_{uuid.uuid4().hex[:4]}"
    else:
        key = name
    if not name or name in dir(ds) or is_muller_keyword(name):
        raise InvalidTensorNameError(name)

    kwargs["is_sequence"] = kwargs.get("is_sequence") or is_sequence
    kwargs["split_tensor_meta"] = ds.split_tensor_meta
    kwargs["verify"] = kwargs.pop("verify", True)

    info_kwargs, meta_kwargs = _get_kwargs(kwargs, htype)

    muller.core.tensor.create_tensor(
        key=key,
        storage=ds.storage,
        htype=htype,
        dtype=dtype,
        sample_compression=sample_compression,
        chunk_compression=chunk_compression,
        version_state=ds.version_state,
        hidden=hidden,
        overwrite=True,
        **meta_kwargs,
    )

    return _manage_creation_of_hidden_tensor(ds,
                                             name,
                                             key,
                                             htype,
                                             dtype,
                                             sample_compression,
                                             chunk_compression,
                                             meta_kwargs,
                                             info_kwargs,
                                             hidden_kwargs
                                             )


def _get_kwargs(kwargs, htype):
    htype_config = HTYPE_CONFIGURATIONS.get(htype, {}).copy()
    info_keys = htype_config.pop("_info", [])
    info_kwargs = {}
    meta_kwargs = {}
    for k, v in kwargs.items():
        if k in info_keys:
            verify_htype_key_value(htype, k, v)
            info_kwargs[k] = v
        else:
            meta_kwargs[k] = v

    # Set defaults
    for k in info_keys:
        if k not in info_kwargs:
            if k == "class_names":
                info_kwargs[k] = htype_config[k].copy()
            else:
                info_kwargs[k] = htype_config[k]
    return info_kwargs, meta_kwargs


def _validate_existing_tensor(name, key, exist_ok, ds, htype, dtype, sample_compression, chunk_compression,
                              hidden, is_sequence):
    if not exist_ok:
        raise TensorAlreadyExistsError(name)
    tensor = ds[key]
    current_config = tensor.config
    new_config = {
        "htype": htype,
        "dtype": dtype,
        "sample_compression": sample_compression,
        "chunk_compression": chunk_compression,
        "hidden": hidden,
        "is_sequence": is_sequence,
    }

    if current_config != new_config:
        raise ValueError(
            f"Tensor {name} already exists with different configuration."
            f"Current config: {current_config}."
            f"New config: {new_config}"
        )
    return tensor


def _manage_creation_of_hidden_tensor(ds,
                                      name,
                                      key,
                                      htype,
                                      dtype,
                                      sample_compression,
                                      chunk_compression,
                                      meta_kwargs,
                                      info_kwargs,
                                      hidden_kwargs
                                      ):

    if not meta_kwargs["verify"]:
        hidden_kwargs["create_shape_tensor"] = False
        hidden_kwargs["create_sample_info_tensor"] = False

    meta: DatasetMeta = ds.meta
    ffw_dataset_meta(meta)
    meta.add_tensor(name, key, hidden=hidden_kwargs.get("hidden", False))
    tensor = Tensor(key, ds)
    tensor.meta.name = name
    ds.version_state["full_tensors"][key] = tensor
    ds.version_state["tensor_names"][name] = key
    if info_kwargs:
        tensor.info.update(info_kwargs)  # chunk_engine._info
    ds.storage.maybe_flush()
    if hidden_kwargs.get("create_sample_info_tensor", False) and htype in (
            "image",
            "audio",
            "video",
            "dicom",
            "point_cloud",
            "mesh",
            "nifti",
    ):
        _create_sample_info_tensor(ds, name)
    if hidden_kwargs.get("create_shape_tensor", False) and htype not in ("text", "json"):
        _create_sample_shape_tensor(ds, name)
    if hidden_kwargs.get("create_id_tensor", False):
        _create_sample_id_tensor(ds, name)  # _id tensor
    if hidden_kwargs.get("downsampling", None):
        if htype not in {"image", "image.rgb", "image.gray", "binary_mask", "segment_mask"}:
            warnings.warn(
                f"Downsampling is only supported for tensor with htypes image, image.rgb, image.gray, binary_mask, "
                f"segment_mask, but got {htype}. "
                f"Skipping downsampling."
            )
        else:
            _create_downsampled_tensor(
                ds,
                name,
                htype,
                dtype,
                sample_compression,
                chunk_compression,
                meta_kwargs,
                hidden_kwargs.get("downsampling", None)
            )
    return tensor


def _create_sample_shape_tensor(ds, tensor: str):
    shape_tensor = get_sample_shape_tensor_key(tensor)
    _ = _create_tensor(
        ds=ds,
        name=shape_tensor,
        dtype="int64",
        hidden=True,
        create_id_tensor=False,
        create_sample_info_tensor=False,
        create_shape_tensor=False,
        max_chunk_size=SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE,
        split_tensor_meta=ds.split_tensor_meta
    )


def _create_sample_id_tensor(ds, tensor: str):
    id_tensor = get_sample_id_tensor_key(tensor)
    _ = _create_tensor(
        ds=ds,
        name=id_tensor,
        hidden=True,
        create_id_tensor=False,
        create_sample_info_tensor=False,
        create_shape_tensor=False,
        split_tensor_meta=ds.split_tensor_meta,
    )


def _create_sample_info_tensor(ds, tensor: str):
    sample_info_tensor = get_sample_info_tensor_key(tensor)
    _ = _create_tensor(
        ds=ds,
        name=sample_info_tensor,
        htype="json",
        max_chunk_size=SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE,
        hidden=True,
        create_id_tensor=False,
        create_sample_info_tensor=False,
        create_shape_tensor=False,
        split_tensor_meta=ds.split_tensor_meta
    )


def _create_downsampled_tensor(
        ds,
        tensor: str,
        htype: str,
        dtype: Union[str, np.dtype],
        sample_compression: str,
        chunk_compression: str,
        meta_kwargs: Dict[str, Any],
        downsampling
):
    downsampling_factor, number_of_layers = _validate_downsampling(downsampling)
    downsampled_tensor = get_downsampled_tensor_key(tensor, downsampling_factor)
    if number_of_layers == 1:
        downsampling = None
    else:
        downsampling = (downsampling_factor, number_of_layers - 1)
    meta_kwargs = meta_kwargs.copy()
    new_tensor = _create_tensor(
        ds=ds,
        name=downsampled_tensor,
        htype=htype,
        dtype=dtype,
        sample_compression=sample_compression,
        chunk_compression=chunk_compression,
        hidden=True,
        create_id_tensor=False,
        create_sample_info_tensor=False,
        create_shape_tensor=False,
        downsampling=downsampling,
        **meta_kwargs,
    )
    new_tensor.info.downsampling_factor = downsampling_factor


def _delete_tensor(ds, name: str, large_ok: bool = False):
    auto_checkout(ds)
    name = filter_name(name)
    key = ds.version_state["tensor_names"].get(name)
    if not key:
        raise TensorDoesNotExistError(name)
    if not tensor_exists(key, ds.storage, ds.version_state["commit_id"], ds.split_tensor_meta):
        raise TensorDoesNotExistError(name)
    if not large_ok:
        chunk_engine = ds.version_state["full_tensors"][key].chunk_engine
        size_approx = chunk_engine.num_samples * chunk_engine.min_chunk_size
        if size_approx > muller.constants.DELETE_SAFETY_SIZE:
            raise TensorTooLargeToDelete(name)
    with ds:
        meta = ds.meta
        key = ds.version_state["tensor_names"].pop(name)
        if key not in meta.hidden_tensors:
            tensor_diff = Tensor(key, ds).chunk_engine.commit_diff
            # if tensor was created in this commit, there's no diff for deleting it.
            if not tensor_diff.created:
                ds.get_dataset_diff.tensor_deleted(name)
        muller.core.tensor.delete_tensor(key, ds)
        ds.version_state["full_tensors"].pop(key)
        ffw_dataset_meta(meta)
        meta.delete_tensor(name)

    hidden_tensor_name = [get_sample_id_tensor_key(name),
                          get_sample_info_tensor_key(name),
                          get_sample_shape_tensor_key(name)]
    for t_name in hidden_tensor_name:
        if t_name == DATASET_UUID_NAME:
            continue
        t_key = ds.meta.tensor_names.get(t_name)
        if t_key and tensor_exists(
                t_key, ds.storage, ds.version_state["commit_id"], ds.split_tensor_meta
        ):
            _delete_tensor(ds, t_name, large_ok=True)

    ds.storage.flush()


def _append_or_extend(
        ds,
        sample: Dict[str, Any],
        extend_flag: bool = False,
        skip_ok: bool = False,
        append_empty: bool = False,
):
    tensors = ds.tensors
    if isinstance(sample, Dataset):
        sample = sample.tensors
    if not isinstance(sample, dict):
        raise SampleAppendingError()

    skipped_tensors = [k for k in tensors if k not in sample]
    if skipped_tensors and not skip_ok and not append_empty:
        raise KeyError(
            f"Required tensors not provided: {skipped_tensors}. "
            f"Pass either `skip_ok=True` to skip tensors or `append_empty=True` to append empty samples "
            f"to unspecified tensors."
        )
    for k in sample:
        if k not in tensors:
            raise TensorDoesNotExistError(k)
    tensors_to_check_length = tensors if append_empty else sample
    if len(set(map(len, (tensors[k] for k in tensors_to_check_length)))) != 1:
        raise ValueError(
            "When appending using Dataset.append or Dataset.extend, "
            "all tensors being updated are expected to have the same length."
        )
    if extend_flag:
        sample_lens = set(map(len, sample.values()))
        if sample_lens == {0}:
            return
        if len(sample_lens) > 1 and not append_empty:
            raise ValueError(
                "All tensors have to be extended to the same length. "
                "Specify `append_empty=True` to pad tensors receiving fewer samples."
            )
    _append_or_extend_dataset(ds, tensors, sample, extend_flag, max_len=max(set(map(len, sample.values()))),
                              skip_ok=skip_ok)


def _append_or_extend_dataset(ds, tensors, sample, extend_flag, max_len, skip_ok):
    tensors_appended = []
    with ds:
        for k in tensors:
            extend_extra_nones = 0
            if k in sample:
                v = sample[k]
                if extend_flag:
                    extend_extra_nones = max(max_len - len(v), 0)
            else:
                if skip_ok:
                    continue
                if extend_flag:
                    v = [None] * max_len
                else:
                    v = None

            _get_tensors_appended(ds, tensors, k, v, extend_flag, extend_extra_nones,
                                                             tensors_appended,)


def _get_tensors_appended(ds, tensors, k, v, extend_flag, extend_extra_nones, tensors_appended,):
    tensor = tensors[k]
    enc = tensor.chunk_engine.chunk_id_encoder
    num_chunks = enc.num_chunks
    num_samples = tensor.meta.length
    try:
        if extend_flag:
            tensor.protected_extend(v)
            if extend_extra_nones:
                tensor.protected_extend([None] * extend_extra_nones)
        else:
            tensor.protected_append(v)
        tensors_appended.append(k)
    except Exception as e:
        if extend_flag:
            raise NotImplementedError(
                "Unable to recover from error while extending multiple tensors with numpy arrays."
            ) from e
        num_chunks_added = enc.num_chunks - num_chunks
        if num_chunks_added > 1:
            # This is unlikely to happen, i.e the sample passed the validation
            # steps and tiling but some error occurred while writing tiles to chunks
            raise NotImplementedError(
                "Unable to recover from error while writing tiles."
            ) from e
        if num_chunks_added == 1:
            enc.encoded = enc.encoded[:-1]
            diff = tensor.meta.length - num_samples
            tensor.meta.update_length(-diff)
        for tensor in tensors_appended:
            try:
                ds[tensor].pop()
            except Exception as e2:
                raise Exception(
                    "Error while attempting to rollback appends"
                ) from e2
        raise e


def _get_sample_from_engine(ds, engine, idx, compression, dtype, decompress):
    decompress = decompress or engine.is_tiled_sample(idx)
    item = engine.get_single_sample(idx, ds.index, decompress=decompress)
    shape = engine.read_shape_for_sample(idx)
    return engine.get_sample_object(item, shape, compression, dtype, decompress)


def _pop(ds, index: List[int], rechunk: bool = False):
    """Removes elements at the given indices."""
    with ds:
        for tensor in ds.tensors.values():
            tensor.protected_pop(index, rechunk)
        _pop_uuid(ds, index)
        ds.storage.flush()


def _pop_uuid(ds, index):
    if not ds.use_dataset_uuid:
        return
    ds[DATASET_UUID_NAME].protected_pop(index)


def _validate_downsampling(downsampling):
    if downsampling is None:
        return None, None
    if len(downsampling) != 2:
        raise ValueError(
            f"Downsampling must be a tuple of the form (downsampling_factor, number_of_layers), got {downsampling}"
        )
    downsampling_factor, number_of_layers = downsampling
    if downsampling_factor < 1 or not isinstance(downsampling_factor, int):
        raise ValueError("Downsampling factor must be an integer >= 1")
    if number_of_layers < 1 or not isinstance(number_of_layers, int):
        raise ValueError("Number of layers must be an integer >= 1")

    return downsampling_factor, number_of_layers
