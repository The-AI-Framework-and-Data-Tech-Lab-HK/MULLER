# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/keys.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from muller.constants import (CHUNKS_FOLDER,
                             COMMIT_INFO_FILENAME,
                             DATASET_DIFF_FILENAME,
                             DATASET_LOCK_FILENAME,
                             ENCODED_CREDS_FOLDER,
                             UNSHARDED_ENCODER_FILENAME,
                             ENCODED_CHUNK_NAMES_FOLDER,
                             ENCODED_SEQUENCE_NAMES_FOLDER,
                             ENCODED_TILE_NAMES_FOLDER,
                             FIRST_COMMIT_ID,
                             DATASET_META_FILENAME,
                             TENSOR_META_FILENAME,
                             TENSOR_COMMIT_CHUNK_MAP_FILENAME,
                             TENSOR_COMMIT_DIFF_FILENAME,
                             VERSION_CONTROL_INFO_FILENAME,
                             VERSION_CONTROL_INFO_LOCK_FILENAME,
                             QUERIES_FILENAME,
                             QUERIES_LOCK_FILENAME, CREATE_TENSOR_HIDDEN_UUID, DATASET_UUID_NAME)
from muller.util.exceptions import RomaGetError


def dataset_exists(storage, commit_id=None) -> bool:
    """
    Returns true if a dataset exists at the given location.
    NOTE: This does not verify if it is a VALID dataset, only that it exists and is likely a muller directory.
    """
    try:
        return (
                get_dataset_meta_key(commit_id or FIRST_COMMIT_ID) in storage
                or get_version_control_info_key() in storage or
                get_dataset_diff_key(commit_id or FIRST_COMMIT_ID) in storage
        )
    except (KeyError, RomaGetError):
        return False


def get_tensor_meta_key(key: str, commit_id: str) -> str:
    """tensor/tensor_meta.json -> tensor_meta.json"""
    if commit_id == FIRST_COMMIT_ID:
        return TENSOR_META_FILENAME if key == "" else "/".join((key, TENSOR_META_FILENAME))
    return "/".join(("versions", commit_id, TENSOR_META_FILENAME)) if key == "" else "/".join(
        ("versions", commit_id, key, TENSOR_META_FILENAME))


def tensor_exists(key: str, storage, commit_id: str, split_tensor_meta: bool) -> bool:
    """Determines whether a tensor exists."""
    if split_tensor_meta:
        try:
            storage[get_tensor_meta_key(key, commit_id)]
            return True
        except KeyError:
            return False
    else:
        return key in storage.get_muller_object(get_tensor_meta_key("", commit_id),
                                              dict)  # check tensor key exists in tensor_meta.json


def get_chunk_id_encoder_key(key: str, commit_id: str) -> str:
    """Obtain a chunk id encoder key."""
    if commit_id == FIRST_COMMIT_ID:
        return "/".join(
            (
                key,
                ENCODED_CHUNK_NAMES_FOLDER,
                UNSHARDED_ENCODER_FILENAME,
            )
        )
    return "/".join(
        (
            "versions",
            commit_id,
            key,
            ENCODED_CHUNK_NAMES_FOLDER,
            UNSHARDED_ENCODER_FILENAME,
        )
    )


def get_chunk_key(key: str, chunk_name: str) -> str:
    """Obtain a chunk key."""
    return "/".join((key, CHUNKS_FOLDER, f"{chunk_name}"))


def filter_name(name: str) -> str:
    """Filters tensor name and returns full name of the tensor"""
    name = name.strip("/")
    while "//" in name:
        name = name.replace("//", "/")
    return name


def get_tensor_commit_chunk_map_key(key: str, commit_id: str) -> str:
    """Get a tensor commit chunk map."""
    if commit_id == FIRST_COMMIT_ID:
        return "/".join((key, TENSOR_COMMIT_CHUNK_MAP_FILENAME))
    return "/".join(("versions", commit_id, key, TENSOR_COMMIT_CHUNK_MAP_FILENAME))


def get_sample_shape_tensor_key(key: str):
    """Return sample shape tensor key."""
    return f"_{key}_shape"


def get_sample_id_tensor_key(key: str):
    """Return sample id tensor key."""
    if not CREATE_TENSOR_HIDDEN_UUID:
        return DATASET_UUID_NAME
    return f"_{key}_id"


def get_sample_info_tensor_key(key: str):
    """Return sample info tensor key."""
    return f"_{key}_info"


def get_version_control_info_key() -> str:
    """Return version control info key."""
    return VERSION_CONTROL_INFO_FILENAME


def get_version_control_info_lock_key() -> str:
    """Return version control info lock key."""
    return VERSION_CONTROL_INFO_LOCK_FILENAME


def get_commit_info_key(commit_id: str) -> str:
    """Return commit info key."""
    if commit_id == FIRST_COMMIT_ID:
        return COMMIT_INFO_FILENAME
    return "/".join(("versions", commit_id, COMMIT_INFO_FILENAME))


def get_dataset_meta_key(commit_id: str) -> str:
    """Return dataset meta key."""
    # dataset meta is always relative to the `StorageProvider`'s root
    if commit_id == FIRST_COMMIT_ID:
        return DATASET_META_FILENAME

    return "/".join(("versions", commit_id, DATASET_META_FILENAME))


def get_tensor_tile_encoder_key(key: str, commit_id: str) -> str:
    """Return tensor tile encoder key."""
    if commit_id == FIRST_COMMIT_ID:
        return "/".join((key, ENCODED_TILE_NAMES_FOLDER, UNSHARDED_ENCODER_FILENAME))
    return "/".join(
        (
            "versions",
            commit_id,
            key,
            ENCODED_TILE_NAMES_FOLDER,
            UNSHARDED_ENCODER_FILENAME,
        )
    )


def get_sequence_encoder_key(key: str, commit_id: str) -> str:
    """Return sequence encoder key."""
    if commit_id == FIRST_COMMIT_ID:
        return "/".join(
            (
                key,
                ENCODED_SEQUENCE_NAMES_FOLDER,
                UNSHARDED_ENCODER_FILENAME,
            )
        )
    return "/".join(
        (
            "versions",
            commit_id,
            key,
            ENCODED_SEQUENCE_NAMES_FOLDER,
            UNSHARDED_ENCODER_FILENAME,
        )
    )


def get_creds_encoder_key(key: str, commit_id: str) -> str:
    """Return creds encoder key."""
    if commit_id == FIRST_COMMIT_ID:
        return "/".join((key, ENCODED_CREDS_FOLDER, UNSHARDED_ENCODER_FILENAME))
    return "/".join(
        (
            "versions",
            commit_id,
            key,
            ENCODED_CREDS_FOLDER,
            UNSHARDED_ENCODER_FILENAME,
        )
    )


def get_dataset_lock_key() -> str:
    """Get dataset lock key."""
    return DATASET_LOCK_FILENAME


def get_queries_key() -> str:
    """Get queries key."""
    return QUERIES_FILENAME


def get_dataset_diff_key(commit_id: str) -> str:
    """Get dataset diff key."""
    if commit_id == FIRST_COMMIT_ID:
        return DATASET_DIFF_FILENAME
    return "/".join(("versions", commit_id, DATASET_DIFF_FILENAME))


def get_tensor_commit_diff_key(key: str, commit_id: str) -> str:
    """Get tensor commit diff key."""
    if commit_id == FIRST_COMMIT_ID:
        return "/".join((key, "commit_diff"))
    return "/".join(("versions", commit_id, key, TENSOR_COMMIT_DIFF_FILENAME))


def get_queries_lock_key() -> str:
    """Get queries lock key."""
    return QUERIES_LOCK_FILENAME


def get_downsampled_tensor_key(key: str, factor: int):
    """Get downsampled tensor key."""
    if key.startswith("_") and "downsampled" in key:
        current_factor = int(key.split("_")[-1])
        factor *= current_factor
        ls = key.split("_")
        ls[-1] = str(factor)
        final_key = "_".join(ls)
    else:
        final_key = f"_{key}_downsampled_{factor}"
    return final_key
