# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/class_label.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

import hashlib
from typing import List

import numpy as np


def convert_to_idx(samples, class_names: List[str]):
    """Converts a list of samples to an array of indices."""
    class_idx = {class_names[i]: i for i in range(len(class_names))}

    def convert(samples):
        idxs = []
        additions = []
        for sample in samples:
            if isinstance(sample, np.ndarray):
                sample = sample.tolist()
            if isinstance(sample, str):
                idx = class_idx.get(sample)
                if idx is None:
                    idx = len(class_idx)
                    class_idx[sample] = idx
                    additions.append((sample, idx))
                idxs.append(idx)
            elif isinstance(sample, list):
                idxs_, additions_ = convert(sample)
                idxs.append(idxs_)
                additions.extend(additions_)
            else:
                idxs.append(sample)
        return idxs, additions

    return convert(samples)


def convert_to_hash(samples, hash_label_map):
    """Converts a list of samples to a hash map."""
    if isinstance(samples, np.ndarray):
        samples = samples.tolist()
    if isinstance(samples, list):
        return [convert_to_hash(sample, hash_label_map) for sample in samples]

    if isinstance(samples, str):
        hash_ = _hash_str_to_int32(samples)
        hash_label_map[hash_] = samples
    else:
        hash_ = samples
    return hash_


def convert_to_text(inp, class_names: List[str], return_original=False):
    """Converts a list of samples to a string."""
    if isinstance(inp, np.integer):
        idx = int(inp)
        if idx < len(class_names):
            return class_names[idx]
        return idx if return_original else None
    return [convert_to_text(item, class_names) for item in inp]


def convert_hash_to_idx(hashes, hash_idx_map):
    """Converts a hash map to a list of indices."""
    if isinstance(hashes, list):
        return [convert_hash_to_idx(hash, hash_idx_map) for hash in hashes]
    try:
        return hash_idx_map[hashes]
    except KeyError:
        return hashes


def _hash_str_to_int32(string: str):
    hash_ = int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) >> 224
    return hash_
