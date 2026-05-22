# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import os
import sys

from pathlib import Path
from .constants import LOCAL_DATA_DIR, HUASHAN_DATA_DIR

DATA_DIR = Path(LOCAL_DATA_DIR)
DATA_DIR /= "cifar10"
DATA_DIR /= "shared"

DATA_DIR_HS = Path(HUASHAN_DATA_DIR)
DATA_DIR_HS /= "cifar10"
DATA_DIR_HS /= "shared"

PERSONAL_PREFIX = None

# Class Names for Filtering
# "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
# See https://www.cs.toronto.edu/~kriz/cifar.html
LABELS_DICT = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}

def official_path(storage="local", data_path=""):
    prefix = ""
    if storage == "huawei-obs":
        prefix = "huawei-obs://"
    elif storage == "s3":
        prefix = "s3://"
    return prefix + data_path


def official_creds(storage="local"):
    creds = None

    # You may write your own cred details here.
    if storage == "s3":
        endpoint = "http://xx:xxxx"
        ak = "xxx"
        sk = "xxx"
        bucket_name = "xxx"
        creds = {"bucket_name": bucket_name, "endpoint": endpoint, "ak": ak, "sk": sk}

    return creds


def verify_storage(storage):
    return False


def get_size(mode="train"):
    if mode == "test":
        return 1000
    else:
        N = 50000
        VAL_SIZE = int(N * 0)
        if mode == "train":
            return N - VAL_SIZE
        else:
            return VAL_SIZE


class Cifar10Dataset:
    """Face Landmarks dataset."""

    def __init__(self, root_dir, mode, train=True, download=False):
        self.root_dir = Path(root_dir) / mode
        self.mode = mode
        self.is_train = train

        if download and not self.root_dir.is_dir():
            self._download()
        else:
            print("Cifar10 dataset already exists. Skipping")

    def _len(self):
        if self.is_train:
            return get_size("train") + get_size("val")
        else:
            return get_size("test")

    def _download(self):
        from torchvision.datasets import CIFAR10

        original_path = self.root_dir.parent / f"original_{'train' if self.is_train else 'test'}"
        ds = CIFAR10(original_path, train=self.is_train, download=True)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(ds):
            if i % 100 == 0:
                print(f"{i} / {self._len()}", end="\r", flush=True)

            img, cls = sample
            cls = str(cls)
            path = self.root_dir / f"{i}.jpeg"
            img.save(path)
            with open(path.with_suffix(".txt"), "w") as fh:
                fh.write(cls)

    def __len__(self):
        return self._len()

    def __getitem__(self, idx):

        path = self.root_dir / f"{idx}.jpeg"
        with open(path.with_suffix(".txt"), "r") as fh:
            cls = int(fh.read())

        return path, cls

    def __iter__(self):
        for i in range(get_size(self.mode)):
            yield self.__getitem__(i)


def get_cifar10(mode="train", download=True):
    import torch

    is_train = mode in ["train", "val"]  # TODO: why?
    path = DATA_DIR
    if is_train:
        path /= "train"
    else:
        path /= "test"

    dataset = Cifar10Dataset(DATA_DIR, mode, train=is_train, download=download)

    gen = torch.Generator()
    gen.manual_seed(0)

    if is_train:
        ds_train, ds_val = torch.utils.data.random_split(
            dataset, [get_size("train"), get_size("val")], generator=gen
        )
        if mode == "train":
            return ds_train

        if mode == "val":
            return ds_val
    else:
        return dataset


def get_cifar10_huashan(mode="train"):
    root_dir = DATA_DIR_HS / mode
    if not root_dir.is_dir():
        raise Exception("Expected a directory of ciar10 data, but not found, please uoload.")
    files_list = sorted(os.listdir(root_dir))
    pic_files = [fn for fn in files_list if fn.endswith("jpeg")]
    files_list = [Path(os.path.join(root_dir, fn)) for fn in pic_files]
    return files_list


def check_skip_time_consuming_test():
    """Check whether to skip the time-consuming test case."""
    skip = True
    for arg in sys.argv:
        if arg == "--test_time_consuming":
            skip = False
            break
    return skip


def check_skip_vector_index_test():
    """Check whether to skip the time-consuming test case."""
    skip = True
    print(f"sys.argv: {sys.argv}")
    for arg in sys.argv:
        if arg == "--vector_index_test":
            skip = False
            break
    return skip
