# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import logging
import numpy as np
import pytest

import muller
from tests.utils import official_path, official_creds, get_cifar10_huashan
from tests.constants import CIFAR10_TEST_PATH


def test_filter(storage):
    """
    Tests filter function with lambda expressions.
    """
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    ds_cifar = muller.dataset(path=official_path(storage, CIFAR10_TEST_PATH),
                             creds=official_creds(storage), overwrite=True)
    with ds_cifar:
        # Create the tensors with names of your choice.
        ds_cifar.create_tensor("images", htype="image", sample_compression="jpeg")
        ds_cifar.create_tensor("labels", htype="class_label", class_names=class_names)

    pics_list = get_cifar10_huashan(mode="train")
    for path in pics_list:
        with open(path.with_suffix(".txt"), "r") as fh:
            cls = int(fh.read())
        ds_cifar.images.append(muller.read(path))
        ds_cifar.labels.append(np.uint8(cls))

    ds_cifar.summary()

    # UDF as condition
    ds_cifar_1 = ds_cifar.filter(lambda sample: sample.labels.data()['value'] == 1)
    assert len(ds_cifar_1) == 2

    ds_3 = ds_cifar.filter(lambda sample: sample.labels.data()["value"] == 1 or sample.labels.data()["value"] == 8)
    assert len(ds_3) == 3


def test_filter_udf(storage):
    """
    Tests filter function with UDF and strings.
    """
    ds_cifar = muller.load(path=official_path(storage, CIFAR10_TEST_PATH), creds=official_creds(storage))

    # UDF with decorator as condition, and num_workers > 0
    @muller.compute
    def filter_labels(sample_in):
        return sample_in.labels.data()['value'] == 1

    ds_cifar_1_2 = ds_cifar.filter(filter_labels(), num_workers=2)
    assert len(ds_cifar_1_2) == 2

    # String as condition
    ds_cifar_2 = ds_cifar.filter("labels == 1")
    assert len(ds_cifar_2) == 2

    ds_cifar_4 = ds_cifar.filter("labels == 1 or labels == 8")
    assert len(ds_cifar_4) == 3

    ds_cifar_5 = ds_cifar.filter("1 in labels")
    assert len(ds_cifar_5) == 2

    ds_cifar_6 = ds_cifar.filter("1 not in labels")
    assert len(ds_cifar_6) == 8

    # filter support return index_map
    ds_cifar_7 = ds_cifar.filter("1 in labels")
    assert len(ds_cifar_7) == 2

    ds_cifar_8 = ds_cifar.filter(filter_labels(), num_workers=2)
    assert len(ds_cifar_8) == 2


def test_filter_with_offset_and_limit(storage):
    """
    Tests filter function with offset and limit.
    """
    # UDF with decorator as condition, and num_workers > 0
    @muller.compute
    def filter_labels(sample_in):
        return sample_in.labels.data()['value'] == 1

    ds_cifar = muller.load(path=official_path(storage, CIFAR10_TEST_PATH), creds=official_creds(storage))
    ds_cifar_9 = ds_cifar.filter(lambda sample: sample.labels.data()['value'] == 1)
    assert len(ds_cifar_9) == 2

    ds_cifar_10 = ds_cifar.filter(lambda sample: sample.labels.data()['value'] == 1, offset=1)
    assert len(ds_cifar_10.filtered_index) == 2

    ds_cifar_11 = ds_cifar.filter(lambda sample: sample.labels.data()['value'] == 1, limit=5)
    assert len(ds_cifar_11.filtered_index) == 2
    assert ds_cifar_11.filtered_index[-1] == 5

    ds_cifar_12 = ds_cifar.filter(lambda sample: sample.labels.data()['value'] >= 2, offset=5, limit=2)
    assert len(ds_cifar_12.filtered_index) == 2
    assert ds_cifar_12.filtered_index[-1] == 7

    ds_cifar_13 = ds_cifar.filter(lambda sample: sample.labels.data()['value'] >= 2, offset=5, limit=2, num_workers=2)
    assert len(ds_cifar_13.filtered_index) == 2
    assert ds_cifar_13.filtered_index[-1] == 7

    ds_cifar_14 = ds_cifar.filter("labels >= 2", offset=1, limit=5)
    assert len(ds_cifar_14.filtered_index) == 5
    assert ds_cifar_14.filtered_index[-1] == 7

    ds_cifar_15 = ds_cifar.filter("labels >= 2", offset=1, limit=5, num_workers=2)
    assert len(ds_cifar_15.filtered_index) == 5
    assert ds_cifar_15.filtered_index[-1] == 7

    ds_cifar_16 = ds_cifar.filter("labels >= 2", offset=ds_cifar_15.filtered_index[-1]+1, limit=5, num_workers=2)
    assert len(ds_cifar_16.filtered_index) == 2
    assert ds_cifar_16.filtered_index[-1] == 9

    ds_cifar_16 = ds_cifar.filter("labels == 10", offset=0, limit=5, num_workers=2)
    assert len(ds_cifar_16.filtered_index) == 0

    # Test distributed filter
    try:
        import ray
        ray.init(address='auto')
        ds_cifar_ray = ds_cifar.filter(filter_labels(), num_workers=2, scheduler="distributed")
        assert len(ds_cifar_ray) == 2
        ray.shutdown()
    except ModuleNotFoundError as e:
        logging.info("Ray not found, Detailed info: %s", e)
    except ConnectionError as e:
        logging.info("Ray is not running on this node. Detailed info:  %s", e)


def test_filer_with_versions(storage):
    """
    Tests filter function after version changes.
    """
    ds_cifar = muller.load(path=official_path(storage, CIFAR10_TEST_PATH), creds=official_creds(storage))
    ds_cifar.checkout("new_b1", create=True)

    pics_list = get_cifar10_huashan(mode="train")
    for path in pics_list[:3]:
        with open(path.with_suffix(".txt"), "r") as fh:
            cls = int(fh.read())
        ds_cifar.images.append(muller.read(path))
        ds_cifar.labels.append(np.uint8(cls))
    ds_cifar.commit("first on new_b1")

    ds_cifar.pop(0)
    cifar_res = ds_cifar.filter("labels == 1 or labels == 8")
    assert len(cifar_res.filtered_index) == 3
    assert cifar_res.filtered_index[-1] == 7

    for path in pics_list[:3]:
        with open(path.with_suffix(".txt"), "r") as fh:
            cls = int(fh.read())
        ds_cifar.images.append(muller.read(path))
        ds_cifar.labels.append(np.uint8(cls))

    cifar_res_2 = ds_cifar.filter("labels >= 2", offset=1, limit=5)
    assert len(cifar_res_2.filtered_index) == 5
    assert cifar_res.filtered_index[-1] == 7
    ds_cifar.commit("sec on new_b1")

    ds_cifar.checkout("main", create=False)
    ds_cifar.merge("new_b1", append_resolution='both', update_resolution='ours', pop_resolution="ours")
    assert len(ds_cifar.images.numpy(aslist=True)) == 16


if __name__ == '__main__':
    pytest.main(["-s", "test_filter.py"])
