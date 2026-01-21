# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

import logging
import os
import time

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import muller
from muller.util.exceptions import UnAuthorizationError, SampleAppendError, UnsupportedMethod, TensorDoesNotExistError, \
    InvalidTensorNameError, SummaryLimit, ToDataFrameLimit, InvalidTensorList, DatasetCreationError, \
    DatasetNotExistsError, DatasetAlreadyExistsError
from muller.util.sensitive_config import SensitiveConfig
from tests.constants import SMALL_TEST_PATH, SAMPLE_FILES, CIFAR10_TEST_PATH, CIFAR10_MULTI_TEST_PATH, \
    DATASET_FILE, TEST_CHECKPOINT_PATH, LARGE_TEST_PATH
from tests.utils import official_path, official_creds, get_cifar10_huashan, verify_storage, \
    check_skip_time_consuming_test


def test_create_invalid_dataset(storage):
    # Use incorrect dataset name (e.g., with "@")
    try:
        muller.dataset(path=official_path(storage, SMALL_TEST_PATH+"@test_1"), creds=official_creds(storage),
                               overwrite=True)
        assert False, "No exception raises"
    except DatasetCreationError as e:
        assert True, f"There is an exception {e}"

    # Load an non-exist dataset
    try:
        muller.load(path=official_path(storage, SMALL_TEST_PATH), creds=official_creds(storage))
        assert False, "No exception raises"
    except DatasetNotExistsError as e:
        assert True, f"There is an exception {e}"

    # Create a dataset on existing MULLER path (without specifying overwrite)s
    muller.empty(path=official_path(storage, SMALL_TEST_PATH), creds=official_creds(storage), overwrite=True)
    try:
        muller.empty(path=official_path(storage, SMALL_TEST_PATH), creds=official_creds(storage))
        assert False, "No exception raises"
    except DatasetAlreadyExistsError as e:
        assert True, f"There is an exception {e}"


def test_create_small_dataset(storage):
    muller_ds = muller.dataset(path=official_path(storage, SMALL_TEST_PATH), creds=official_creds(storage), overwrite=True)
    muller_ds.create_tensor(name="images", htype="image", sample_compression="jpg")
    muller_ds.images.extend([muller.read(SAMPLE_FILES["jpg_1"])])
    assert muller_ds.images[0][0][0][0].numpy() == 243
    assert muller_ds.images[0].shape == (640, 640, 3)
    assert muller_ds.images[0].dtype == "uint8"


def test_create_cifar10_dataset(storage):
    ds_1 = muller.dataset(path=official_path(storage, CIFAR10_TEST_PATH), creds=official_creds(storage), overwrite=True)
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    with ds_1:
        # Create the tensors with names of your choice.
        ds_1.create_tensor("images", htype="image", sample_compression="jpeg")
        ds_1.create_tensor("labels", htype="class_label", class_names=class_names)

    pics_list = get_cifar10_huashan(mode="train")
    for path in pics_list:
        with open(path.with_suffix(".txt"), "r") as fh:
            cls = int(fh.read())
        ds_1.images.append(muller.read(path))
        ds_1.labels.append(np.uint8(cls))

    append_sample_label = ds_1.labels[0].numpy()
    assert ds_1.images[9].shape == (32, 32, 3)
    assert ds_1.labels[6:9].numpy()[0][0] == 2
    assert len(ds_1[3:9].numpy(aslist=True, asrow=True)) == 6
    assert len(ds_1[3:9][2].numpy(aslist=True, asrow=True)) == 1
    assert len(ds_1[3:9].numpy(aslist=True, asrow=False)) == 2
    assert len(ds_1[3:9].numpy(aslist=True, asrow=False)['images']) == 6
    assert len(ds_1[3:4].numpy(aslist=True, asrow=True)) == len(ds_1[3].numpy(aslist=True, asrow=True))
    assert type(ds_1[3:4].numpy(aslist=True, asrow=True)[0]) == type(ds_1[3].numpy(aslist=True, asrow=True)[0])

    ds_1.labels.append(append_sample_label)
    try:
        ds_1[9:].numpy(aslist=True, asrow=True)
        assert False, "No exception raises"
    except ValueError as exc:
        assert True, f"Caused exception {exc}"

    try:
        ds_1[9:].numpy(aslist=True, asrow=False)
        assert True, "No exception raises"
    except ValueError as exc:
        assert False, f"Caused exception {exc}"


def create_cifar10_dataset_parallel(storage, num_workers=2, scheduler="threaded"):
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    ds_multi = muller.dataset(path=official_path(storage, CIFAR10_MULTI_TEST_PATH),
                             creds=official_creds(storage), overwrite=True)
    with ds_multi:
        # Create the tensors with names of your choice.
        ds_multi.create_tensor("images", htype="image", sample_compression="jpg")
        ds_multi.create_tensor("labels", htype="class_label", class_names=class_names)
    iter_dict = []
    pics_list = get_cifar10_huashan(mode="train")
    i = 0
    for path in pics_list:
        with open(path.with_suffix(".txt"), "r") as fh:
            cls = int(fh.read())
        iter_dict.append((i, (path, cls)))

    @muller.compute
    def file_to_muller(data_pair, sample_out):
        sample_out.labels.append(np.uint32(data_pair[1][1]))
        sample_out.images.append(muller.read(data_pair[1][0]))
        return sample_out

    with ds_multi:
        file_to_muller().eval(iter_dict, ds_multi, num_workers=num_workers, scheduler=scheduler)

    return ds_multi


def create_cifar10_dataset_is_batch(storage, num_workers=2, scheduler="threaded"):
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    ds_multi = muller.dataset(path=official_path(storage, CIFAR10_MULTI_TEST_PATH),
                             creds=official_creds(storage), overwrite=True)
    with ds_multi:
        # Create the tensors with names of your choice.
        ds_multi.create_tensor("images", htype="image", sample_compression="jpg")
        ds_multi.create_tensor("labels", htype="class_label", class_names=class_names)
    iter_path = []
    iter_cls = []
    pics_list = get_cifar10_huashan(mode="train")
    for path in pics_list:
        with open(path.with_suffix(".txt"), "r") as fh:
            cls = int(fh.read())
        iter_path.append(muller.read(path))
        iter_cls.append(np.uint32(cls))

    @muller.compute(batch_enable=True)
    def file_to_muller(iter_path, iter_cls, sample_out):
        sample_out.labels.append(iter_cls)
        sample_out.images.append(iter_path)
        return sample_out

    with ds_multi:
        file_to_muller().eval(iter_path, iter_cls, ds_multi, num_workers=num_workers, scheduler=scheduler)

    return ds_multi


def test_create_cifar10_dataset_isbatch(storage):
    num_workers = 2
    ds_batch_thread = create_cifar10_dataset_is_batch(storage=storage, num_workers=num_workers, scheduler="threaded")
    ds_batch_thread.summary()
    assert ds_batch_thread.images[9].shape == (32, 32, 3)
    assert ds_batch_thread.labels[6:9].numpy()[0][0] == 2

    ds_batch_process = create_cifar10_dataset_is_batch(storage=storage, num_workers=num_workers, scheduler="processed")
    assert ds_batch_process.images[9].shape == (32, 32, 3)
    assert ds_batch_process.labels[6:9].numpy()[0][0] == 2


def test_create_cifar10_dataset_parallel(storage):
    num_workers = 2
    ds_multi_thread = create_cifar10_dataset_parallel(storage=storage, num_workers=num_workers, scheduler="threaded")
    ds_multi_thread.summary()
    assert ds_multi_thread.images[9].shape == (32, 32, 3)
    assert ds_multi_thread.labels[6:9].numpy()[0][0] == 2

    ds_multi_process = create_cifar10_dataset_parallel(storage=storage, num_workers=num_workers, scheduler="processed")
    assert ds_multi_process.images[9].shape == (32, 32, 3)
    assert ds_multi_process.labels[6:9].numpy()[0][0] == 2

    try:
        import ray
        ds_multi_distributed = create_cifar10_dataset_parallel(storage=storage,
                                                               num_workers=num_workers,
                                                               scheduler="distributed")
        assert ds_multi_distributed.images[9].shape == (32, 32, 3)
        assert ds_multi_distributed.labels[6:9].numpy()[0][0] == 2
        ray.shutdown()
    except ModuleNotFoundError as e:
        logging.info("Ray not found, Detailed info: %s", e)
    except ConnectionError as e:
        logging.info(f"Ray is not running on this node. Detailed info: %s", e)


def test_create_dataset_with_checkpoint(storage):
    """ A test case of creating dataset with checkpoints."""
    ds = muller.empty(path=official_path(storage, TEST_CHECKPOINT_PATH),
                     creds=official_creds(storage), overwrite=True)
    ds.create_tensor("test", htype="text")
    ds.create_tensor("test2", htype="text")

    @muller.compute
    def add_data(values, sample_out):
        sample_out.test.append("test")
        sample_out.test2.append("test2")
        return sample_out

    values = [0] * 10000
    with ds:
        add_data().eval(values, ds, num_workers=2, progressbar=True, scheduler="processed", checkpoint_interval=5000)

    assert ds.test.numpy()[0][0] == "test"


@pytest.mark.skipif(check_skip_time_consuming_test(), reason="It should be skipped if not in huashan-obs")
def test_create_dataset_with_checkpoint_after_multi_commits(storage):
    """ A test case of appending data with multi processes after 100 commits """
    ds = muller.empty(path=official_path(storage, TEST_CHECKPOINT_PATH),
                     creds=official_creds(storage), overwrite=True)
    ds.create_tensor("test", htype="text")
    ds.create_tensor("test2", htype="text")

    for i in range(100):
        ds.commit("committing " + str(i) + "th", allow_empty=True)

    @muller.compute
    def add_data(values, sample_out):
        sample_out.test.append("test")
        sample_out.test2.append("test2")
        return sample_out

    values = [0] * 10000
    with ds:
        add_data().eval(values, ds, num_workers=2, progressbar=True, scheduler="processed", checkpoint_interval=5000)

    assert ds.test.numpy()[0][0] == "test"


def test_load_dataset(storage):
    ds = muller.load(path=official_path(storage, CIFAR10_TEST_PATH), creds=official_creds(storage))
    ds.summary()
    assert ds.images[9].shape == (32, 32, 3)
    assert ds.labels[6:9].numpy()[0][0] == 2


def test_random_access(storage):
    total = 0
    index_bound = 9
    ds = muller.load(path=official_path(storage, CIFAR10_TEST_PATH), creds=official_creds(storage))
    for index, member in enumerate(ds):
        start_time = time.time()
        _ = member.images.numpy()  # x = member.images.data()["value"]
        end_time = time.time()
        total = total + (end_time - start_time)
        if index > index_bound:
            break


def test_fully_update(storage):
    ds_1 = muller.load(path=official_path(storage, CIFAR10_TEST_PATH), creds=official_creds(storage))
    ds_1[7].update({"labels": 3})
    assert ds_1.labels[7].numpy()[0] == 3
    ds_1.labels[7] = 4
    assert ds_1.labels[7].numpy()[0] == 4


def test_partial_update(storage):
    ds_1 = muller.dataset(path=official_path(storage, SMALL_TEST_PATH), creds=official_creds(storage), overwrite=True)
    ds_1.create_tensor(name="labels", htype="class_label")
    ds_1.labels.append([1, 2, 3, 4])
    ds_1.labels.append([5, 6, 7, 8])
    ds_1.labels[0][1:] = 0
    assert ds_1.labels[0].numpy()[1] == 0
    assert ds_1.labels[0].numpy()[2] == 0
    assert ds_1.labels[0].numpy()[3] == 0


def test_extend_pop(storage):
    ds_1 = muller.load(path=official_path(storage, CIFAR10_TEST_PATH), creds=official_creds(storage))
    ds_1.images.extend([
        muller.read(SAMPLE_FILES["jpg_2"]),
    ])
    ds_1.labels.extend([8, 9])
    ds_1.summary()
    assert ds_1.labels[-1].numpy()[0]

    ds_1.pop(index=0)
    assert ds_1.labels[0].numpy()[0] == 9

    with pytest.raises(SampleAppendError):
        ds_1.labels.extend([[[1], [2]], [[2], [3]]])


def test_pop_multiple_samples(storage):
    if not verify_storage(storage):
        ds_1 = muller.load(path=official_path(storage, CIFAR10_TEST_PATH), creds=official_creds(storage))
        ds_1.pop([0, 1])
        assert ds_1.labels[0].numpy()[0] == 4
        ds_1.pop([0, 1, 2, 4])
        assert ds_1.labels[0].numpy()[0] == 2
    else:
        return


def test_pop_single_tensor(storage):
    """ Verify that we cannot conduct direct pop from single tensor column. """
    ds_1 = muller.load(path=official_path(storage, CIFAR10_TEST_PATH), creds=official_creds(storage))
    try:
        ds_1.labels.pop([0, 1])
        assert False, "No exception raise"
    except UnsupportedMethod as e:
        assert True, f"Pop from tensor column causes error {e}"


def test_delete_tensor(storage):
    ds_1 = muller.load(path=official_path(storage, CIFAR10_TEST_PATH), creds=official_creds(storage))
    assert len(ds_1.tensors) == 2
    ds_1.delete_tensor("labels",large_ok=True)
    assert len(ds_1.tensors) == 1


def test_tensor_rename(storage):
    """Function to rename tensor."""
    ds = muller.dataset(path=official_path(storage, SMALL_TEST_PATH), creds=official_creds(storage), overwrite=True)
    ds.create_tensor("xyz")
    ds["xyz"].append([1])
    ds.rename_tensor("xyz", "xyy")

    assert ds["xyy"][0].numpy()[0] == 1

    with pytest.raises(TensorDoesNotExistError):
        ds["xyz"].numpy()

    ds.create_tensor("xyz")
    ds["xyz"].append([4, 5, 6])
    assert ds["xyz"][0].numpy()[0] == 4

    ds.create_tensor("abc")
    with pytest.raises(InvalidTensorNameError):
        ds.rename_tensor("abc", "append")


def test_create_and_add_dataset():
    sch = {
        'ori_query': ('text', '', 'lz4'),
        'ori_response': ('text', '', 'lz4'),
        'query分析': ('text', '', 'lz4'),
        'type': ('text', '', 'lz4'),
        '质检得分': ('generic', 'float32', 'lz4'),
        '质检结果': ('text', '', 'lz4'),
    }

    schema = {
        'ori_query': ('text', '', 'lz4'),
        'ori_response': ('text', '', 'lz4'),
        'query分析': {
            '通顺性评分': ('generic', 'float32', 'lz4'),
            '完整性评分': ('generic', 'float32', 'lz4'),
            '复杂度评分': ('generic', 'float32', 'lz4'),
            '安全性评分': ('generic', 'float32', 'lz4'),
            '综合质量评分': ('generic', 'float32', 'lz4'),
            '意图标签': ('text', '', 'lz4'),
        },
        'type': ('text', '', 'lz4'),
        '质检得分': ('generic', 'float32', 'lz4'),
        '质检结果': ('text', 'str', 'lz4'),
    }

    if not os.path.exists(DATASET_FILE):
        print("Please prepare your testing dataset to execute `test_create_and_add_dataset`.")
        return

    # with non-nested schema
    ds = muller.create_dataset_from_file(DATASET_FILE, SMALL_TEST_PATH, schema=sch, workers=0)
    assert ds.质检得分.meta.sample_compression == "lz4"
    ds.add_data_from_file(DATASET_FILE, schema=sch, workers=0)
    assert ds["type"][2].data()["value"] == "代码生成"

    # with nested schema
    ds_1 = muller.create_dataset_from_file(DATASET_FILE, SMALL_TEST_PATH, schema=schema, workers=0)
    assert ds_1["query分析.综合质量评分"].dtype == "float32"
    ds_1.add_data_from_file(DATASET_FILE, schema=schema, workers=0)
    assert ds_1["query分析.完整性评分"][20].data()["value"] == 5.0

    # with no schema
    ds_2 = muller.create_dataset_from_file(DATASET_FILE, SMALL_TEST_PATH, workers=0)
    assert ds_2.质检得分.dtype == "float64"
    ds_2.add_data_from_file(DATASET_FILE, workers=0)
    assert ds_2["ori_query"][199].data()["value"] == "找出我的5个优点"
    pass


def test_create_and_add_dataset_from_dataframes():
    sch = {
        'ori_query': ('text', '', 'lz4'),
        'ori_response': ('text', '', 'lz4'),
        'query分析': ('text', '', 'lz4'),
        'type': ('text', '', 'lz4'),
        '质检得分': ('generic', 'float32', 'lz4'),
        '质检结果': ('text', '', 'lz4'),
    }

    schema = {
        'ori_query': ('text', '', 'lz4'),
        'ori_response': ('text', '', 'lz4'),
        'query分析': {
            '通顺性评分': ('generic', 'float32', 'lz4'),
            '完整性评分': ('generic', 'float32', 'lz4'),
            '复杂度评分': ('generic', 'float32', 'lz4'),
            '安全性评分': ('generic', 'float32', 'lz4'),
            '综合质量评分': ('generic', 'float32', 'lz4'),
            '意图标签': ('text', '', 'lz4'),
        },
        'type': ('text', '', 'lz4'),
        '质检得分': ('generic', 'float32', 'lz4'),
        '质检结果': ('text', 'str', 'lz4'),
    }

    dataframes = [{
                      "ori_query": "1．使用if-else语句，实现如下比较功能：\n输入：in1,in2是两个3bit的二进制数；输出：out 是一个 3bit 的二进制数。\n如果 in1>in2，输出 out=001；如果 in1=in2，输出 out=010；如果 in1<in2，输出 out=100.",
                      "ori_response": "in1 = in2 = 0\n\nif in1 > in2:\n   out = 0b001\nelif in1 == in2:\n   out = 0b010\nelse:\n   out =0b100",
                      "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 3.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"实现比较功能\": [\"使用if-else语句\", \"输入：两个3bit的二进制数\", \"输出：一个3bit的二进制数\", \"比较条件：in1>in2，输出 out=001；如果 in1=in2，输出 out=010；如果 in1<in2，输出 out=100\"]}]}",
                      "质检结果": "综合质量分析:助手的回复在逻辑上是正确的，它准确地实现了用户所要求的功能。然而，助手的回复中存在一个问题，即它没有考虑到用户输入的“in1”和“in2”都是3位二进制数的情况。如果输入的“in1”和“in2”都是3位二进制数，那么助手的回复就会出现错误。此外，助手的回复中没有对用户输入进行任何检查或验证，这可能会导致用户输入错误的数据，从而引发错误的结果。因此，虽然助手的回复在语法和逻辑上是正确的，但在实际应用中可能会出现问题。\n\n综合质量评分:6分",
                      "质检得分": 6.0, "type": "代码生成"},
                  {
                      "ori_query": "编写一个Python程序，使用正则表达式匹配字符串中的所有手机号码。\n解析：首先，我们需要导入re模块。然后，使用re.findall(）函数，传入一个正则表达式模式（表示手机号码）和要搜索的字符串。最后，打印出匹配到的手机号码。",
                      "ori_response": "",
                      "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 2.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"编写Python程序\": [\"使用正则表达式匹配字符串中的手机号码\", \"导入re模块\", \"使用re.findall()函数\", \"打印匹配到的手机号码\"]}]}",
                      "质检结果": "综合质量分析:助手没有提供任何回复，所以无法进行质量分析。\n\n综合质量评分:0分",
                      "质检得分": 0.0, "type": "代码生成"},
                  {"ori_query": "帮我写一个从1加到100的代码",
                   "ori_response": "在Python中，可以使用for循环和一个累加器变量来实现从1加到100的代码。以下是一个简单的示例:\n\n```python\n#初始化累加器变量\ntotal = 0\n\n#使用for循环从1加到100\nfor i in range(1, 101):\n   total += i\n\n#打印结果\nprint(\"从1加到100的结果是:\", total)\n```\n\n在这个代码中，`range(1, 101)`生成一个从1到100的数字序列，`for`循环遍历这个序列，每次循环中的`i`变量依次取值1到100。`total`变量在每次循环中被累加，最终得到从1加到100的总和。\n\n执行这段代码，你会得到结果:\n\n```\n从1加到100的结果是: 5050\n```\n\n这是因为1 + 2 + 3 ... + 100等于5050。",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 1.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"编写代码\": [\"代码从1加到100\"]}]}",
                   "质检结果": "综合质量分析:助手的回复非常有用，它提供了一个清晰的 Python 代码片段来解决用户的问题。此外，助手还解释了代码的工作原理，这对于初学者来说是非常有帮助的。然而，助手的回复中存在一个错误，即1+2+3...+100的总和实际上是5050，而不是助手所说的5050。这是一个事实错误，可能会引起用户的困惑。\n\n综合质量评分:7分",
                   "质检得分": 7.0, "type": "代码生成"},
                  {"ori_query": "写一段1×2×3×4×5G5的c++语言",
                   "ori_response": "下面是一段1×2×3×4×5G5的C++代码:\n\n```c++\n#include <iostream>\n\nint main() {\n   int a = 1;\n   int b = 2;\n   int c = 3;\n   int d = 4;\n   int e = 5;\n\n   int result = a * b * c * d * e;\n\n   std::cout << \"1×2×3×4×5 = \" << result << std::endl;\n\n   return 0;\n}\n```\n\n这段代码定义了五个变量a、b、c、d和e,它们的值分别为1、2、3、4和5。然后通过将这些变量相乘，计算出了1×2×3×4×5的结果，并将其存储在变量result中。最后，使用std::cout输出了结果。",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 2.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"编写代码\": [\"使用C++语言\", \"代码内容为1×2×3×4×5G5\"]}]}",
                   "质检结果": "综合质量分析:助手的回复在事实性上存在问题，用户的问题是要求编写一个可以计算1到5的阶乘的 C++程序，而不是简单的乘法。因此，虽然助手的回复在语法和格式上是正确的，但在理解用户需求和提供正确答案的能力上存在明显的问题。\n\n综合质量评分:3分",
                   "质检得分": 3.0, "type": "代码生成"},
                  {"ori_query": "用C语言编译：计算圆柱的表面积与体积",
                   "ori_response": "在C语言中计算圆柱的表面积和体积，我们可以使用以下代码:\n\n```c\n#include <stdio.h>\n#include <math.h>\n\nint main() {\n   double r, h,表面积，体积；\n   \n   printf(\"请输入圆柱的底面半径(单位:米):\");\n   scanf(\"%lf\", &r);\n   \n   printf(\"请输入圆柱的高(单位:米):\");\n   scanf(\"%lf\", &h);\n   \n   //计算表面积\n   表面积= 2 * π * r * h + 2 * π * r * r;\n   printf(\"表面积:%.2f平方米\",表面积);\n   \n   //计算体积\n   体积= π * r * r * h;\n   printf(\"体积:%.2f立方米\",体积);\n   \n   return 0;\n}\n```\n\n这段代码首先定义了一个主函数`main`，在其中定义了四个变量:`r`(半径),`h`(高),`表面积`(表面积),`体积`(体积)。\n\n然后，程序使用`printf`和`scanf`函数提示用户输入半径和高，并将输入的值存储在相应的变量中。\n\n接着，程序使用公式计算圆柱的表面积和体积，并使用`printf`函数输出结果。\n\n注意，这里使用了`math.h`头文件中的`π`常数，你需要在编译时确保你的编译器支持这个头文件。如果你的编译器不支持，你可以使用`3.14159`或者`M_PI`宏来代替`π`。\n\n最后，程序返回0表示成功执行。\n\n在使用这段代码时，请确保你的编译器和环境变量设置正确，并且在编译时添加了`-stdio.h`和`-math.h`选项。\n\n编译和运行这段代码的命令可能如下:\n\n```sh\ngcc -o圆柱计算.c\n./圆柱\n```\n\n请注意，这里的`gcc`是GNU编译器，`-o`选项表示输出文件名，`圆柱计算.c`是你的源代码文件名。最后的`./圆柱`是运行编译后的程序的命令。",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 2.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"编写程序\": [\"使用C语言\", \"计算圆柱的表面积与体积\"]}]}",
                   "质检结果": "综合质量分析:助手的回复非常详细且准确，它提供了一段 C 语言代码来计算圆柱的表面积和体积，并解释了代码的工作原理。此外，助手还提供了编译和运行代码的步骤，这对于初学者来说非常有帮助。然而，助手在解释代码时，没有注意到代码中存在的错误。在计算圆柱体积的公式中，助手没有正确地使用半径的平方，而是使用了半径乘以高度，这是不正确的。此外，助手也没有注意到代码中的格式错误，例如，变量`表面积`和`体积`没有正确地初始化为0。因此，虽然助手的回复在其他方面都很好，但由于这些错误，其综合质量评分不能超过7分。\n\n综合质量评分:6分",
                   "质检得分": 6.0, "type": "代码生成"},
                  {
                      "ori_query": "int r = (pixel & 0xff0000) >> 16; //提取红色通道\nint g = (pixel & 0xff00) >> 8; //提取绿色通道\nint b = (pixel & 0xff); //提取蓝色通道\n怎么将上述三个通道的值合成一个rgb值",
                      "ori_response": "在Java中，你可以通过将提取出的R、G、B通道的值相加来合成一个RGB值。具体来说，如果R、G、B的值分别为r、g和b，那么合成的RGB值为：\n\n```java\nint rgb = r + g + b;\n```\n\n这个方法适用于大多数情况，因为RGB颜色空间中每个通道的值都是0到255之间的整数。但是请注意，如果R、G、B的值超出了0到255的范围，可能需要进行一些调整以确保结果在合法范围内。",
                      "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 2.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"解答编程问题\": [\"将三个通道的值合成一个rgb值\"]}]}",
                      "质检结果": "综合质量分析:助手的回答是准确和有用的，它准确地解释了如何将提取出的红色、绿色和蓝色通道的值相加来生成 RGB 值。此外，助手还提醒用户注意如果通道的值超出了0到255的范围，可能需要进行一些调整以确保结果在合法范围内。然而，助手的回答没有完全遵循用户的问题，用户的问题是如何将上述三个通道的值合成一个 rgb 值，而助手给出的回答是如何将三个通道的值相加，这两者在概念上是不同的。所以，虽然助手的回答在技术上是正确的，但它并没有完全回答用户的问题。\n\n综合质量评分:7分",
                      "质检得分": 7.0, "type": "代码生成"},
                  {"ori_query": "quartus ll 设计一个1hz到1Mhz的代码",
                   "ori_response": "在Quartus II中设计一个1Hz到1MHz的代码，你需要确定几个关键点:\n\n1.时钟信号的生成:你需要在你的设计中创建一个时钟信号，这个信号将用于同步你的电路。\n\n2.时钟频率的设置:在Quartus II中，你可以通过设置一个分频器(如一个2x2的计数器)来将你的时钟信号频率从1Hz调整到1MHz。\n\n以下是一个简单的示例，展示了如何在Quartus II中设计一个1Hz到1MHz的时钟信号:\n\n```verilog\n`timescale 1ns / 1ps\n\n模块clk_gen(\n   input wire clk_in,\n   output wire clk_out\n);\n\n// 2x2计数器\nreg [3:0] counter = 0;\n\n//时钟生成器\nalways @(posedge clk_in)\nbegin\n   counter = counter + 1;\n   if (counter == 4'b1111)\n   begin\n       counter <= 0;\n   end\nend\n\n//分频器\nassign clk_out = counter ？ 1 : 0;\n\nendmodule\n```\n\n在这个例子中，我们创建了一个简单的时钟生成器，它使用了一个2x2的计数器来生成一个4比特的时钟信号。这个计数器在每个时钟上升沿时增加，当计数器达到11111时，它重置为零。这个时钟信号被用来控制其他逻辑的运行频率。\n\n要将这个时钟信号的频率从1Hz调整到1MHz,你需要将计数器的深度从4位增加到更高的位数，以确保计数器不会在一个时钟周期内完成其循环。例如，如果你想要一个1MHz的时钟，你可以将计数器的深度设置为16位:\n\n```verilog\n`timescale 1ns / 1ps\n\n模块clk_gen(\n   input wire clk_in,\n   output wire clk_out\n);\n\n",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 3.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"设计代码\": [\"代码频率范围为1Hz到1MHz\", \"使用Quartus II\"]}]}",
                   "质检结果": "综合质量分析:助手的回答详细地解释了如何在 Quartus II 中设计一个1Hz 到1MHz 的代码，包括了时钟信号的生成和频率设置的关键点，并给出了一个简单的代码示例。然而，回答在最后一部分突然中断，没有完整的代码示例，这可能会让用户感到困惑。\n\n综合质量评分:7分",
                   "质检得分": 7.0, "type": "代码生成"},
                  {"ori_query": "用C语言编译：输入长方形的边长，求其面积。",
                   "ori_response": "下面是一个简单的C语言程序，用于输入长方形的边长，然后计算并打印出其面积:\n\n```c\n#include <stdio.h>\n\nint main() {\n",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 1.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"编译程序\": [\"使用C语言\", \"输入长方形的边长\", \"计算并输出其面积\"]}]}",
                   "质检结果": "综合质量分析:助手的回复并未完成，它没有给出完整的 C 语言代码来解决用户的问题。用户需要一个可以输入长方形的边长并计算其面积的 C 语言程序，但助手只提供了一个空的代码框架，没有具体的代码实现。因此，这个回复没有提供实际的帮助，也无法满足用户的需求。\n\n综合质量评分:2分",
                   "质检得分": 2.0, "type": "代码生成"},
                  {"ori_query": "帮我用C语言编写在51单片机上的程序，要求是让LED灯呼吸",
                   "ori_response": "在51单片机上编写LED灯呼吸程序，通常需要使用Timer和中断来实现。以下是一个简单的示例程序，它可以让LED灯按照一定的频率闪烁:\n\n```c\n#include <reg51.h>\n\nsbit LED = P1^0; //定义LED引脚\n\nvoid main() {\n   unsigned int i;\n\n   //初始化Timer0\n   TMR0 = 0;\n   肉.Timer0Init(10000); // 10ms中断\n   肉.OpenTimer0();\n\n   while (1) {\n       肉.CloseTimer0();\n       for (i = 0; i < 200; i++) { // LED每隔10ms闪烁一次\n           LED = ！LED;\n       }\n       肉.OpenTimer0();\n   }\n}\n\nvoid Timer0中断(void) {\n   TMR0 = 0; //清除中断标志\n   LED = ！LED; //切换LED状态\n}\n```\n\n这段程序使用了`Timer0`中断来实现LED灯的闪烁。`Timer0`的中断周期为10毫秒，因此LED灯会每隔10毫秒切换一次状态。在`main`函数中，我们使用了一个`for`循环来让LED灯闪烁200次，然后再次开启`Timer0`。\n\n请注意，这只是一个简单的示例程序，实际应用中可能需要更复杂的逻辑来实现不同的闪烁模式。此外，由于51单片机的资源限制，可能需要对程序进行优化以确保良好的性能。",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 3.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"编写程序\": [\"使用C语言\", \"在51单片机上运行\", \"程序要求：让LED灯呼吸\"]}]}",
                   "质检结果": "综合质量分析:助手的回复非常详细和准确，为用户提供了一个在51单片机上编写 LED 灯呼吸程序的示例代码。此外，助手还解释了代码的工作原理，并提醒用户在实际应用中可能需要更复杂的逻辑，以及可能需要对程序进行优化以确保良好的性能。然而，回复中的代码存在一些错误，例如\"肉.Timer0Init(10000);\"和\"肉.OpenTimer0();\"等，这些错误可能会导致程序无法编译和运行。\n\n综合质量评分:6分",
                   "质检得分": 6.0, "type": "代码生成"},
                  {"ori_query": "写一段基于深度学习的智能客服系统设计与实现的程序代码",
                   "ori_response": "以下是一个简单的基于深度学习的智能客服系统的程序代码示例:\n\n```python\n#导入必要的库\nimport numpy as np\nimport tensorflow as tf\nfrom keras.preprocessing.text import Tokenizer\nfrom keras.preprocessing.sequence import pad_sequences\n\n#设置模型训练参数\nmax_seq_length = 50  #最大序列长度\nmax_words = 10000  #最大词汇量\nepochs = 10  #训练轮数\n\n#构建模型\nmodel = tf.keras.Sequential()\nmodel.add(tf.keras.layers.Embedding(max_words, 16))  #嵌入层\nmodel.add(tf.keras.layers.GlobalAveragePooling1D())  #全局平均池化层\nmodel.add(tf.keras.layers.Dense(1, activation='sigmoid'))  #输出层\nmodel.compile(optimizer='adam', loss='binary_crossentropy', metricss=['accuracy'])\n\n#数据预处理\ntokenizer = Tokenizer(num_words=max_words)\ntokenizer.fit_on_texts(texts)  #训练tokenizer\nsequences = tokenizer.texts_to_sequences(texts)\npadded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')\n\n#训练模型\nmodel.fit(padded_sequences, labels, epochs=epochs)\n\n#预测\ndef model_predict(text):\n   sequence = tokenizer.texts_to_sequences([text])\n   padded = pad_sequences([sequence], maxlen=max_seq_length, padding='post')\n   prediction = model.predict(padded)\n   return prediction[0][0]\n\n#使用模型进行预测\ntext = \"How do I reset my password?\"\nprediction = model_predict(text)\nprint(f\"Prediction: {prediction}\")\n```\n\n这段代码实现了一个简单的基于深度学习的智能客服系统，包括数据预处理、模型构建、模型训练和预测等步骤。在实际应用中，需要根据具体的业务需求和数据情况进行调整和优化。",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 4.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"编写程序代码\": [\"基于深度学习的智能客服系统设计与实现\"]}]}",
                   "质检结果": "综合质量分析:助手的回复提供了一个基于深度学习的智能客服系统的代码实现，包括数据预处理、模型构建、模型训练和预测等步骤。然而，这段代码并未完全符合用户的需求，因为它并未包含智能客服系统的所有关键组件，如对话管理、意图识别和实体识别等。此外，该代码也没有提供任何关于如何使用这个模型的上下文信息，例如如何在实际的客服场景中部署和使用这个模型。因此，虽然该回复在技术上是准确的，但它并未完全满足用户的需求。\n\n综合质量评分:6分",
                   "质检得分": 6.0, "type": "代码生成"},
                  {"ori_query": "给我写一个办公室为主题的搞笑脚本",
                   "ori_response": "场景:办公室\n\n人物:A(经理)、B(员工)、C(另一位员工)\n\nA:(看着电脑屏幕，突然A的手机响起)喂，喂？(接起电话)哦，是的，我在办公室。(看了看周围) B,你能帮我倒杯咖啡吗？\n\nB:(从旁边的咖啡机旁边站起来)当然，经理。(走到咖啡机前，但咖啡机是坏的)哇，咖啡机又坏了？\n\nA:(尴尬地笑了笑)是的，看来我们需要请个专业人士来修。\n\nB:(也笑了)好的，我会去找我们的“办公室修理工”。\n\nC:(从旁边经过，听到了他们的对话)哦，我就是那个“办公室修理工”。\n\nA:(笑)原来如此，那就拜托你了，C。\n\nB:(也笑)我们的“咖啡机修理工”，谢谢你。\n\nC:(摆手)不客气，我很高兴能帮到你们。\n\n(大家都笑了起来，继续各自的工作)",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 4.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"编写脚本\": [\"主题为办公室\", \"要求搞笑\"]}]}",
                   "质检结果": "综合质量分析:该回复的内容是一个办公室场景的幽默小故事，人物设定和情节设计都比较合理，符合用户的需求。在语言表达上，对话流畅，没有出现语法错误或者不通顺的地方。然而，这个故事的幽默效果可能会因人而异，对于一些人来说，可能并不觉得好笑。此外，这个故事并没有提供实际的办公室问题解决方案，只是一个轻松的办公室小插曲。\n\n综合质量评分:7分",
                   "质检得分": 7.0, "type": "代码生成"},
                  {"ori_query": "我想用matlab设计模型，生成代码，最后放到博图里控制s1200plc可以吗",
                   "ori_response": "是的，你可以使用MATLAB来设计模型，并生成可以在言控(Y煤控)里运行的代码。以下是一个简单的步骤，展示了如何从MATLAB生成可以控制S1200系列的PLC的代码:\n\n1. **安装MATLAB**:确保你的计算机上安装了MATLAB。\n\n2. **了解S1200 PLC**:在开始之前，你需要了解S1200 PLC的基本知识，包括它的指令集和编程结构。\n\n3. **设计你的模型**:在MATLAB中，你可以使用各种工具和函数来设计你的控制模型。例如，你可以使用`fmt`函数来定义你的指令，或者使用`sim`函数来模拟你的模型。\n\n4. **生成言控代码**:\n  -使用MATLAB的`串流`(StringStream)功能来创建你的指令字符串。\n  -使用`通信`(Communications)功能来将你的指令发送到S1200 PLC。\n\n5. **将代码导入到博图**:\n  -将生成的代码导出为文本格式，如`.m`文件。\n  -使用博图的编程工具(如STEP 7 Studio)打开这个文件。\n  -将代码在博图的编程环境中执行，确保它能够正确地控制S1200 PLC。\n\n6. **测试和调试**:在博图中运行你的代码，并进行必要的测试和调试，以确保它能够按照预期工作。\n\n请注意，这只是一个基本的指导，具体的实现可能会根据你的具体需求和S1200 PLC的配置有所不同。如果你需要更详细的指导，建议查阅相关的MATLAB文档和S1200 PLC的用户手册，或者寻求专业的技术支持。",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 3.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"回答问题\": [\"使用matlab设计模型\", \"生成代码\", \"最后放到博图里控制s1200plc\"]}]}",
                   "质检结果": "综合质量分析:助手的回复在有用性和事实性上表现良好，提供了详细的步骤和方法来从 MATLAB 生成可以在 S1200 PLC 上运行的代码。然而，回复中存在一些明显的错误，例如“言控(Y 煤控)”应该是“博图”，这是一个明显的输入错误。此外，“fmt”和“通信”函数在 MATLAB 中并不存在，这也是一个错误。因此，虽然助手的回复在内容上是有价值的，但由于这些明显的错误，其事实性和无害性受到了影响。\n\n综合质量评分:6分",
                   "质检得分": 6.0, "type": "代码生成"},
                  {"ori_query": "给我写一个贪吃蛇微信小程序代码",
                   "ori_response": "以下是一个简单的贪吃蛇小程序的示例代码:\n\n```\n//微信小程序基础框架\n\nPage({\n //游戏初始化\n onInit: function() {\n   var that = this;\n   //创建游戏画布\n   var ctx = that.canvas.getContext('2d');\n   //设置画布大小\n   that.canvas.width = 400;\n   that.canvas.height = 400;\n   //初始化蛇\n   var snake = [\n     { x: 20, y: 20 },\n     { x: 10, y: 20 },\n     { x: 0, y: 20 }\n   ];\n   //设置蛇的移动方向\n   var dx = 0;\n   var dy = 0;\n   //设置食物\n   var food = { x: Math.floor(Math.random() * 400), y: Math.floor(Math.random() * 400) };\n   //游戏循环\n   setInterval(function() {\n     //绘制蛇\n     ctx.fillStyle = '#000';\n     ctx.fillRect(snake[0].x, snake[0].y, 20, 20);\n     for (var i = 1; i < snake.length; i++) {\n       ctx.fillRect(snake[i].x, snake[i].y, 20, 20);\n     }\n     //绘制食物\n     ctx.fillStyle = '#00F';\n     ctx.fillRect(food.x, food.y, 20, 20);\n     //判断游戏是否结束\n     if (snake[0].x == food.x && snake[0].y == food.y) {\n       //吃到食物\n       food = { x: Math.floor(Math.random() * 400), y: Math.floor(Math.random() * 400) };\n     } else {\n       //撞到墙壁或自身\n       snake.splice(0, 1);\n     }\n     //更新蛇的移动方向\n     if (dy > 0) {\n       dx = -10;\n     } else if (dy < 0) {\n       dx = 10;\n     }\n   }, 200);\n },\n //触摸开始\n touchStart: function(e) {\n   var that = this;\n   var x = e.touches[0].clientX;\n   var y = e.touches[0].clientY;\n   //判断是否为开始位置\n   if (x == 0 && y == 0) {\n     return;\n   }\n   //移动蛇\n   var head = { x: snake[0].x, y: snake[0].y };\n   snake.unshift(head);\n   if (snake.length > 20) {\n     snake.pop();\n   }\n },\n //触摸结束\n touchEnd: function(e) {\n   var that = this;\n   var x = e.changedTouches[0].clientX;\n   var y = e.changedTouches[0].clientY;\n   //判断是否为结束位置\n   if (x == 0 && y == 0) {\n     return;\n   }\n   //停止游戏循环\n   clearInterval(timer);\n }\n})\n```",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 4.0, \"安全性评分\": 5.0, \"综合质量评分\": 4.75, \"意图标签\": [{\"编写代码\": [\"微信小程序代码\", \"游戏类型为贪吃蛇\"]}]}",
                   "质检结果": "综合质量分析:助手的回复提供了一个基本的贪吃蛇小程序的代码，包括游戏初始化、蛇的移动、食物的生成、游戏循环、触摸事件等功能。代码的结构清晰，注释也比较详细，对于初学者来说，这是一个很好的学习资料。然而，这个代码只是一个基本的实现，没有包括微信小程序的所有特性，比如页面跳转、数据存储等。此外，对于一些高级特性，比如碰撞检测，助手没有给出实现的代码。总体来说，这个回复对于初学者来说是有用的，但是对于有一定经验的开发者来说，可能需要更多的信息和代码。\n\n综合质量评分:7分",
                   "质检得分": 7.0, "type": "代码生成"},
                  {"ori_query": "用fortran画黄金螺旋线",
                   "ori_response": "在Fortran中绘制黄金螺旋线(Golden Spiral)可以通过循环和数学计算来实现。黄金螺旋线是一种特殊的无限大迭代函数，其迭代公式如下:\n\n\\\\[ x_{n+1} = \\\\frac{1}{n+1} \\\\]\n\n\\\\[ y_{n+1} = \\\\frac{x_{n+1}}{n+1} \\\\]\n\n\\\\[ z_{n+1} = \\\\frac{1}{y_{n+1}} \\\\]\n\n\\\\[ x_{n+2} = \\\\frac{1}{z_{n+1}} \\\\]\n\n\\\\[ y_{n+2} = \\\\frac{x_{n+2}}{z_{n+1}} \\\\]\n\n\\\\[ z_{n+2} = \\\\frac{1}{y_{n+2}} \\\\]\n\n可以使用上述公式来生成黄金螺旋线的点，然后通过Fortran的绘图函数将这些点连接起来，从而得到黄金螺旋线的图像。\n\n以下是一个简单的Fortran程序示例，用于生成黄金螺旋线的前几项:\n\n```fortran\nprogram golden_spiral\n implicit none\n \n integer :: n, i\n real :: x(100), y(100), z(100)\n \n x = [1./(n+1), 1./(n+2), 1./(n+3)] ！初始化x, y, z数组\n \n do i = 1, 10 ！迭代生成10个点\n   z(i) = 1./y(i)\n   y(i+1) = x(i)/z(i)\n   x(i+2) = 1./y(i+1)\n end do\n \n ！绘制图形\n call plot_golden_spiral(x, y, z, n)\n \ncontains\n subroutine plot_golden_spiral(x, y, z, n)\n   implicit none\n   \n   integer :: i\n   \n   call plot(x, y, '-o', 'Name', 'Golden Spiral')\n   \n   do i = 1, n\n     call plot(x(i), y(i), '-', 'Width', 1)\n   end do\n end subroutine plot_golden_spiral\nend program golden_spiral\n```\n\n在这个程序中，`plot_golden_spiral`子程序用于绘制黄金螺旋线，而`golden_spiral`程序则负责生成一系列点，并调用`plot_golden_spiral`子程序来绘制这些点。\n\n请注意，这个程序是一个简化的示例，实际的Fortran绘图可能需要更多的错误处理和绘图函数调用。此外，你可能需要使用Fortran的图形库，如Intel Math Kernel Library (MKL)或OpenGL,来进行更复杂的绘图操作。",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 2.0, \"安全性评分\": 5.0, \"综合质量评分\": 5.0, \"意图标签\": [{\"编写代码\": [\"使用fortran\", \"画黄金螺旋线\"]}]}",
                   "质检结果": "综合质量分析:助手的回复非常详细且具有实用性，它提供了一个用 Fortran 生成黄金螺旋线的具体程序，并解释了如何使用这个程序。此外，助手还提醒用户可能需要使用 Fortran 的图形库，如 Intel Math Kernel Library (MKL)或 OpenGL,来进行更复杂的绘图操作。然而，这个回复中的 Fortran 代码可能存在一些问题，例如数组 x, y, z 的大小被设定为100，但并没有解释为什么要设定为100，这可能会让用户产生困惑。此外，该回复没有提供如何绘制黄金螺旋线的具体函数，只是提到了一个`plot_golden_spiral`子程序，但并没有给出这个子程序的具体实现。因此，虽然这个回复提供了一些有用的信息，但可能还需要一些改进。\n\n综合质量评分:7分",
                   "质检得分": 7.0, "type": "代码生成"},
                  {"ori_query": "使用pyqt写一个登陆界面，并显示pyqt的代码",
                   "ori_response": "在这里，我将为您提供一个简单的PyQt5登陆界面的代码示例。请注意，这只是一个基本的示例，您可能需要根据您的具体需求进行修改。\n\n首先，您需要安装PyQt5库。您可以使用以下命令在命令行中安装:\n\n```bash\npip install PyQt5\n```\n\n接下来，我们将创建一个基本的登陆窗口。以下是示例代码:\n\n```python\nimport sys\nfrom PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout\n\nclass LoginWindow(QWidget):\n   def __init__(self):\n       super().__init__()\n\n       #创建布局\n       layout = QVBoxLayout()\n\n       #添加标签\n       label_username = QLabel('用户名:')\n       layout.addWidget(label_username)\n\n       #添加输入框\n       self.username_edit = QLineEdit()\n       layout.addWidget(self.username_edit)\n\n       #添加标签\n       label_password = QLabel('密码:')\n       layout.addWidget(label_password)\n",
                   "query分析": "{\"通顺性评分\": 5.0, \"完整性评分\": 5.0, \"复杂度评分\": 3.0, \"安全性评分\": 5.0, \"综合质量评分\": 4.5, \"意图标签\": [{\"编写代码\": [\"使用pyqt\", \"编写登陆界面\"], \"显示代码\": [\"显示pyqt的代码\"]}]}",
                   "质检结果": "综合质量分析:助手的回复非常有用，它提供了一个基本的 PyQt5登陆界面的代码示例，并详细解释了如何安装 PyQt5库。然而，回复在代码示例部分被截断，没有提供完整的代码，这可能会让用户感到困惑。此外，助手没有提供任何关于如何运行或测试代码的说明，这可能会使用户在实际使用中遇到困难。\n\n综合质量评分:6分",
                   "质检得分": 6.0, "type": "代码生成"}]

    # with non-nested schema
    ds = muller.create_dataset_from_dataframes(dataframes, SMALL_TEST_PATH, schema=sch, workers=0)
    assert len(ds) == 15
    assert ds.质检得分.meta.sample_compression == "lz4"
    ds.add_data_from_dataframes(dataframes=dataframes, schema=sch, workers=0)
    assert len(ds) == 30
    assert ds["type"][2].data()["value"] == "代码生成"

    # with nested schema
    ds_1 = muller.create_dataset_from_dataframes(dataframes, SMALL_TEST_PATH, schema=schema, workers=0)
    assert ds_1["query分析.综合质量评分"].dtype == "float32"
    ds_1.add_data_from_dataframes(dataframes, schema=schema, workers=0)
    assert ds_1["query分析.完整性评分"][20].data()["value"] == 5.0

    # with no schema
    ds_2 = muller.create_dataset_from_dataframes(dataframes, SMALL_TEST_PATH, workers=0)
    assert ds_2.质检得分.dtype == "float64"
    ds_2.add_data_from_dataframes(dataframes, workers=0)
    assert ds_2["ori_query"][14].data()["value"] == "使用pyqt写一个登陆界面，并显示pyqt的代码"

    pass

def test_merge_tensor_meta(storage):
    muller_ds = muller.dataset(path=official_path(storage, SMALL_TEST_PATH),
                           creds=official_creds(storage), overwrite=True, split_tensor_meta=False)
    muller_ds.create_tensor(name="images", htype="image", sample_compression="jpg")
    muller_ds.images.extend([muller.read(SAMPLE_FILES["jpg_1"])])
    assert muller_ds.images[0][0][0][0].numpy() == 243
    assert muller_ds.images[0].shape == (640, 640, 3)
    assert muller_ds.images[0].dtype == "uint8"

    ds_1 = muller.load(path=official_path(storage, SMALL_TEST_PATH),
                      creds=official_creds(storage), split_tensor_meta=False)
    ds_1.reset()
    ds_1.create_tensor(name="labels", htype="class_label")
    ds_1.labels.append([1, 2, 3, 4])
    ds_1.labels.append([5, 6, 7, 8])
    ds_1.labels[0][1:] = 0
    assert ds_1.labels[0].numpy()[1] == 0
    assert ds_1.labels[0].numpy()[2] == 0
    assert ds_1.labels[0].numpy()[3] == 0


def test_checkpoint_with_multi_user(storage):
    """tests ckpt with multi users"""
    # User A append data via ckpt.
    SensitiveConfig().uid = "A"
    ds_multi = muller.dataset(path=official_path(storage, SMALL_TEST_PATH),
                             creds=official_creds(storage), overwrite=True)
    with ds_multi:
        ds_multi.create_tensor("test1", htype="text")
        ds_multi.create_tensor("test2", htype="text")

    iter_dict = []
    for i in range(0, 1000):
        iter_dict.append((i, ("hi", "hello")))

    @muller.compute
    def file_to_muller(data_pair, sample_out):
        sample_out.test1.append(data_pair[1][1])
        sample_out.test2.append(data_pair[1][0])
        return sample_out

    with ds_multi:
        file_to_muller().eval(iter_dict, ds_multi, num_workers=4, scheduler="processed", disable_rechunk=True,
                            checkpoint_interval=500)

    ds_multi.commit()

    # User B query the data
    SensitiveConfig().uid = "B"
    ds_filter1 = ds_multi.filter_vectorized([("test1", "==", 'hello')])
    assert len(ds_filter1) == 1000

    # User B append data
    with pytest.raises(UnAuthorizationError):
        ds_multi.test1.append("hi")
        ds_multi.test2.append("hello")

    assert len(ds_multi) == 1000

    # User B checkout to Branch B and append data
    ds_multi = muller.load(path=official_path(storage, SMALL_TEST_PATH), creds=official_creds(storage))
    ds_multi.checkout("branchB", create=True)
    ds_multi.test1.append("hi")
    ds_multi.test2.append("hello")
    assert len(ds_multi) == 1001

    # User C checkout Branch C and append data using ckpt
    SensitiveConfig().uid = "C"
    ds_multi = muller.load(path=official_path(storage, SMALL_TEST_PATH), creds=official_creds(storage))
    ds_multi.checkout("branchC", create=True)
    with ds_multi:
        file_to_muller().eval(iter_dict, ds_multi, num_workers=4, scheduler="processed", disable_rechunk=True,
                            checkpoint_interval=500)
    assert len(ds_multi) == 2000

    # User B query the data of user C
    SensitiveConfig().uid = "B"
    ds_filter1 = ds_multi.filter_vectorized([("test1", "==", 'hello')])
    assert len(ds_filter1) == 2000

    # User B append data
    with pytest.raises(UnAuthorizationError):
        ds_multi.test1.append("hi")
        ds_multi.test2.append("hello")
    assert len(ds_multi) == 2000

    # User B checkout to Branch B_2 and append data
    ds_multi = muller.load(path=official_path(storage, SMALL_TEST_PATH), creds=official_creds(storage))
    ds_multi.checkout("branchC")
    ds_multi.checkout("branchB_2", create=True)
    ds_multi.test1.append("hi")
    ds_multi.test2.append("hello")
    assert len(ds_multi) == 2001


def test_summary(storage):
    ds = muller.dataset(path=official_path(storage, LARGE_TEST_PATH), creds=official_creds(storage), overwrite=True)
    with ds:
        ds.create_tensor(name="labels", htype="generic", dtype="int")
        ds.labels.extend([0, 1, 2, 3, 4]*2001)
        ds.create_tensor(name="categories", htype="text")
        ds.categories.extend(["a", "b", "c", "d", "e"]*2001)
    try:
        ds.summary()
    except SummaryLimit as e:
        assert True, f"exception: {e}"

    ds.summary(force=True)


def test_to_dataframe_small(storage):
    """ Small test """
    ds = muller.dataset(path=official_path(storage, SMALL_TEST_PATH), creds=official_creds(storage), overwrite=True)
    with ds:
        ds.create_tensor(name="labels", htype="generic", dtype="int")
        ds.labels.extend([0, 1, 2, 3, 4])
        ds.create_tensor(name="categories", htype="text")
        ds.categories.extend(["a", "b", "c", "d", "e"])

    df = ds.to_dataframe()
    df_1 = pd.DataFrame({"labels": [0, 1, 2, 3, 4], "categories": ["a", "b", "c", "d", "e"]})
    assert_frame_equal(df, df_1)

    numpy_list = [2, 3]
    df_2 = ds.to_dataframe(index_list=numpy_list)
    df_3 = pd.DataFrame({"labels":[2, 3], "categories":["c", "d"]})
    assert_frame_equal(df_2, df_3)

    try:
        df_4 = ds.to_dataframe(tensor_list=["test", "categories"])
    except InvalidTensorList as e:
        assert True, f"exception: {e}"

    df_4 = ds.to_dataframe(tensor_list=["categories"], index_list=[2, 3])
    df_5 = pd.DataFrame({"categories": ["c", "d"]})
    assert_frame_equal(df_4, df_5)

    df_6 = ds.to_dataframe(tensor_list=["categories"], index_list=[-1, -2])
    df_7 = pd.DataFrame({"categories": ["e", "d"]})
    assert_frame_equal(df_6, df_7)


def test_to_dataframe_large(storage):
    """ Large test """
    ds = muller.dataset(path=official_path(storage, LARGE_TEST_PATH), creds=official_creds(storage), overwrite=True)
    with ds:
        ds.create_tensor(name="labels", htype="generic", dtype="int")
        ds.labels.extend(np.arange(10010).tolist())
        ds.create_tensor(name="categories", htype="text")
        ds.categories.extend(["a"] * 10010)

    try:
        df = ds.to_dataframe()
    except ToDataFrameLimit as e:
        assert True, f"exception: {e}"

    try:
        df = ds.to_dataframe(index_list=np.arange(1, 10004).tolist())
    except ToDataFrameLimit as e:
        assert True, f"exception: {e}"

    df = ds.to_dataframe(index_list=np.arange(1, 10004).tolist(), force=True)
    df_1 = pd.DataFrame({"labels":np.arange(1, 10004).tolist(), "categories":["a"]*10003})
    assert_frame_equal(df, df_1)


if __name__ == '__main__':
    pytest.main(["-s", "test_dataset.py"])
