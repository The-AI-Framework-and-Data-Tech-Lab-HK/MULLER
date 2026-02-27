# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""
Comprehensive tests for numpy conversion methods including:
- Basic .numpy() functionality with various parameters
- Optimization methods: get_samples_continuous, get_samples_full, get_samples_batch_random_access
- Multiple data types and chunk types
"""

import random

import numpy as np
import pytest

import muller
from tests.constants import TEST_NUMPY_PATH, SAMPLE_FILES
from tests.utils import official_path, official_creds


def get_raw_data():
    return [
        ['1．使用if-else语句，实现比较功能', 'if in1 > in2:', '{"通顺性评分": 5.0, "完整性评分": 3.0}]}', '综合质量评分:6分', 6.0, 5.0],
        ['编写一个Python程序，使用正则表达式匹配手机号码。', '首先，我们需要导入re模块。', '{"通顺性评分": 5.0, "完整性评分": 2.0}]}', '综合质量评分:0分', 0.0, 4.0],
        ['用C语言编译：计算圆柱的表面积与体积', '定义了四个变量:`r`,`h`,`表面积`,`体积`', '{"通顺性评分": 5.0, "完整性评分": 4.0}]}', '综合质量评分:6分', 6.0, 6.0],
        ['怎么将rgb三个通道的值合成一个rgb值', 'Java中，将提取出的R、G、B通道的值相加来合成一个RGB值。', '{"通顺性评分": 5.0, "完整性评分": 4.0}]}', '综合质量评分:7分', 7.0, 4.0],
        ['给我生成一个游戏昵称', '风之旅者', '{"通顺性评分": 5.0, "完整性评分": 5.0}]}', '综合质量评分:5分', 5.0, 7.0],
        ['写一段基于深度学习的智能客服系统设计', '基于深度学习的智能客服系统，包括数据预处理、模型构建、模型训练和预测等步骤。', '{"通顺性评分": 5.0, "完整性评分": 5.0}]}', '综合质量评分:6分', 6.0, 3.0]]


def create_dataset(storage):
    ds = muller.dataset(path=official_path(storage, TEST_NUMPY_PATH), creds=official_creds(storage), overwrite=True)
    tensors = ["ori_query", "ori_response", "query_analysis", "result", "score", "type"]
    ds.create_tensor("ori_query", htype="text", exist_ok=True)
    ds.create_tensor("ori_response", htype="text", exist_ok=True)
    ds.create_tensor("query_analysis", htype="text", exist_ok=True)
    ds.create_tensor("result", htype="text", exist_ok=True)
    ds.create_tensor("score", htype="generic", exist_ok=True, dtype="float64")
    ds.create_tensor("type", htype="generic", exist_ok=True, dtype="float64")
    np_data = np.array(get_raw_data())
    for i, item in enumerate(tensors):
        ds[item].extend(np_data[:, i].astype(ds[item].dtype))
    return ds


def create_dataset_with_fixed_shape(storage):
    ds = muller.dataset(path=official_path(storage, TEST_NUMPY_PATH), creds=official_creds(storage), overwrite=True)
    ds.create_tensor(name="age", dtype="int", max_chunk_size=7)
    for i in range(20):
        ds.age.append([i])
    ds.create_tensor(name="height", dtype="float64", chunk_compression="lz4")
    for i in range(150, 170):
        ds.height.append([i])
    ds.create_tensor(name="photo", htype="image", sample_compression="jpg")
    ds.photo.extend([muller.read(SAMPLE_FILES["jpg_1"])])
    return ds


def create_text_dataset(storage):
    values = ["白日依山尽，黄河入海流，欲穷千里目，更上一层楼",
              "床前明月光，疑是地上霜，举头邀明月，低头思故乡",
              "[unused10][unused9]助手",
              "我是deepseek，迅雷不及掩耳盗铃儿响叮当仁不让世界充满爱之势!你是谁？",
              "All happy families are happy alike, all unhappy families are unhappy in their own way."]
    ds = muller.dataset(path=official_path(storage, TEST_NUMPY_PATH), creds=official_creds(storage), overwrite=True)

    with ds:
        ds.create_tensor('value', htype='text')
        ds.value.extend(values*2000)
    return ds


def create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100):
    """Create UncompressedChunk dataset with specified dtype."""
    ds = muller.dataset(
        path=official_path(storage, TEST_NUMPY_PATH),
        creds=official_creds(storage),
        overwrite=True
    )
    ds.create_tensor("tensor_data", dtype=dtype, max_chunk_size=1024*1024)

    # Generate appropriate data for dtype
    if dtype in ['float32', 'float64']:
        data = np.arange(num_samples, dtype=dtype).reshape(-1, 1)
    elif dtype in ['int32', 'int64', 'uint32', 'uint64']:
        data = np.arange(num_samples, dtype=dtype).reshape(-1, 1)
    elif dtype == 'uint8':
        data = (np.arange(num_samples) % 256).astype(dtype).reshape(-1, 1)
    elif dtype == 'bool':
        data = (np.arange(num_samples) % 2).astype(bool).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    ds.tensor_data.extend(data)
    return ds


def create_chunk_compressed_dataset(storage, num_samples=100):
    """Create dataset with ChunkCompressedChunk (chunk-wise compression)."""
    ds = muller.dataset(
        path=official_path(storage, TEST_NUMPY_PATH),
        creds=official_creds(storage),
        overwrite=True
    )
    ds.create_tensor("tensor_data", dtype="float64", chunk_compression="lz4")

    # Create fixed-shape data
    data = np.arange(num_samples, dtype=np.float64).reshape(-1, 1)
    ds.tensor_data.extend(data)

    return ds


def create_sample_compressed_dataset(storage, num_samples=10):
    """Create dataset with SampleCompressedChunk (sample-wise compression)."""
    ds = muller.dataset(
        path=official_path(storage, TEST_NUMPY_PATH),
        creds=official_creds(storage),
        overwrite=True
    )
    ds.create_tensor("images", htype="image", sample_compression="jpg")

    # Extend with sample images
    for _ in range(num_samples):
        ds.images.append(muller.read(SAMPLE_FILES["jpg_1"]))

    return ds


def assert_arrays_equal(result, expected, dtype):
    """Assert arrays are equal with dtype-appropriate comparison."""
    if dtype in ['float32', 'float64']:
        assert np.allclose(result, expected)
    elif dtype == 'bool':
        assert np.array_equal(result, expected)
    else:  # integer types
        assert np.array_equal(result, expected)


def generate_expected_data(dtype, num_samples):
    """Generate expected data for given dtype and number of samples."""
    if dtype in ['float32', 'float64']:
        return np.arange(num_samples, dtype=dtype).reshape(-1, 1)
    elif dtype in ['int32', 'int64', 'uint32', 'uint64']:
        return np.arange(num_samples, dtype=dtype).reshape(-1, 1)
    elif dtype == 'uint8':
        return (np.arange(num_samples) % 256).astype(dtype).reshape(-1, 1)
    elif dtype == 'bool':
        return (np.arange(num_samples) % 2).astype(bool).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def test_read_single_sample(storage):
    ds = create_dataset(storage)

    assert len(ds[3].numpy(aslist=True, asrow=True)) == 1
    assert ds[3].numpy(aslist=True, asrow=True)[0]["ori_query"] == "怎么将rgb三个通道的值合成一个rgb值"

    assert len(ds[3].numpy(aslist=True, asrow=False)) == 6
    assert ds[3].numpy(aslist=True, asrow=True)[0] == ds[3].numpy(aslist=True, asrow=False)


def test_read_dataset_slices(storage):
    ds = create_dataset(storage)

    assert len(ds[3:6].numpy(aslist=True, asrow=True)) == 3
    assert ds[3:6].numpy(aslist=True, asrow=True)[1]["ori_query"] == "给我生成一个游戏昵称"

    assert len(ds[3:6].numpy(aslist=True, asrow=False)) == 6
    assert len(ds[3:6].numpy(aslist=True, asrow=False)["ori_response"]) == 3
    assert ds[3:6].numpy(aslist=True, asrow=False)["result"][2] == "综合质量评分:6分"


def test_read_dataset_slices_with_step(storage):
    ds = create_dataset(storage)

    assert len(ds[3:6:2].numpy(aslist=True, asrow=True)) == 2
    assert ds[3:6:2].numpy(aslist=True, asrow=True)[1]["ori_query"] == "写一段基于深度学习的智能客服系统设计"

    assert len(ds[3:6:2].numpy(aslist=True, asrow=False)) == 6
    assert len(ds[3:6:2].numpy(aslist=True, asrow=False)["ori_response"]) == 2
    assert ds[3:6:2].numpy(aslist=True, asrow=False)["result"][0] == "综合质量评分:7分"


def test_tensors_with_different_length(storage):
    ds = create_dataset(storage)

    ds.ori_query.append("阅读这个页面的文本")
    ds.ori_response.append("当然，我可以帮助您阅读文本。您可以将页面的链接或者截图发给我，或者您可以直接在这个对话框中粘贴文本。请告诉我您需要我阅读的页面。")
    # without query_analysis
    ds.result.append("综合质量评分:7分")
    ds.score.append(7.0)
    ds.type.append(5.0)

    assert len(ds[5:].numpy(aslist=True, asrow=False)["result"]) == 2
    assert len(ds[5:].numpy(aslist=True, asrow=False)["query_analysis"]) == 1
    assert ds[5:].numpy(aslist=True, asrow=False)["ori_query"][0] == "写一段基于深度学习的智能客服系统设计"
    assert ds[5:].numpy(aslist=True, asrow=False)["ori_response"][1] == "当然，我可以帮助您阅读文本。您可以将页面的链接或者截图发给我，或者您可以直接在这个对话框中粘贴文本。请告诉我您需要我阅读的页面。"

    with pytest.raises(ValueError) as e:
        ds[5:].numpy(aslist=True, asrow=True)
    assert e.type == ValueError
    print(e.value)
    assert str(e.value) == "The number of samples in each tensor is different or the number not equal to the length of dataset index. Please set asrow = False."


def test_fixed_shape(storage):
    ds = create_dataset_with_fixed_shape(storage)
    assert ds.age[4].numpy() == [4]
    assert ds.age[17].numpy() == [17]
    assert ds.height[3].numpy() == [153.]
    assert ds.height[19].numpy() == [169.]
    assert ds.photo[0][0][0][0].numpy() == 243


def test_numpy_batch_random_access(storage):
    ds = create_text_dataset(storage)

    def old_random_access(ds, index_list):
        line_list = []
        for i in index_list:
            line = ds.value[i].numpy(fetch_chunks=True)
            line_list.append(line)
        return line_list

    def new_random_access(ds, index_list):
        line_list = []
        line_list.extend(ds.value.numpy_batch_random_access(index_list=index_list, parallel="threaded"))
        return line_list

    random_list = [random.randint(0, 9999) for _ in range(10)]

    list_1 = old_random_access(ds, index_list=random_list)
    list_2 = new_random_access(ds, index_list=random_list)

    assert list_1 == list_2


if __name__ == '__main__':
    pytest.main(["-s", "test_numpy.py"])


# ============================================================================
# Optimization Method Tests (from test_to_numpy.py)
# ============================================================================


class TestGetSamplesFull:
    """Test get_samples_full() method for full dataset access."""

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "uint8", "uint32", "bool"])
    def test_full_access_uncompressed_multiple_dtypes(self, storage, dtype):
        """Test full access on UncompressedChunk with various dtypes."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=50)

        result = ds.tensor_data[:].numpy()
        expected = generate_expected_data(dtype, 50)

        assert result.shape == expected.shape
        assert_arrays_equal(result, expected, dtype)

    def test_full_access_chunk_compressed(self, storage):
        """Test full access on ChunkCompressedChunk - should fallback to get_samples()."""
        ds = create_chunk_compressed_dataset(storage, num_samples=50)

        result = ds.tensor_data[:].numpy()
        expected = np.arange(50, dtype=np.float64).reshape(-1, 1)

        assert result.shape == expected.shape
        assert np.allclose(result, expected)

    def test_full_access_sample_compressed(self, storage):
        """Test full access on SampleCompressedChunk - should fallback to get_samples()."""
        ds = create_sample_compressed_dataset(storage, num_samples=5)

        result = ds.images[:].numpy()

        assert len(result) == 5
        assert result[0].shape[2] == 3  # RGB image

    def test_full_access_explicit_range(self, storage):
        """Test full access with explicit range [0:len(ds)]."""
        ds = create_uncompressed_dataset_with_dtype(storage, "float64", num_samples=30)

        result = ds.tensor_data[0:30].numpy()
        expected = np.arange(30, dtype=np.float64).reshape(-1, 1)

        assert result.shape == expected.shape
        assert np.allclose(result, expected)


class TestGetSamplesContinuous:
    """Test get_samples_continuous() method for continuous slice access."""

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "uint8", "uint32", "bool"])
    def test_continuous_slice_uncompressed_multiple_dtypes(self, storage, dtype):
        """Test continuous slice on UncompressedChunk with various dtypes."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        result = ds.tensor_data[20:50].numpy()
        expected = generate_expected_data(dtype, 100)[20:50]

        assert result.shape == expected.shape
        assert_arrays_equal(result, expected, dtype)

    def test_continuous_slice_chunk_compressed(self, storage):
        """Test continuous slice on ChunkCompressedChunk - should fallback to get_samples()."""
        ds = create_chunk_compressed_dataset(storage, num_samples=100)

        result = ds.tensor_data[20:50].numpy()
        expected = np.arange(20, 50, dtype=np.float64).reshape(-1, 1)

        assert result.shape == expected.shape
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
    def test_continuous_slice_beginning(self, storage, dtype):
        """Test continuous slice from beginning."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        result = ds.tensor_data[0:30].numpy()
        expected = generate_expected_data(dtype, 100)[0:30]

        assert result.shape == expected.shape
        assert_arrays_equal(result, expected, dtype)

    @pytest.mark.parametrize("dtype", ["float64", "int64", "uint32"])
    def test_continuous_slice_end(self, storage, dtype):
        """Test continuous slice to end."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        result = ds.tensor_data[70:100].numpy()
        expected = generate_expected_data(dtype, 100)[70:100]

        assert result.shape == expected.shape
        assert_arrays_equal(result, expected, dtype)

    @pytest.mark.parametrize("dtype", ["float32", "int32", "bool"])
    def test_continuous_slice_middle(self, storage, dtype):
        """Test continuous slice in middle."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        result = ds.tensor_data[40:60].numpy()
        expected = generate_expected_data(dtype, 100)[40:60]

        assert result.shape == expected.shape
        assert_arrays_equal(result, expected, dtype)


class TestGetSamplesBatchRandomAccess:
    """Test get_samples_batch_random_access() method for discrete index access."""

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "uint8", "uint32", "bool"])
    def test_batch_random_access_uncompressed_multiple_dtypes(self, storage, dtype):
        """Test batch random access on UncompressedChunk with various dtypes."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        indices = [5, 15, 25, 35, 45]
        result = ds.tensor_data[indices].numpy()
        expected = generate_expected_data(dtype, 100)[indices]

        assert result.shape == expected.shape
        assert_arrays_equal(result, expected, dtype)

    def test_batch_random_access_chunk_compressed(self, storage):
        """Test batch random access on ChunkCompressedChunk - should fallback to get_samples()."""
        ds = create_chunk_compressed_dataset(storage, num_samples=100)

        indices = [5, 15, 25, 35, 45]
        result = ds.tensor_data[indices].numpy()
        expected = np.array(indices, dtype=np.float64).reshape(-1, 1)

        assert result.shape == expected.shape
        assert np.allclose(result, expected)

    def test_batch_random_access_sample_compressed(self, storage):
        """Test batch random access on SampleCompressedChunk - should fallback to get_samples()."""
        ds = create_sample_compressed_dataset(storage, num_samples=10)

        indices = [1, 3, 5, 7]
        result = ds.images[indices].numpy()

        assert len(result) == 4
        assert result[0].shape[2] == 3  # RGB image

    @pytest.mark.parametrize("dtype", ["float64", "int32", "uint8"])
    def test_batch_random_access_unordered(self, storage, dtype):
        """Test batch random access with unordered indices."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        indices = [50, 10, 80, 20, 60]
        result = ds.tensor_data[indices].numpy()
        expected = generate_expected_data(dtype, 100)[indices]

        assert result.shape == expected.shape
        assert_arrays_equal(result, expected, dtype)

    @pytest.mark.parametrize("dtype", ["float32", "int64", "bool"])
    def test_batch_random_access_duplicates(self, storage, dtype):
        """Test batch random access with duplicate indices."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        indices = [10, 20, 10, 30, 20]
        result = ds.tensor_data[indices].numpy()
        expected = generate_expected_data(dtype, 100)[indices]

        assert result.shape == expected.shape
        assert_arrays_equal(result, expected, dtype)


class TestGetSamples:
    """Test get_samples() method (default fallback)."""

    @pytest.mark.parametrize("dtype", ["float64", "int32", "uint8"])
    def test_get_samples_with_step(self, storage, dtype):
        """Test get_samples with step parameter (cannot use optimization)."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        result = ds.tensor_data[10:50:5].numpy()
        expected = generate_expected_data(dtype, 100)[10:50:5]

        assert result.shape == expected.shape
        assert_arrays_equal(result, expected, dtype)

    @pytest.mark.parametrize("dtype", ["float32", "int64", "bool"])
    def test_get_samples_single_index(self, storage, dtype):
        """Test get_samples with single index."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        result = ds.tensor_data[42].numpy()
        expected = generate_expected_data(dtype, 100)[42]

        assert_arrays_equal(result, expected, dtype)

    def test_get_samples_negative_index(self, storage):
        """Test get_samples with negative index."""
        ds = create_uncompressed_dataset_with_dtype(storage, "float64", num_samples=100)

        result = ds.tensor_data[-1].numpy()
        expected = np.array([99.0])

        assert np.allclose(result, expected)

    def test_get_samples_negative_slice(self, storage):
        """Test get_samples with negative slice."""
        ds = create_uncompressed_dataset_with_dtype(storage, "float64", num_samples=100)

        result = ds.tensor_data[-10:].numpy()
        expected = np.arange(90, 100, dtype=np.float64).reshape(-1, 1)

        assert result.shape == expected.shape
        assert np.allclose(result, expected)


class TestAutoDetection:
    """Test automatic detection of access patterns."""

    @pytest.mark.parametrize("dtype", ["float64", "int32", "uint8"])
    def test_auto_detect_full_access(self, storage, dtype):
        """Test that full access is automatically detected."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=50)

        result1 = ds.tensor_data[:].numpy()
        result2 = ds.tensor_data[0:50].numpy()

        assert_arrays_equal(result1, result2, dtype)

    @pytest.mark.parametrize("dtype", ["float32", "int64", "bool"])
    def test_auto_detect_continuous_access(self, storage, dtype):
        """Test that continuous access is automatically detected."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        result = ds.tensor_data[20:40].numpy()
        expected = generate_expected_data(dtype, 100)[20:40]

        assert_arrays_equal(result, expected, dtype)

    @pytest.mark.parametrize("dtype", ["float64", "int32", "uint32"])
    def test_auto_detect_batch_random_access(self, storage, dtype):
        """Test that batch random access is automatically detected."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        indices = [10, 20, 30, 40]
        result = ds.tensor_data[indices].numpy()
        expected = generate_expected_data(dtype, 100)[indices]

        assert_arrays_equal(result, expected, dtype)

    def test_fallback_to_get_samples(self, storage):
        """Test fallback to get_samples when optimization not applicable."""
        ds = create_uncompressed_dataset_with_dtype(storage, "float64", num_samples=100)

        result = ds.tensor_data[0:50:2].numpy()
        expected = np.arange(0, 50, 2, dtype=np.float64).reshape(-1, 1)

        assert np.allclose(result, expected)


class TestExplicitModeSpecification:
    """Test explicit specification of access modes."""

    def test_explicit_continuous_mode(self, storage):
        """Test explicitly specifying continuous mode."""
        ds = create_uncompressed_dataset_with_dtype(storage, "float64", num_samples=100)

        result = ds.tensor_data[20:50].numpy_continuous()
        expected = np.arange(20, 50, dtype=np.float64).reshape(-1, 1)

        assert np.allclose(result, expected)

    def test_explicit_full_mode(self, storage):
        """Test explicitly specifying full mode."""
        ds = create_uncompressed_dataset_with_dtype(storage, "float64", num_samples=50)

        result = ds.tensor_data[:].numpy_full()
        expected = np.arange(50, dtype=np.float64).reshape(-1, 1)

        assert np.allclose(result, expected)

    def test_explicit_batch_random_access_mode(self, storage):
        """Test explicitly specifying batch_random_access mode."""
        ds = create_uncompressed_dataset_with_dtype(storage, "float64", num_samples=100)

        indices = [5, 15, 25, 35]
        result = ds.tensor_data.numpy_batch_random_access(index_list=indices, parallel='threaded')
        expected = np.array(indices, dtype=np.float64).reshape(-1, 1)

        assert np.allclose(result, expected)

    def test_auto_vs_explicit_continuous(self, storage):
        """Test that auto-detected continuous access matches explicit continuous mode."""
        ds = create_uncompressed_dataset_with_dtype(storage, "float64", num_samples=100)

        result_auto = ds.tensor_data[20:50].numpy()
        result_explicit = ds.tensor_data[20:50].numpy_continuous()

        assert np.allclose(result_auto, result_explicit)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_slice(self, storage):
        """Test empty slice."""
        ds = create_uncompressed_dataset_with_dtype(storage, "float64", num_samples=100)

        result = ds.tensor_data[50:50].numpy()

        assert result.shape[0] == 0

    def test_single_sample_dataset(self, storage):
        """Test dataset with single sample."""
        ds = create_uncompressed_dataset_with_dtype(storage, "float64", num_samples=1)

        result = ds.tensor_data[:].numpy()
        expected = np.array([0.0]).reshape(-1, 1)

        assert np.allclose(result, expected)

    @pytest.mark.parametrize("dtype", ["float64", "int32"])
    def test_large_continuous_slice(self, storage, dtype):
        """Test large continuous slice."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=1000)

        result = ds.tensor_data[100:900].numpy()
        expected = generate_expected_data(dtype, 1000)[100:900]

        assert result.shape == expected.shape
        assert_arrays_equal(result, expected, dtype)

    def test_random_indices_large_dataset(self, storage):
        """Test random indices on large dataset."""
        ds = create_uncompressed_dataset_with_dtype(storage, "float64", num_samples=1000)

        random.seed(42)
        indices = sorted(random.sample(range(1000), 50))
        result = ds.tensor_data[indices].numpy()
        expected = np.array(indices, dtype=np.float64).reshape(-1, 1)

        assert result.shape == expected.shape
        assert np.allclose(result, expected)


class TestMultipleChunkTypes:
    """Test with datasets containing multiple chunk types."""

    def test_mixed_chunk_types_in_dataset(self, storage):
        """Test dataset with multiple tensors of different chunk types."""
        ds = muller.dataset(
            path=official_path(storage, TEST_NUMPY_PATH),
            creds=official_creds(storage),
            overwrite=True
        )

        # UncompressedChunk
        ds.create_tensor("uncompressed", dtype="float64")
        ds.uncompressed.extend(np.arange(50, dtype=np.float64).reshape(-1, 1))

        # ChunkCompressedChunk
        ds.create_tensor("chunk_compressed", dtype="float64", chunk_compression="lz4")
        ds.chunk_compressed.extend(np.arange(50, dtype=np.float64).reshape(-1, 1))

        # Test full access on both
        result_uncompressed = ds.uncompressed[:].numpy()
        result_compressed = ds.chunk_compressed[:].numpy()

        assert np.allclose(result_uncompressed, result_compressed)

        # Test continuous slice on both
        result_uncompressed = ds.uncompressed[10:30].numpy()
        result_compressed = ds.chunk_compressed[10:30].numpy()

        assert np.allclose(result_uncompressed, result_compressed)

        # Test batch random access on both
        indices = [5, 15, 25, 35]
        result_uncompressed = ds.uncompressed[indices].numpy()
        result_compressed = ds.chunk_compressed[indices].numpy()

        assert np.allclose(result_uncompressed, result_compressed)


class TestPerformanceComparison:
    """Test consistency between auto-detected and explicit optimization methods."""

    @pytest.mark.parametrize("dtype", ["float64", "int32", "uint8"])
    def test_consistency_full_access(self, storage, dtype):
        """Test that auto-detected and explicit full access return same results."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        # Auto-detected
        result_auto = ds.tensor_data[:].numpy()

        # Explicit
        result_explicit = ds.tensor_data[:].numpy_full()

        assert_arrays_equal(result_auto, result_explicit, dtype)

    @pytest.mark.parametrize("dtype", ["float32", "int64", "bool"])
    def test_consistency_continuous_access(self, storage, dtype):
        """Test that auto-detected and explicit continuous access return same results."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        # Auto-detected
        result_auto = ds.tensor_data[20:60].numpy()

        # Explicit
        result_explicit = ds.tensor_data[20:60].numpy_continuous()

        assert_arrays_equal(result_auto, result_explicit, dtype)

    @pytest.mark.parametrize("dtype", ["float64", "int32", "uint32"])
    def test_consistency_batch_random_access(self, storage, dtype):
        """Test that auto-detected and explicit batch random access return same results."""
        ds = create_uncompressed_dataset_with_dtype(storage, dtype, num_samples=100)

        indices = [10, 20, 30, 40, 50]

        # Auto-detected
        result_auto = ds.tensor_data[indices].numpy()

        # Explicit
        result_explicit = ds.tensor_data.numpy_batch_random_access(index_list=indices, parallel='threaded')

        assert_arrays_equal(result_auto, result_explicit, dtype)

