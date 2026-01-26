# Creating a MULLER Dataset

## 1. Manually Loading Data and Creating a MULLER Dataset

### Step 1. Create an Empty MULLER Dataset

A MULLER dataset is represented as a **directory**, whose name may consist of letters, digits, underscores (`_`), and hyphens (`-`).
Assume we would like to create a dataset named `muller_dataset`.  
An empty MULLER dataset can be created on different storage backends via the `path` argument, as shown below.

**Example 1: Local Storage**  
(The MULLER dataset is stored on the local filesystem.)

No prefix is required for local paths. Both relative and absolute paths are supported.
> Note: This approach is not recommended in non-local environments. File I/O operations rely directly on native libraries such as `os`, `pathlib`, and `shutil`, which may lead to lower performance.

```python
>>> ds_1 = muller.empty(path="muller_dataset/")
>>> ds_2 = muller.empty(path="./my_data/muller_dataset/")
```
**Example 2: S3 Object Storage**
(The MULLER dataset is stored in an S3-compatible service.)

Use the `s3://` prefix to specify the S3 path. Credentials must be provided.
> Note: File I/O operations in this mode are implemented via the S3 client interface using boto3, which generally provides higher performance. For non-local storage backends (e.g., S3), I/O performance is highly dependent on available network bandwidth. In general, private cloud environments offer lower latency and higher bandwidth due to dedicated compute, storage, and network resources, resulting in performance close to local disk access. In contrast, bandwidth contention may significantly impact I/O throughput in shared environments. Under bandwidth constraints, MULLER workloads may become I/O-bound.
**It is therefore strongly recommended to reserve sufficient network bandwidth for MULLER operations.**

```python
>>> endpoint = "http://x.x.x.x:xxxx"  # Example only; use a valid endpoint
>>> ak = "xxx"                              # Example only; use a valid access key
>>> sk = "yyy"                              # Example only; use a valid secret key
>>> bucket_name = "zzz"              # Example only; use a valid bucket name
>>> creds = {"bucket_name": bucket_name, "endpoint": endpoint, "ak": ak, "sk": sk}
>>> ds_3 = muller.empty(path="s3://muller_dataset", creds=creds)
```

In addition to `muller.empty()`, you may also use `muller.dataset()` to create a dataset.
The usage is similar to the examples above; simply add the appropriate path prefix according to the storage backend.

**Example 3: Local Storage**
```python
>>> muller.dataset(path="./my_data/muller_dataset/")
```

**Example 4: S3 Object Storage**
```python
>>> muller.dataset(path="s3://muller_dataset", creds=creds)
```
* If the path specified by `path` already contains an existing MULLER dataset, a `DatasetHandlerError` will be raised. You may set `overwrite=True` to explicitly overwrite the existing dataset.
* For additional optional arguments and advanced usage of these APIs, please refer to the API documentation.

### Step 2. Create Tensor Columns and Specify Column Types

In a MULLER dataset, columns are referred to as **tensor columns**.  
MULLER adopts a column-oriented storage layout, where each tensor column can be configured with a specific column type (`htype`), sample compression format (`sample_compression`), and data type (`dtype`).

For most column types, specifying an appropriate compression format can significantly reduce storage footprint and improve read performance.

```python
>>> ds = muller.dataset(path='my_muller_dataset/', overwrite=True)  # Create a new MULLER dataset
>>> ds.create_tensor(name='my_text', htype='text')
>>> ds.create_tensor(name='my_photos', htype='image', sample_compression='jpg')
>>> ds.summary()  # Inspect the dataset schema
  tensor      htype     shape     dtype   compression
  -------    -------   -------   -------  -----------
  my_text     text      (0,)      str      None
  my_photos   image     (0,)      uint8    jpeg
```
For the `generic` column type, the data type (`dtype`) can be explicitly specified (e.g., `int16`, `int32`, `float32`, `bytes`, `bool`).
For other column types, the `dtype` is predefined and does not need to be specified.

```python
>>> ds.create_tensor(name='my_label', htype='generic', dtype='int32')
>>> ds.summary()
  tensor      htype     shape     dtype   compression
  -------    -------   -------   -------  -----------
  my_text     text      (0,)      str      None
  my_photos   image     (0,)      uint8    jpeg
  my_label    generic   (0,)      int32    None
```
Currently, MULLER supports the following tensor column types (`htype`), compression formats (`sample_compression`), and default data types (`dtype`).

| htype | sample_compression | dtype |
| --- | --- | --- |
| image | Required (one of): bmp, dib, gif, ico, jpg, jpeg, <br>jpeg2000, pcx, png, ppm, sgi, tga, tiff, <br>webp, wmf, xbm, eps, fli, im, msp, mpo | Default: `uint8` (modification not recommended) |
| video | Required (one of): mp4, mkv, avi | Default: `uint8` (modification not recommended) |
| audio | Required (one of): flac, mp3, wav | Default: `float64` (modification not recommended) |
| class_label | Default: None (null); Optional: lz4 | Default: `uint32` (modification not recommended) |
| bbox | Default: None (null); Optional: lz4 | Default: `float32` (modification not recommended) |
| text  | Default: None (null); Optional: lz4 | Default: `str` (modification not recommended) |
| json  | Default: None (null); Optional: lz4 | - |
| list  | Default: None (null); Optional: lz4 | - |
| vector  | Default: None (null); Optional: lz4 | Default:  `float32` |
| generic  | Default: None (null); Optional: lz4 | Default: None (undeclared, inferred from data); Declaration at creation is recommended.<br>Options: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, <br>`uint32`, `float32`, `float64`, `bool` |

- **Note 1:** If `htype` is not specified when creating a tensor column, it defaults to `generic`.
- **Note 2:** For `class_label`, `bbox`, `text`, `json`, `list`, and `generic` columns, it is recommended to leave `sample_compression` as `None` unless storage savings are critical. Using `lz4` introduces additional compression and decompression overhead, which may negatively impact read and write performance.
- **Note 3:** In addition to `htype`, `sample_compression`, and `dtype`, additional parameters can be specified when creating tensor columns for advanced use cases.  
  For example, `chunk_compression` can be configured (see [chunk_compression](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405203579308)), and `bbox` columns may accept extra parameters such as `coords` (see [BoundingBox Htype](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405203579194)).  
  For the complete list of supported column types and parameters, please refer to [Htypes](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405203578912) and its subpages.
- **Note 4:** Tensor column names must not contain any MULLER reserved keywords or any Python reserved keywords.
  ```
  keyword_list_1 = ['__bool__', '__class__', '__del__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_all_tensors_filtered', '_append_or_extend', '_append_to_queries_json', '_check_values', '_checkout', '_checkout_hooks', '_client', '_commit', '_commit_hooks', '_copy', '_create_downsampled_tensor', '_create_sample_id_tensor', '_create_sample_info_tensor', '_create_sample_shape_tensor', '_create_tensor', '_dataset_diff', '_delete_branch', '_delete_tensor', '_deserialize_uuids', '_disable_padding', '_ds_diff', '_enable_padding', '_find_subtree', '_first_load_init', '_flush_vc_info', '_get_chunk_names', '_get_commit_id_for_address', '_get_empty_vds', '_get_inverted_index', '_get_tensor_from_root', '_get_tensor_uuids', '_get_view', '_get_view_info', '_groups', '_groups_filtered', '_has_group_in_root', '_indexing_history', '_info', '_initial_autoflush', '_inverted_index', '_is_filtered_view', '_is_root', '_link_tensors', '_load_link_creds', '_load_version_info', '_lock', '_lock_lost_handler', '_lock_queries_json', '_lock_timeout', '_locked_out', '_locking_enabled', '_pad_tensors', '_parent_dataset', '_pop', '_populate_meta', '_query_string', '_read_from_upper_cache', '_read_only', '_read_only_error', '_read_queries_json', '_register_dataset', '_reload_version_state', '_resolve_tensor_list', '_sample_indices', '_save_view', '_save_view_in_path', '_save_view_in_subdir', '_send_branch_creation_event', '_send_branch_deletion_event', '_send_commit_event', '_send_compute_progress', '_send_query_progress', '_set_derived_attributes', '_set_read_only', '_sub_ds', '_subtree_to_dict', '_temp_tensors', '_tensors', '_token', '_ungrouped_tensors', '_unlock', '_update_hooks', '_update_upper_cache', '_vc_info_updated', '_view_base', '_view_hash', '_view_id', '_view_use_parent_commit', '_write_queries_json', '_write_vds', 'add_data', 'add_data_from_dataframes', 'add_data_from_file', 'aggregate', 'append', 'base_storage', 'branch', 'branches', 'checkout', 'client', 'commit', 'commit_id', 'commits', 'commits_between', 'create_index', 'create_tensor', 'create_tensor_like', 'delete', 'delete_branch', 'delete_tensor', 'delete_view', 'diff', 'ds_name', 'enabled_tensors', 'extend', 'filter', 'filter_next', 'filtered_index', 'flush', 'generate_add_update_value', 'get_children_nodes', 'get_commit_details', 'get_view', 'get_views', 'group_index', 'groups', 'has_head_changes', 'index', 'indexed_tensors', 'info', 'is_first_load', 'is_head_node', 'is_iteration', 'is_optimized', 'is_view', 'libgtnf_dataset', 'link_creds', 'load_view', 'log', 'max_len', 'max_view', 'maybe_flush', 'merge', 'meta', 'min_len', 'min_view', 'no_view_dataset', 'num_samples', 'numpy', 'org_id', 'pad_tensors', 'parent', 'parse_changes', 'path', 'pending_commit_id', 'pop', 'public', 'query', 'read_only', 'rechunk', 'rename', 'reset', 'root', 'sample_indices', 'save_as_branch', 'save_view', 'set_token', 'size_approx', 'split_tensor_meta', 'statistics', 'storage', 'summary', 'tensors', 'to_arrow', 'to_json', 'to_mindrecord', 'token', 'update', 'username', 'verbose', 'version_state', 'write_to_parquet']
  keyword_list_2 = ['__class__', '__class_getitem__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__ior__', '__iter__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__or__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__ror__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values']
  attr_and_keyword_list_3 = ['tensors', 'data', 'all_chunk_engines', 'group_index', 'cache_size', 'cache_used' 'idx',  'pg_callback', 'start_input_idx', '_client', 'path', 'storage', '_read_only_error', 'base_storage', '_read_only',  '_locked_out', 'is_iteration', 'is_first_load', '_is_filtered_view', 'index', 'group_index', 'version_state', 'pad_tensors', '_locking_enabled', '_lock_timeout', '_temp_tensors', '_commit_hooks', '_checkout_hooks', 'public', 'verbose', '_vc_info_updated', '_info', '_ds_diff', 'enabled_tensors', 'link_creds', '_update_hooks', '_view_id', '_view_base', '_view_use_parent_commit', '_pad_tensors', 'libgtnf_dataset', '_parent_dataset', '_query_string', '_inverted_index', 'filtered_index', 'split_tensor_meta', 'creds', '_vector_index', 'append_only', '_initial_autoflush', '_indexing_history', 'read_only', 'is_first_load']
  ```
- For more detailed usage, please refer to [`create_tensor()`](https:...). We plan to support additional `htype`s and provide clearer guidelines for tensor column creation in future releases. This documentation will be updated accordingly.

### Step 3. Append Data to Tensor Columns

#### 1. Appending a small number of samples (recommended: default single-process mode)

**Append a single sample:**

```python
>>> import numpy as np
>>> with ds:  # Using the `with` context is critical for improving write performance.
...     # Append a file by calling `muller.read()`.
...     # Here we use the `my_photos` column (htype=image, sample_compression=jpg) as an example.
...     # A JPEG image file can be appended directly.
...     ds.my_photos.append(muller.read("path/to/file/image.jpg"))
...     # Alternatively, a NumPy tensor representation can be appended, as long as it
...     # matches the expected image tensor format (H × W × RGB).
...     ds.my_photos.append(np.ones((400, 300, 3), dtype=np.uint8))
```

**Append multiple samples:**

```python
>>> with ds:
...     ds.my_photos.extend([
...         muller.read("path/to/file/image.jpg"),
...         muller.read("path/to/file/image.jpg")
...     ])
...     ds.my_text.extend(["cat", "dog", "tree", "car"])
...     ds.my_label.extend(np.array([1, 2, 3, 4]))

>>> ds.summary()
  tensor      htype            shape              dtype   compression
  -------    -------          -------            -------  -----------
  my_text     text             (4, 1)              str     None
  my_photos   image      (4, 400, 300:500, 3)     uint8    jpeg
  my_label    generic          (4, 1)             int32    None
```
* Refer to [[append()]()] and [[extend()] for detailed usage.

#### 2. Use the `with ds:` context to improve write performance
Always wrap dataset write operations inside `with ds:`, as this can significantly improve data write throughput.
For a detailed explanation, see [[Using the with context to improve write performance]()].

#### 3. Appending large-scale data (parallel mode)
For large-scale data ingestion, it is recommended to use the `@muller.compute` decorator to enable parallel execution (multi-threading, multi-processing, or multi-worker setups).
You can also enable periodic checkpointing by setting `checkpoint_interval=<commit_every_N_samples>` to persist data in batches.

See [[Creating datasets and appending large-scale data]()] for details.

##### 4. Data consistency requirement
**Important**: When appending data column-wise, ensure that the number of newly added samples is identical across all columns.
This guarantees dataset consistency and prevents mismatched column lengths.
