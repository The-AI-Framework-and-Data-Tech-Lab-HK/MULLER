# Metadata in MULLER

MULLER uses two types of metadata files to manage datasets and tensors: `dataset_meta.json` for dataset-level information and `tensor_meta.json` for individual tensor configurations.

## Dataset Metadata (`dataset_meta.json`)

The `dataset_meta.json` file stores dataset-level metadata and is located at the root of each dataset directory and in version subdirectories.

### Structure

```json
{
    "dataset_creator": "public",
    "hidden_tensors": ["_uuid"],
    "info": {},
    "tensor_names": {
        "_uuid": "_uuid",
        "images": "images",
        "labels": "labels"
    },
    "tensors": ["_uuid", "images", "labels"],
    "version": "0.7.0"
}
```

### Fields

#### `dataset_creator` (string)
The username of the dataset creator. Defaults to `"public"`.

```python
ds = muller.dataset("my_dataset")
print(ds.meta.dataset_creator)  # "public"
```

#### `hidden_tensors` (list)
List of tensor keys that should be hidden from normal operations. Hidden tensors are typically system-internal tensors like `_uuid`.

```python
# Hidden tensors are not included in ds.tensors
visible = ds.tensors  # Does not include _uuid
all_tensors = ds.get_tensors(include_hidden=True)  # Includes _uuid
```

#### `info` (dict)
Optional metadata dictionary used primarily for virtual datasets and views. For regular datasets, this is typically empty. Virtual datasets use this to store query metadata such as `source-dataset`, `query`, and `id`.

```python
# For virtual datasets
vds = ds[:100].save_view(view_id="sample_view")
print(vds.meta.info)  # Contains view-specific metadata
```

#### `tensor_names` (dict)
Maps user-facing tensor names to internal tensor keys. This mapping is essential for tensor rename operations where the name changes but the key remains constant.

```python
ds.create_tensor("labels")
# tensor_names: {"labels": "labels"}

ds.rename_tensor("labels", "new_labels")
# tensor_names: {"new_labels": "labels"}
# Key stays the same, only name changes
```

#### `tensors` (list)
List of all tensor keys in the dataset. Used for version control operations and merge conflict detection.

```python
ds.create_tensor("images")
ds.create_tensor("labels")
print(ds.meta.tensors)  # ["_uuid", "images", "labels"]
```

#### `version` (string)
The MULLER version used to create the dataset.

---

## Tensor Metadata (`tensor_meta.json`)

Each tensor has its own `tensor_meta.json` file stored in `{tensor_key}/tensor_meta.json`. This file contains tensor-specific configuration and statistics.

### Structure

```json
{
    "chunk_compression": null,
    "dtype": "int64",
    "hidden": false,
    "htype": "generic",
    "length": 100,
    "max_chunk_size": null,
    "max_shape": [10, 224, 224, 3],
    "min_shape": [1, 224, 224, 3],
    "sample_compression": null,
    "tiling_threshold": null,
    "typestr": "<i8",
    "version": "0.7.0"
}
```

### Fields

#### `htype` (string)
The high-level data type of the tensor. Determines default compression and validation behavior.

Common htypes:
- `"generic"`: General-purpose numeric data
- `"text"`: Text data
- `"image"`: Image data
- `"class_label"`: Classification labels
- `"bbox"`: Bounding boxes

```python
ds.create_tensor("images", htype="image", sample_compression="jpg")
ds.create_tensor("labels", htype="class_label", dtype="int32")
```

#### `dtype` (string)
The numpy data type name (e.g., `"int64"`, `"float32"`, `"str"`).

```python
ds.create_tensor("scores", htype="generic", dtype="float32")
print(ds.scores.meta.dtype)  # "float32"
```

#### `typestr` (string)
The numpy dtype string representation (e.g., `"<i8"` for little-endian int64). Used for precise dtype reconstruction.

#### `sample_compression` (string or null)
Compression applied to individual samples. Mutually exclusive with `chunk_compression`.

```python
ds.create_tensor("images", htype="image", sample_compression="jpg")
# Each image is stored as a compressed JPEG
```

#### `chunk_compression` (string or null)
Compression applied to entire chunks of data. Mutually exclusive with `sample_compression`.

```python
ds.create_tensor("embeddings", htype="generic", chunk_compression="lz4")
# Chunks of embeddings are compressed together
```

#### `min_shape` / `max_shape` (list)
Track the minimum and maximum shapes encountered across all samples. Updated automatically as samples are added.

```python
ds.create_tensor("images", htype="image")
ds.images.append(np.zeros((224, 224, 3)))
ds.images.append(np.zeros((512, 512, 3)))

print(ds.images.meta.min_shape)  # [224, 224, 3]
print(ds.images.meta.max_shape)  # [512, 512, 3]
```

#### `length` (integer)
The total number of samples in the tensor.

```python
ds.create_tensor("data", htype="generic")
ds.data.extend([1, 2, 3, 4, 5])
print(ds.data.meta.length)  # 5
```

#### `hidden` (boolean)
Whether the tensor is hidden from normal operations. Hidden tensors are used for internal bookkeeping.

```python
# System tensors like _uuid are hidden
print(ds["_uuid"].meta.hidden)  # True

# User tensors are visible
print(ds["images"].meta.hidden)  # False
```

#### `max_chunk_size` (integer or null)
Maximum size in bytes for a single chunk. Controls how data is split into chunks for storage.

#### `tiling_threshold` (integer or null)
Threshold for tiling large samples. Samples exceeding this size are split into tiles.

#### `version` (string)
The MULLER version used to create the tensor.

---

## Backward Compatibility

MULLER maintains backward compatibility with older metadata formats:

- **Deprecated fields** (`statistics`, `default_index` in dataset_meta; `name`, `verify` in tensor_meta) are automatically ignored when loading older datasets
- Older datasets can be read and updated without migration
- New datasets use the optimized metadata format

```python
# Loading an old dataset with deprecated fields works seamlessly
old_ds = muller.dataset("path/to/old/dataset")
# Deprecated fields are silently ignored
```

---

## Best Practices

1. **Don't manually edit metadata files** - Use MULLER's API to modify datasets and tensors
2. **Use appropriate htypes** - Choose the correct htype for your data to get optimal compression and validation
3. **Monitor shape ranges** - Check `min_shape` and `max_shape` to ensure data consistency
4. **Leverage hidden tensors** - Use hidden tensors for internal bookkeeping without cluttering the user-facing API

```python
# Good: Use the API
ds.create_tensor("data", htype="generic", dtype="float32")
ds.data.extend([1.0, 2.0, 3.0])

# Bad: Don't manually edit JSON files
# Editing tensor_meta.json directly can corrupt the dataset
```
