# API Cheatsheet

## Dataset Creation

```python
# Create/open dataset
ds = muller.dataset(path, overwrite=False, read_only=False)

# Load existing dataset
ds = muller.load(path, read_only=False)

# Create empty dataset
ds = muller.empty(path, overwrite=False)

# Delete dataset
muller.delete(path, large_ok=False)

# Copy structure
muller.like(dest, src, tensors=None, overwrite=False)
```

## Tensor Management

```python
# Create tensor
ds.create_tensor(name, htype="generic", dtype=None, sample_compression=None)

# Delete tensor
ds.delete_tensor(name, large_ok=False)

# Rename tensor
ds.rename_tensor(old_name, new_name)

# Copy tensor structure
ds.create_tensor_like(name, source_tensor)
```

## Data Operations

```python
# Append single sample
with ds:
    ds.append({"tensor1": value1, "tensor2": value2})

# Extend with multiple samples
with ds:
    ds.extend({"tensor1": [v1, v2], "tensor2": [v3, v4]})

# Update sample
with ds:
    ds[index].update({"tensor1": new_value})

# Delete samples
with ds:
    ds.pop(index)  # Single index
    ds.pop([0, 5, 10])  # Multiple indices
```

## Query Operations

```python
# Filter samples
result = ds.filter_vectorized([("tensor", "op", value)])

# Operators: ==, !=, >, <, >=, <=, CONTAINS
result = ds.filter_vectorized([("labels", ">", 5)])
result = ds.filter_vectorized([("text", "CONTAINS", "hello")])

# Multiple conditions
result = ds.filter_vectorized(
    [("labels", ">", 5), ("labels", "<", 10)],
    ["AND"]
)

# Limit results
result = ds.filter_vectorized([("labels", "==", 1)], limit=10)
```

## Dataset Info

```python
# Summary
ds.summary()

# Statistics
stats = ds.statistics()

# Sample count
count = ds.num_samples

# Tensor list
tensors = ds.tensors

# Dataset info
info = ds.info
```

## Import Data

```python
# From JSONL file
ds = muller.from_file(ori_path, muller_path, schema=None)

# From dataframes (list of dicts)
ds = muller.from_dataframes(dataframes, muller_path, schema=None)

# Schema format
schema = {
    "tensor_name": (htype, dtype, compression),
    "images": ("image", "uint8", "jpg"),
    "labels": ("class_label", "uint32", None)
}
```

## Common Parameters

- `path` - Dataset path (local or s3://)
- `overwrite` - Replace existing dataset
- `read_only` - Open in read-only mode
- `creds` - Credentials for S3/OBS
- `htype` - Data type (image, text, vector, etc.)
- `dtype` - NumPy dtype (int32, float32, etc.)
- `sample_compression` - Compression format (jpg, lz4, etc.)

## Return Types

- `Dataset` - Dataset object
- `Tensor` - Tensor object
- `dict` - Statistics, info
- `int` - Sample count, size
- `None` - Operations with side effects
