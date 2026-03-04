# Dataset Creation and Management

This page documents functions for creating, loading, and managing datasets.

## Table of Contents

- [dataset()](#dataset)
- [load()](#load)
- [empty()](#empty)
- [like()](#like)
- [delete()](#delete)
- [get_col_info()](#get_col_info)
- [from_file()](#from_file)
- [from_dataframes()](#from_dataframes)

---

## dataset()

### Overview

Returns a `Dataset` object referencing either a new or existing dataset. This is the primary function for creating or opening datasets.

### Parameters

- **path** (`str` or `pathlib.Path`): The full path to the dataset. Can be:
  - An s3 path of the form `s3://bucketname/path/to/dataset`
  - A local file system path of the form `./path/to/dataset`, `~/path/to/dataset` or `path/to/dataset`
- **read_only** (`bool`, optional): Opens dataset in read only mode if `True`. Defaults to `False`.
- **overwrite** (`bool`, optional): If `True`, overwrites the dataset if it already exists. Defaults to `False`.
- **memory_cache_size** (`int`, optional): The size of the memory cache to be used in MB. Defaults to `DEFAULT_MEMORY_CACHE_SIZE`.
- **local_cache_size** (`int`, optional): The size of the local filesystem cache to be used in MB. Defaults to `DEFAULT_LOCAL_CACHE_SIZE`.
- **creds** (`dict` or `str`, optional): Credentials for OBS service. Defaults to `None`.
- **verbose** (`bool`, optional): If `True`, logs will be printed. Defaults to `True`.
- **reset** (`bool`, optional): If the specified dataset cannot be loaded due to a corrupted HEAD state, setting `reset=True` will reset HEAD changes and load the previous version. Defaults to `False`.
- **check_integrity** (`bool`, optional): Performs an integrity check by default if the dataset has 20 or fewer tensors. Defaults to `True`.
- **lock_timeout** (`int`, optional): Number of seconds to wait before throwing a LockException. Defaults to `0`.
- **lock_enabled** (`bool`, optional): If `True`, the dataset manages a write lock. Defaults to `True`.
- **split_tensor_meta** (`bool`, optional): Each tensor has a tensor_meta.json if `True`. Defaults to `True`.

### Returns

- **Dataset**: The dataset object.

### Examples

```python
import muller

# Create a new local dataset
ds = muller.dataset("./datasets/my_dataset", overwrite=True)

# Open an existing dataset
ds = muller.dataset("./datasets/my_dataset")

# Create a dataset on remote storage
ds = muller.dataset("s3://mybucket/my_dataset", creds={"aws_access_key_id": "...", "aws_secret_access_key": "..."})

# Open in read-only mode
ds = muller.dataset("./datasets/my_dataset", read_only=True)

# Create with custom cache sizes
ds = muller.dataset("./datasets/my_dataset", memory_cache_size=512, local_cache_size=2048)
```

---

## load()

### Overview

Load an existing dataset from the given path. Unlike `dataset()`, this function will raise an error if the dataset does not exist.

### Parameters

- **path** (`str` or `pathlib.Path`): The full path to the dataset.
- **read_only** (`bool`, optional): Opens dataset in read only mode if `True`. Defaults to `False`.
- **memory_cache_size** (`int`, optional): The size of the memory cache to be used in MB. Defaults to `DEFAULT_MEMORY_CACHE_SIZE`.
- **local_cache_size** (`int`, optional): The size of the local filesystem cache to be used in MB. Defaults to `DEFAULT_LOCAL_CACHE_SIZE`.
- **creds** (`dict` or `str`, optional): Credentials for OBS service. Defaults to `None`.
- **verbose** (`bool`, optional): If `True`, logs will be printed. Defaults to `True`.
- **check_integrity** (`bool`, optional): Performs an integrity check by default if the dataset has 20 or fewer tensors. Defaults to `True`.
- **lock_enabled** (`bool`, optional): If `True`, the dataset manages a write lock. Defaults to `True`.
- **lock_timeout** (`int`, optional): Number of seconds to wait before throwing a LockException. Defaults to `0`.
- **split_tensor_meta** (`bool`, optional): Each tensor has a tensor_meta.json if `True`. Defaults to `True`.

### Returns

- **Dataset**: The loaded dataset.

### Examples

```python
import muller

# Load an existing dataset
ds = muller.load("./datasets/my_dataset")

# Load in read-only mode
ds = muller.load("./datasets/my_dataset", read_only=True)

# Load from remote storage
ds = muller.load("s3://mybucket/my_dataset", creds={"aws_access_key_id": "...", "aws_secret_access_key": "..."})

# Load without integrity check for faster loading
ds = muller.load("./datasets/large_dataset", check_integrity=False)
```

---

## empty()

### Overview

Creates an empty dataset at the specified path. This is useful when you want to create a dataset structure before adding any data.

### Parameters

- **path** (`str` or `pathlib.Path`): The full path to the dataset.
- **overwrite** (`bool`, optional): If `True`, overwrites the dataset if it already exists. Defaults to `False`.
- **memory_cache_size** (`int`, optional): The size of the memory cache to be used in MB. Defaults to `DEFAULT_MEMORY_CACHE_SIZE`.
- **local_cache_size** (`int`, optional): The size of the local filesystem cache to be used in MB. Defaults to `DEFAULT_LOCAL_CACHE_SIZE`.
- **creds** (`dict` or `str`, optional): Credentials used to access the dataset. Defaults to `None`.
- **verbose** (`bool`, optional): If `True`, logs will be printed. Defaults to `True`.
- **lock_timeout** (`int`, optional): Number of seconds to wait before throwing a LockException. Defaults to `0`.
- **lock_enabled** (`bool`, optional): If `True`, the dataset manages a write lock. Defaults to `True`.
- **split_tensor_meta** (`bool`, optional): Each tensor has a tensor_meta.json if `True`. Defaults to `True`.

### Returns

- **Dataset**: The created empty dataset.

### Examples

```python
import muller

# Create an empty dataset
ds = muller.empty("./datasets/new_dataset")

# Create and overwrite if exists
ds = muller.empty("./datasets/new_dataset", overwrite=True)

# Create empty dataset on remote storage
ds = muller.empty("s3://mybucket/new_dataset", creds={"aws_access_key_id": "...", "aws_secret_access_key": "..."})

# Add tensors to the empty dataset
with ds:
    ds.create_tensor("images")
    ds.create_tensor("labels")
```

### Warning

Setting `overwrite=True` will delete all of your data if it exists!

---

## like()

### Overview

Copies the source dataset's structure to a new location. No samples are copied, only the meta/info for the dataset and its tensors. This is useful for creating a new dataset with the same schema as an existing one.

### Parameters

- **dest** (`str` or `Dataset`): Empty Dataset or Path where the new dataset will be created.
- **src** (`str` or `Dataset`): Path or dataset object that will be used as the template for the new dataset.
- **tensors** (`List[str]`, optional): Names of tensors (and groups) to be replicated. If not specified, all tensors in source dataset are considered. Defaults to `None`.
- **overwrite** (`bool`, optional): If `True` and a dataset exists at `dest`, it will be overwritten. Defaults to `False`.
- **verbose** (`bool`, optional): If `True`, logs will be printed. Defaults to `True`.

### Returns

- **Dataset**: New dataset object with the same structure as the source.

### Examples

```python
import muller

# Create a new dataset with the same structure as an existing one
source_ds = muller.load("./datasets/source_dataset")
new_ds = muller.like(dest="./datasets/new_dataset", src=source_ds)

# Copy structure from path
new_ds = muller.like(dest="./datasets/new_dataset", src="./datasets/source_dataset")

# Copy only specific tensors
new_ds = muller.like(
    dest="./datasets/new_dataset",
    src="./datasets/source_dataset",
    tensors=["images", "labels"]
)

# Overwrite if destination exists
new_ds = muller.like(
    dest="./datasets/new_dataset",
    src="./datasets/source_dataset",
    overwrite=True
)
```

---

## delete()

### Overview

Delete a dataset at the given path. This permanently removes all data and metadata associated with the dataset.

### Parameters

- **path** (`str` or `pathlib.Path`): The full path to the dataset to delete.
- **large_ok** (`bool`, optional): If `True`, allows deletion of large datasets. Defaults to `False`.
- **creds** (`dict` or `str`, optional): Credentials for OBS service. Defaults to `None`.

### Returns

- **None**

### Examples

```python
import muller

# Delete a dataset
muller.delete("./datasets/old_dataset")

# Delete a large dataset
muller.delete("./datasets/large_dataset", large_ok=True)

# Delete a remote dataset
muller.delete("s3://mybucket/old_dataset", creds={"aws_access_key_id": "...", "aws_secret_access_key": "..."})
```

### Warning

This operation is irreversible. All data will be permanently deleted.

---

## get_col_info()

### Overview

Get column (tensor) information from a dataset without loading the entire dataset. This is useful for quickly inspecting dataset metadata.

### Parameters

- **path** (`str` or `pathlib.Path`): The full path to the dataset.
- **col_name** (`str`, optional): Name of the column (tensor) to get info for. If `None` or empty string, returns dataset-level info. Defaults to `None`.

### Returns

- **bytes**: The raw metadata content in bytes.

### Examples

```python
import muller

# Get dataset-level info
info = muller.get_col_info("./datasets/my_dataset")

# Get specific tensor info
tensor_info = muller.get_col_info("./datasets/my_dataset", col_name="images")

# Parse the info (it's in JSON format)
import json
parsed_info = json.loads(info)
print(parsed_info)
```

---

## from_file()

### Overview

Create a dataset from a file (JSON lines format). This function reads data from a file and creates a MULLER dataset with the appropriate schema.

### Parameters

- **ori_path** (`str`): Path to the source file (JSON lines format).
- **muller_path** (`str`): Path where the muller dataset will be created.
- **schema** (`dict`, optional): Schema definition for the dataset. If not provided, schema will be inferred from the first record. Defaults to `None`.
- **workers** (`int`, optional): Number of workers for parallel processing. Defaults to `0`.
- **scheduler** (`str`, optional): Scheduler type for compute operations. Defaults to `"processed"`.
- **disable_rechunk** (`bool`, optional): Whether to disable rechunking. Defaults to `True`.
- **progressbar** (`bool`, optional): Whether to show progress bar. Defaults to `True`.
- **ignore_errors** (`bool`, optional): Whether to ignore errors during processing. Defaults to `True`.
- **split_tensor_meta** (`bool`, optional): Each tensor has a tensor_meta.json if `True`. Defaults to `True`.

### Returns

- **Dataset**: The created dataset.

### Examples

```python
import muller

# Create dataset from JSON lines file with inferred schema
ds = muller.from_file(
    ori_path="./data/records.jsonl",
    muller_path="./datasets/my_dataset"
)

# Create with explicit schema
schema = {
    "id": ("text", "str", None),
    "image": ("image", "uint8", "jpeg"),
    "label": ("class_label", "int32", None)
}
ds = muller.from_file(
    ori_path="./data/records.jsonl",
    muller_path="./datasets/my_dataset",
    schema=schema
)

# Use multiple workers for faster processing
ds = muller.from_file(
    ori_path="./data/large_records.jsonl",
    muller_path="./datasets/large_dataset",
    workers=4,
    progressbar=True
)

# Nested schema example
nested_schema = {
    "metadata": {
        "author": ("text", "str", None),
        "date": ("text", "str", None)
    },
    "content": ("text", "str", None)
}
ds = muller.from_file(
    ori_path="./data/nested_records.jsonl",
    muller_path="./datasets/nested_dataset",
    schema=nested_schema
)
```

### Notes

- The input file should be in JSON lines format (one JSON object per line).
- Schema format: `{column_name: (htype, dtype, sample_compression)}` or nested dictionaries.
- If schema is not provided, it will be inferred from the first record in the file.

---

## from_dataframes()

### Overview

Create a dataset from a list of dataframes (dictionaries). This is useful for creating datasets from in-memory data structures.

### Parameters

- **dataframes** (`list`): List of dataframes (dicts) to import.
- **muller_path** (`str`): Path where the muller dataset will be created.
- **schema** (`dict`, optional): Schema definition for the dataset. If not provided, schema will be inferred from the first record. Defaults to `None`.
- **workers** (`int`, optional): Number of workers for parallel processing. Defaults to `0`.
- **scheduler** (`str`, optional): Scheduler type for compute operations. Defaults to `"processed"`.
- **disable_rechunk** (`bool`, optional): Whether to disable rechunking. Defaults to `True`.
- **progressbar** (`bool`, optional): Whether to show progress bar. Defaults to `True`.
- **ignore_errors** (`bool`, optional): Whether to ignore errors during processing. Defaults to `True`.
- **split_tensor_meta** (`bool`, optional): Each tensor has a tensor_meta.json if `True`. Defaults to `True`.

### Returns

- **Dataset**: The created dataset.

### Examples

```python
import muller

# Create dataset from list of dictionaries
data = [
    {"id": "001", "value": 10, "label": "A"},
    {"id": "002", "value": 20, "label": "B"},
    {"id": "003", "value": 30, "label": "C"}
]
ds = muller.from_dataframes(
    dataframes=data,
    muller_path="./datasets/my_dataset"
)

# Create with explicit schema
schema = {
    "id": ("text", "str", None),
    "value": ("generic", "int32", None),
    "label": ("class_label", "str", None)
}
ds = muller.from_dataframes(
    dataframes=data,
    muller_path="./datasets/my_dataset",
    schema=schema
)

# Use multiple workers for large datasets
large_data = [{"col1": i, "col2": i*2} for i in range(10000)]
ds = muller.from_dataframes(
    dataframes=large_data,
    muller_path="./datasets/large_dataset",
    workers=4,
    progressbar=True
)

# Nested data example
nested_data = [
    {
        "user": {"name": "Alice", "age": 30},
        "score": 95
    },
    {
        "user": {"name": "Bob", "age": 25},
        "score": 87
    }
]
nested_schema = {
    "user": {
        "name": ("text", "str", None),
        "age": ("generic", "int32", None)
    },
    "score": ("generic", "int32", None)
}
ds = muller.from_dataframes(
    dataframes=nested_data,
    muller_path="./datasets/nested_dataset",
    schema=nested_schema
)
```

### Notes

- Each item in the `dataframes` list should be a dictionary representing one record.
- Schema format: `{column_name: (htype, dtype, sample_compression)}` or nested dictionaries.
- If schema is not provided, it will be inferred from the first record.
