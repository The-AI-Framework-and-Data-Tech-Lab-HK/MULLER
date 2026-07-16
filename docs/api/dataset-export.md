# Dataset Export Methods

This page documents methods for exporting dataset data to various formats.

## Table of Contents

- [ds.to_dataframe()](#dsto_dataframe)
- [ds.to_json()](#dsto_json)
- [ds.to_arrow()](#dsto_arrow)
- [ds.to_mindrecord()](#dsto_mindrecord)
- [ds.write_to_parquet()](#dswrite_to_parquet)

---

### ds.to_dataframe()

#### Overview

Convert the dataset to a pandas DataFrame. This is useful for data analysis and integration with pandas-based workflows.

#### Signature

```python
ds.to_dataframe(
    tensor_list=None,
    index_list=None,
    force=False,
)
```

#### Parameters

- **tensor_list** (`List[str]`, optional): The tensor columns to export. If not provided, all tensors will be exported. Defaults to `None`.
- **index_list** (`List[int]`, optional): The indices of rows to export. If not provided, all rows will be exported. Defaults to `None`.
- **force** (`bool`, optional): If `True`, exports the dataset regardless of size. Datasets with more than `TO_DATAFRAME_SAFE_LIMIT` samples might take a long time to export. Defaults to `False`.

#### Returns

- **pandas.DataFrame**: The dataset as a pandas DataFrame.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Export entire dataset to DataFrame
df = ds.to_dataframe()
print(df.head())

# Export specific tensors
df = ds.to_dataframe(tensor_list=["images", "labels"])

# Export specific rows
df = ds.to_dataframe(index_list=[0, 1, 2, 10, 20])

# Export specific tensors and rows
df = ds.to_dataframe(
    tensor_list=["labels", "categories"],
    index_list=[1, 2, 4, 8, 16]
)

# Export last few samples
df = ds.to_dataframe(index_list=[-1, -2, -3])

# Force export of large dataset
df = ds.to_dataframe(force=True)

# Use DataFrame for analysis
df = ds.to_dataframe()
print(df.describe())
print(df["labels"].value_counts())
```

#### Notes

- For large datasets, consider using `index_list` to export in batches.
- Image and large binary data will be represented as arrays in the DataFrame.
- Use `force=True` carefully with large datasets as it may consume significant memory.

---

### ds.to_json()

#### Overview

Export the dataset to a JSON or JSONL file, row by row.

#### Signature

```python
ds.to_json(
    path,
    tensors=None,
    num_workers=1,
)
```

#### Parameters

- **path** (`str`): Output file path. The filename must end with `.json` or `.jsonl`.
- **tensors** (`List[str]`, optional): Tensor columns to export. If not provided, all tensors are exported. Defaults to `None`.
- **num_workers** (`int`, optional): Number of worker processes used to convert dataset slices before writing. Must be greater than `0`. Defaults to `1`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Export to JSON file
ds.to_json("./output/dataset.json")

# Export to JSONL file
ds.to_json("./output/dataset.jsonl")

# Export specific tensors
ds.to_json("./output/labels_only.json", tensors=["labels"])

# Use multiple workers
ds.to_json("./output/dataset.jsonl", num_workers=4)

# Export filtered view
filtered = ds.filter("labels == 5")
filtered.to_json("./output/label_5_samples.json")
```

#### Notes

- `to_json()` writes to a file and does not return a JSON string.
- The current API does not accept `index_list` or `indent`. Slice/filter the dataset before calling `to_json()` if you need a subset of rows.

---

### ds.to_arrow()

#### Overview

Create a MULLER-backed Arrow Dataset object. Use its Arrow Dataset methods, such as `to_table()`, `scanner()`, `head()`, and `count_rows()`, to materialize or inspect data.

#### Signature

```python
ds.to_arrow()
```

#### Parameters

None

#### Returns

- **MULLERArrowDataset**: A `pyarrow.dataset.Dataset` subclass backed by the MULLER dataset.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Create an Arrow Dataset wrapper
arrow_ds = ds.to_arrow()
print(arrow_ds.schema)

# Convert to Arrow Table
arrow_table = arrow_ds.to_table()
print(arrow_table.schema)

# Export specific columns
arrow_table = arrow_ds.to_table(columns=["labels", "features"])

# Write to Parquet using Arrow
import pyarrow.parquet as pq
pq.write_table(arrow_table, "./output/dataset.parquet")

# Convert to pandas via Arrow
df = arrow_table.to_pandas()

# Inspect the first rows
preview = arrow_ds.head(10)
```

#### Notes

- Supported Arrow conversions are defined by tensor htype/dtype. Unsupported combinations raise `UnsupportedArrowConvertError`.
- Use Arrow Dataset methods to choose columns or materialize a table; `ds.to_arrow()` itself does not accept `tensor_list` or `index_list`.

---

### ds.to_mindrecord()

#### Overview

Export the dataset to MindRecord format, which is used by MindSpore framework. This is useful for training models with MindSpore.

#### Signature

```python
ds.to_mindrecord(
    file_name,
    shard_num=1,
    batch_size=100000,
    overwrite=False,
    scheduler="threaded",
)
```

#### Parameters

- **file_name** (`str`): Output MindRecord filename.
- **shard_num** (`int`, optional): Number of MindRecord files to generate. Defaults to `1`.
- **batch_size** (`int`, optional): Batch size used when reading NumPy data from MULLER. Defaults to `100000`.
- **overwrite** (`bool`, optional): If `True`, overwrite existing files with the same name. Defaults to `False`.
- **scheduler** (`str`, optional): Scheduler used while reading tensor batches. Supported values include `"serial"`, `"threaded"`, `"processed"`, and `"distributed"`. Defaults to `"threaded"`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Export to MindRecord
ds.to_mindrecord("./output/dataset.mindrecord")

# Export with multiple shards
ds.to_mindrecord("./output/dataset.mindrecord", shard_num=8)

# Export a subset by slicing first
train_subset = ds[:1000]
train_subset.to_mindrecord(
    "./output/train_subset.mindrecord",
    batch_size=1000,
)

# Overwrite existing files
ds.to_mindrecord(
    "./output/dataset.mindrecord",
    overwrite=True,
)

# Choose a scheduler
ds.to_mindrecord(
    "./output/dataset.mindrecord",
    scheduler="threaded",
)

# Export filtered view
train_ds = ds.filter("split == 'train'")
train_ds.to_mindrecord("./output/train.mindrecord", shard_num=4)
```

#### Notes

- MindRecord format is optimized for MindSpore training workflows.
- Multiple shards can improve parallel data loading performance.
- Requires MindSpore to be installed.
- The current API exports the dataset's tensors as a whole and does not accept `tensor_list` or `index_list`. Slice/filter the dataset before exporting a row subset.

---

### ds.write_to_parquet()

#### Overview

Write the dataset to Parquet format through the dataset's storage backend.

#### Signature

```python
ds.write_to_parquet(path, columns=None)
```

#### Parameters

- **path** (`str`): Storage key/path where the Parquet bytes will be written.
- **columns** (`List[str]`, optional): Columns to include in the Parquet output. If omitted, all columns are exported. Defaults to `None`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Write to Parquet
ds.write_to_parquet("./output/dataset.parquet")

# Write specific tensors
ds.write_to_parquet(
    "./output/labels_only.parquet",
    columns=["labels", "categories"]
)

# Write a subset by slicing first
ds[:1000].write_to_parquet("./output/sample_subset.parquet")

# Write filtered view
filtered = ds.filter("score > 80")
filtered.write_to_parquet("./output/high_scores.parquet")

# Write multiple partitions
train_ds = ds.filter("split == 'train'")
test_ds = ds.filter("split == 'test'")
train_ds.write_to_parquet("./output/train.parquet")
test_ds.write_to_parquet("./output/test.parquet")
```

#### Notes

- Parquet format is highly efficient for columnar data access.
- Parquet files can be read by many tools including pandas, Spark, and DuckDB.
- The current API does not expose compression, row group size, or row index parameters. Use `to_arrow().to_table()` with PyArrow directly if you need those writer options.

---

## Comparison of Export Formats

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| **DataFrame** | Data analysis, pandas workflows | Easy to use, familiar API | Memory intensive for large datasets |
| **JSON** | Data interchange, human-readable | Universal format, readable | Large file size, slower parsing |
| **Arrow** | Interoperability, efficient transfer | Fast, zero-copy, language-agnostic | Requires Arrow ecosystem |
| **MindRecord** | MindSpore training | Optimized for MindSpore | MindSpore-specific |
| **Parquet** | Analytics, data warehousing | Efficient, columnar, widely supported | Not human-readable |

### Choosing the Right Format

- Use **to_dataframe()** for quick analysis and pandas integration
- Use **to_json()** for data interchange and human readability
- Use **to_arrow()** for efficient data transfer and Arrow ecosystem integration
- Use **to_mindrecord()** for MindSpore model training
- Use **write_to_parquet()** for efficient storage and analytics workflows
