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

Export the dataset to JSON format. This creates a JSON file or returns JSON data for the dataset.

#### Parameters

- **path** (`str`, optional): Path where the JSON file will be saved. If not provided, returns JSON string. Defaults to `None`.
- **tensor_list** (`List[str]`, optional): The tensor columns to export. If not provided, all tensors will be exported. Defaults to `None`.
- **index_list** (`List[int]`, optional): The indices of rows to export. If not provided, all rows will be exported. Defaults to `None`.
- **indent** (`int`, optional): Number of spaces for JSON indentation. Defaults to `2`.

#### Returns

- **str** or **None**: JSON string if `path` is not provided, otherwise `None` (writes to file).

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Export to JSON file
ds.to_json("./output/dataset.json")

# Export specific tensors
ds.to_json("./output/labels_only.json", tensor_list=["labels"])

# Export specific samples
ds.to_json("./output/sample_subset.json", index_list=[0, 1, 2, 3, 4])

# Get JSON string without saving
json_str = ds.to_json()
print(json_str)

# Export with custom indentation
ds.to_json("./output/dataset.json", indent=4)

# Export filtered view
filtered = ds.filter("labels == 5")
filtered.to_json("./output/label_5_samples.json")
```

---

### ds.to_arrow()

#### Overview

Convert the dataset to Apache Arrow format. This is useful for interoperability with Arrow-based tools and efficient data transfer.

#### Parameters

- **tensor_list** (`List[str]`, optional): The tensor columns to export. If not provided, all tensors will be exported. Defaults to `None`.
- **index_list** (`List[int]`, optional): The indices of rows to export. If not provided, all rows will be exported. Defaults to `None`.

#### Returns

- **pyarrow.Table**: The dataset as an Arrow Table.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Convert to Arrow Table
arrow_table = ds.to_arrow()
print(arrow_table.schema)

# Export specific tensors
arrow_table = ds.to_arrow(tensor_list=["labels", "features"])

# Export specific samples
arrow_table = ds.to_arrow(index_list=range(100))

# Write to Parquet using Arrow
arrow_table = ds.to_arrow()
import pyarrow.parquet as pq
pq.write_table(arrow_table, "./output/dataset.parquet")

# Convert to pandas via Arrow
arrow_table = ds.to_arrow()
df = arrow_table.to_pandas()

# Use with Arrow datasets
arrow_table = ds.to_arrow()
import pyarrow.dataset as ds_arrow
ds_arrow.write_dataset(arrow_table, "./output/arrow_dataset", format="parquet")
```

---

### ds.to_mindrecord()

#### Overview

Export the dataset to MindRecord format, which is used by MindSpore framework. This is useful for training models with MindSpore.

#### Parameters

- **path** (`str`): Path where the MindRecord files will be saved.
- **tensor_list** (`List[str]`, optional): The tensor columns to export. If not provided, all tensors will be exported. Defaults to `None`.
- **index_list** (`List[int]`, optional): The indices of rows to export. If not provided, all rows will be exported. Defaults to `None`.
- **num_shards** (`int`, optional): Number of MindRecord shards to create. Defaults to `1`.
- **overwrite** (`bool`, optional): If `True`, overwrites existing files. Defaults to `False`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Export to MindRecord
ds.to_mindrecord("./output/dataset.mindrecord")

# Export with multiple shards
ds.to_mindrecord("./output/dataset.mindrecord", num_shards=8)

# Export specific tensors
ds.to_mindrecord(
    "./output/images_labels.mindrecord",
    tensor_list=["images", "labels"]
)

# Export subset of data
ds.to_mindrecord(
    "./output/train_subset.mindrecord",
    index_list=range(1000)
)

# Overwrite existing files
ds.to_mindrecord(
    "./output/dataset.mindrecord",
    overwrite=True
)

# Export filtered view
train_ds = ds.filter("split == 'train'")
train_ds.to_mindrecord("./output/train.mindrecord", num_shards=4)
```

#### Notes

- MindRecord format is optimized for MindSpore training workflows.
- Multiple shards can improve parallel data loading performance.
- Requires MindSpore to be installed.

---

### ds.write_to_parquet()

#### Overview

Write the dataset to Parquet format. Parquet is a columnar storage format that is efficient for analytics and widely supported.

#### Parameters

- **path** (`str`): Path where the Parquet file(s) will be saved.
- **tensor_list** (`List[str]`, optional): The tensor columns to export. If not provided, all tensors will be exported. Defaults to `None`.
- **index_list** (`List[int]`, optional): The indices of rows to export. If not provided, all rows will be exported. Defaults to `None`.
- **compression** (`str`, optional): Compression codec to use (e.g., "snappy", "gzip", "brotli"). Defaults to `"snappy"`.
- **row_group_size** (`int`, optional): Number of rows per row group. Defaults to `None` (automatic).

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Write to Parquet
ds.write_to_parquet("./output/dataset.parquet")

# Write with specific compression
ds.write_to_parquet("./output/dataset.parquet", compression="gzip")

# Write specific tensors
ds.write_to_parquet(
    "./output/labels_only.parquet",
    tensor_list=["labels", "categories"]
)

# Write subset of data
ds.write_to_parquet(
    "./output/sample_subset.parquet",
    index_list=range(1000)
)

# Write with custom row group size
ds.write_to_parquet(
    "./output/dataset.parquet",
    row_group_size=10000
)

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
- Compression reduces file size but may increase read/write time.
- Parquet files can be read by many tools including pandas, Spark, and DuckDB.

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
