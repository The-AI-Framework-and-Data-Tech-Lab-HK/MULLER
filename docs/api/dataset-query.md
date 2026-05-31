# Dataset Query, Filter, and Indexing

This page documents methods for querying, filtering, searching, and managing indexes in datasets.

## Table of Contents

### Query and Filter
- [ds.filter()](#dsfilter)
- [ds.filter_vectorized()](#dsfilter_vectorized)
- [ds.query()](#dsquery)
- [ds.vector_search()](#dsvector_search)

### Indexing
- [ds.create_index()](#dscreate_index)
- [ds.create_index_vectorized()](#dscreate_index_vectorized)
- [ds.optimize_index()](#dsoptimize_index)
- [ds.create_vector_index()](#dscreate_vector_index)
- [ds.drop_vector_index()](#dsdrop_vector_index)
- [ds.update_vector_index()](#dsupdate_vector_index)
- [ds.load_vector_index()](#dsload_vector_index)
- [ds.unload_vector_index()](#dsunload_vector_index)
- [ds.create_hot_shard_index()](#dscreate_hot_shard_index)
- [ds.reshard_index()](#dsreshard_index)

### Views
- [ds.load_view()](#dsload_view)
- [ds.save_view()](#dssave_view)
- [ds.delete_view()](#dsdelete_view)
- [ds.get_views()](#dsget_views)

---

## Query and Filter

### ds.filter()

#### Overview

Filter the dataset based on a query expression. Returns a filtered view of the dataset without modifying the original data.

#### Parameters

- **expression** (`str` or `callable`): Filter expression or function. Can be:
  - A string expression (e.g., `"labels == 5"`)
  - A callable function that takes a sample and returns a boolean

#### Returns

- **Dataset**: A filtered view of the dataset.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Filter with string expression
filtered_ds = ds.filter("labels == 5")

# Filter with comparison operators
filtered_ds = ds.filter("age > 18")
filtered_ds = ds.filter("score >= 90")

# Filter with logical operators
filtered_ds = ds.filter("(labels == 1) | (labels == 2)")
filtered_ds = ds.filter("(age > 18) & (score > 80)")

# Filter with callable function
def custom_filter(sample):
    return sample["labels"].numpy() > 5

filtered_ds = ds.filter(custom_filter)

# Chain multiple filters
filtered_ds = ds.filter("labels > 0").filter("score < 100")

# Access filtered data
print(f"Original: {ds.num_samples} samples")
print(f"Filtered: {filtered_ds.num_samples} samples")

for sample in filtered_ds:
    print(sample)
```

---

### ds.filter_vectorized()

#### Overview

Filter the dataset using vectorized operations for better performance. This method is optimized for large datasets and uses indexed tensors.

#### Parameters

- **expression** (`str`): Filter expression using tensor names.
- **scheduler** (`str`, optional): Scheduler type for parallel processing. Defaults to `"threaded"`.
- **num_workers** (`int`, optional): Number of workers for parallel processing. Defaults to `0`.

#### Returns

- **Dataset**: A filtered view of the dataset.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Vectorized filter (faster for large datasets)
filtered_ds = ds.filter_vectorized("labels == 5")

# Use with multiple workers
filtered_ds = ds.filter_vectorized("score > 80", num_workers=4)

# Complex expressions
filtered_ds = ds.filter_vectorized("(age >= 18) & (age <= 65)")

# Filter on indexed tensors for best performance
ds.create_index("labels")
filtered_ds = ds.filter_vectorized("labels == 3")
```

---

### ds.query()

#### Overview

Query a specific tensor using a query expression. This is useful for searching within a single tensor.

#### Parameters

- **tensor_name** (`str`): Name of the tensor to query.
- **query** (`str`): Query expression.

#### Returns

- **Dataset**: A filtered view containing matching samples.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Query a specific tensor
result = ds.query("labels", "value == 5")

# Query with range
result = ds.query("age", "value >= 18 and value <= 65")

# Query text tensor
result = ds.query("description", "value.contains('important')")

# Access query results
for sample in result:
    print(sample["labels"], sample["description"])
```

---

### ds.vector_search()

#### Overview

Perform vector similarity search on a tensor with a vector index. This is useful for finding similar embeddings or features.

#### Parameters

- **query_vector** (`np.ndarray` or `Tensor`): The query vector to search for.
- **tensor_name** (`str`): Name of the tensor to search in.
- **index_name** (`str`): Name of the vector index to use.
- **k** (`int`, optional): Number of nearest neighbors to return. Defaults to `10`.
- **kwargs**: Additional arguments passed to the vector search backend.

#### Returns

- **Dataset**: A view containing the k nearest neighbors.

#### Examples

```python
import muller
import numpy as np

ds = muller.load("./my_dataset")

# Create vector index first
ds.create_vector_index("embeddings", index_name="emb_idx")

# Perform vector search
query_vec = np.random.rand(512)
results = ds.vector_search(
    query_vector=query_vec,
    tensor_name="embeddings",
    index_name="emb_idx",
    k=10
)

# Access search results
print(f"Found {results.num_samples} similar samples")
for i, sample in enumerate(results):
    print(f"Rank {i+1}: {sample['id']}")

# Search with additional parameters
results = ds.vector_search(
    query_vector=query_vec,
    tensor_name="embeddings",
    index_name="emb_idx",
    k=20,
    metric="cosine"
)
```

---

## Indexing

### ds.create_index()

#### Overview

Create an index on a tensor to speed up filtering and querying operations.

#### Parameters

- **tensor_name** (`str`): Name of the tensor to index.
- **index_type** (`str`, optional): Type of index to create. Defaults to `"hash"`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Create index on labels tensor
ds.create_index("labels")

# Create index with specific type
ds.create_index("categories", index_type="hash")

# Create indexes on multiple tensors
for tensor_name in ["labels", "categories", "user_id"]:
    ds.create_index(tensor_name)

# Use indexed tensor for faster filtering
ds.create_index("labels")
filtered = ds.filter_vectorized("labels == 5")  # Much faster with index
```

---

### ds.create_index_vectorized()

#### Overview

Create an index optimized for vectorized operations. This is useful for large-scale filtering.

#### Parameters

- **tensor_name** (`str`): Name of the tensor to index.
- **num_workers** (`int`, optional): Number of workers for parallel index creation. Defaults to `0`.
- **scheduler** (`str`, optional): Scheduler type. Defaults to `"threaded"`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Create vectorized index
ds.create_index_vectorized("labels")

# Create with multiple workers for faster indexing
ds.create_index_vectorized("categories", num_workers=4)

# Create vectorized indexes on multiple tensors
for tensor_name in ["labels", "user_id", "timestamp"]:
    ds.create_index_vectorized(tensor_name, num_workers=4)
```

---

### ds.optimize_index()

#### Overview

Optimize an existing index to improve query performance. This reorganizes the index structure for better efficiency.

#### Parameters

- **tensor_name** (`str`): Name of the tensor whose index to optimize.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Optimize index after many updates
ds.optimize_index("labels")

# Optimize all indexes
for tensor_name in ds.indexed_tensors:
    ds.optimize_index(tensor_name)
```

---

### ds.create_vector_index()

#### Overview

Create a vector index for similarity search on embedding tensors. This enables fast nearest neighbor search.

#### Parameters

- **tensor_name** (`str`): Name of the tensor containing vectors/embeddings.
- **index_name** (`str`): Name for the vector index.
- **index_type** (`str`, optional): Type of vector index (e.g., "HNSW", "IVF"). Defaults to `"HNSW"`.
- **metric** (`str`, optional): Distance metric (e.g., "cosine", "euclidean"). Defaults to `"cosine"`.
- **kwargs**: Additional parameters for the vector index.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Create vector index with default settings
ds.create_vector_index("embeddings", index_name="emb_idx")

# Create with specific metric
ds.create_vector_index(
    "embeddings",
    index_name="emb_idx",
    metric="euclidean"
)

# Create with custom parameters
ds.create_vector_index(
    "embeddings",
    index_name="emb_idx",
    index_type="HNSW",
    metric="cosine",
    M=16,
    ef_construction=200
)
```

---

### ds.drop_vector_index()

#### Overview

Delete a vector index from a tensor.

#### Parameters

- **tensor_name** (`str`): Name of the tensor.
- **index_name** (`str`): Name of the vector index to drop.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Drop a vector index
ds.drop_vector_index("embeddings", index_name="emb_idx")

# Drop and recreate with different parameters
ds.drop_vector_index("embeddings", index_name="old_idx")
ds.create_vector_index("embeddings", index_name="new_idx", metric="euclidean")
```

---

### ds.update_vector_index()

#### Overview

Update a vector index after adding new samples to the dataset.

#### Parameters

- **tensor_name** (`str`): Name of the tensor.
- **index_name** (`str`): Name of the vector index to update.

#### Returns

- **None**

#### Examples

```python
import muller
import numpy as np

ds = muller.load("./my_dataset")

# Add new samples
with ds:
    for i in range(100):
        ds.append({
            "embeddings": np.random.rand(512),
            "labels": i
        })

# Update the vector index to include new samples
ds.update_vector_index("embeddings", index_name="emb_idx")
```

---

### ds.load_vector_index()

#### Overview

Load a vector index into memory for faster search operations.

#### Parameters

- **tensor_name** (`str`): Name of the tensor.
- **index_name** (`str`): Name of the vector index to load.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Load vector index into memory
ds.load_vector_index("embeddings", index_name="emb_idx")

# Now searches will be faster
query_vec = np.random.rand(512)
results = ds.vector_search(query_vec, "embeddings", "emb_idx", k=10)
```

---

### ds.unload_vector_index()

#### Overview

Unload a vector index from memory to free up resources.

#### Parameters

- **tensor_name** (`str`): Name of the tensor.
- **index_name** (`str`): Name of the vector index to unload.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Unload vector index to free memory
ds.unload_vector_index("embeddings", index_name="emb_idx")

# Load again when needed
ds.load_vector_index("embeddings", index_name="emb_idx")
```

---

### ds.create_hot_shard_index()

#### Overview

Create a hot shard index for frequently accessed data. This optimizes access patterns for hot data.

#### Parameters

- **tensor_name** (`str`): Name of the tensor to create hot shard index for.
- **shard_size** (`int`, optional): Size of each shard. Defaults to automatic calculation.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Create hot shard index
ds.create_hot_shard_index("labels")

# Create with custom shard size
ds.create_hot_shard_index("embeddings", shard_size=1000)
```

---

### ds.reshard_index()

#### Overview

Reorganize index shards for better performance. This is useful after significant data changes.

#### Parameters

- **tensor_name** (`str`): Name of the tensor whose index to reshard.
- **num_shards** (`int`, optional): Target number of shards. Defaults to automatic calculation.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Reshard index
ds.reshard_index("labels")

# Reshard with specific number of shards
ds.reshard_index("categories", num_shards=10)
```

---

## Views

### ds.load_view()

#### Overview

Load a saved view of the dataset. Views are filtered or transformed versions of the dataset that have been saved for reuse.

#### Parameters

- **view_name** (`str`): Name of the view to load.

#### Returns

- **Dataset**: The loaded view.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Load a saved view
view = ds.load_view("high_quality_samples")

# Access view data
print(f"View has {view.num_samples} samples")
for sample in view:
    print(sample)

# Load and further filter
view = ds.load_view("category_a")
filtered_view = view.filter("score > 80")
```

---

### ds.save_view()

#### Overview

Save the current dataset view for later reuse. This is useful for saving filtered or transformed datasets.

#### Parameters

- **view_name** (`str`): Name to save the view as.
- **overwrite** (`bool`, optional): If `True`, overwrites existing view with same name. Defaults to `False`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Create and save a filtered view
filtered = ds.filter("labels == 5")
filtered.save_view("label_5_samples")

# Save a complex view
view = ds.filter("age >= 18").filter("score > 80")
view.save_view("qualified_adults")

# Overwrite existing view
new_view = ds.filter("labels == 3")
new_view.save_view("label_3_samples", overwrite=True)
```

---

### ds.delete_view()

#### Overview

Delete a saved view from the dataset.

#### Parameters

- **view_name** (`str`): Name of the view to delete.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Delete a view
ds.delete_view("old_view")

# Delete multiple views
for view_name in ["temp_view1", "temp_view2", "test_view"]:
    ds.delete_view(view_name)
```

---

### ds.get_views()

#### Overview

Get a list of all saved views in the dataset.

#### Parameters

None

#### Returns

- **List[str]**: List of view names.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Get all views
views = ds.get_views()
print(f"Available views: {views}")

# Load each view
for view_name in views:
    view = ds.load_view(view_name)
    print(f"{view_name}: {view.num_samples} samples")

# Check if view exists
if "my_view" in ds.get_views():
    view = ds.load_view("my_view")
```
