# Dataset Query, Filter, and Indexing

This page documents methods for querying, filtering, searching, and managing indexes in datasets.

## Table of Contents

### Query and Filter
- [ds.filter()](#dsfilter)
- [ds.filter_vectorized()](#dsfilter_vectorized)
- [ds.aggregate()](#dsaggregate)
- [ds.aggregate_vectorized()](#dsaggregate_vectorized)
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
- [ds.get_view()](#dsget_view)

---

## Query and Filter

### ds.filter()

#### Overview

Filter the dataset with a Python callable, a string query expression, an optional inverted-index query, or a combination of them. Returns a filtered view and does not modify the original dataset.

#### Signature

```python
ds.filter(
    function=None,
    index_query=None,
    connector="AND",
    offset=0,
    limit=None,
    **kwargs,
)
```

#### Parameters

- **function** (`callable` or `str`, optional): Filtering condition. A callable receives a sample and returns `True` for rows to keep; a string is evaluated as a dataset query expression such as `"labels == 1"`.
- **index_query** (`str`, optional): Query expression evaluated against the dataset's `query_string` for inverted-index results.
- **connector** (`str`, optional): How to combine `function` and `index_query`. Supported values are `"AND"` and `"OR"`. Defaults to `"AND"`.
- **offset** (`int`, optional): Start filtering from this dataset index. Defaults to `0`.
- **limit** (`int`, optional): Maximum number of matching rows to return. Defaults to `None`.
- **kwargs**: Additional execution options passed to the filtering backend, including `num_workers`, `scheduler`, `progressbar`, `save_result`, `result_path`, `result_ds_args`, and `compute_future`.

#### Returns

- **Dataset**: A filtered view of the dataset.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# String query expression
filtered_ds = ds.filter("labels == 1")

# Callable filter
filtered_ds = ds.filter(lambda sample: sample.labels.data()["value"] == 1)

# Pagination
page = ds.filter("labels >= 2", offset=10, limit=5)
```

---

### ds.filter_vectorized()

#### Overview

Filter the dataset with vectorized NumPy operations and, where requested, inverted indexes. Conditions are passed as tuples in `condition_list`; this method does not accept a single string expression.

#### Signature

```python
ds.filter_vectorized(
    condition_list,
    connector_list=None,
    offset=0,
    limit=None,
    compute_future=True,
    use_local_index=True,
    max_workers=16,
    show_progress=False,
)
```

#### Parameters

- **condition_list** (`list[tuple]`): Filtering conditions. Each condition is `(tensor, operator, value)`, `(tensor, operator, value, use_inverted_index)`, or `(tensor, operator, value, use_inverted_index, negation)`.
- **connector_list** (`list[str]`, optional): Connectors between conditions. Each value must be `"AND"` or `"OR"`, and the list length must be `len(condition_list) - 1`.
- **offset** (`int`, optional): Start filtering from this dataset index. Defaults to `0`.
- **limit** (`int`, optional): Maximum number of matching rows to return. Defaults to `None`.
- **compute_future** (`bool`, optional): If `True`, precomputes the next page of results for limited queries. Defaults to `True`.
- **use_local_index** (`bool`, optional): Use the vectorized local inverted index when a condition requests index lookup. Defaults to `True`.
- **max_workers** (`int`, optional): Maximum workers used by indexed search paths. Defaults to `16`.
- **show_progress** (`bool`, optional): Log progress while computing individual conditions. Defaults to `False`.

Supported operators are `>`, `<`, `>=`, `<=`, `==`, `!=`, `CONTAINS`, `BETWEEN`, and `LIKE`. `CONTAINS` and `BETWEEN` use an inverted index; `LIKE` performs regex matching on text tensors. The optional `negation` value is `"NOT"`.

#### Returns

- **Dataset**: A filtered view of the dataset with `filtered_index` set to the matching source indices.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Numeric comparison
filtered_ds = ds.filter_vectorized([("labels", ">", 1)])

# Combine conditions
filtered_ds = ds.filter_vectorized(
    [("labels", ">=", 1), ("labels", "<", 4)],
    ["AND"],
)

# Text search with an inverted index
ds.commit()
ds.create_index_vectorized("description")
text_matches = ds.filter_vectorized([("description", "CONTAINS", "cat")])
```

---

### ds.aggregate()

#### Overview

Aggregate rows by one or more tensors, optionally after applying a callable filter. This implementation can run in-process or with compute workers.

#### Signature

```python
ds.aggregate(
    group_by_tensors,
    selected_tensors,
    order_by_tensors=None,
    aggregate_tensors=None,
    function=None,
    order_direction="DESC",
    num_workers=0,
    scheduler="processed",
    progressbar=True,
    method="count",
)
```

#### Parameters

- **group_by_tensors** (`list[str]`): Tensor names used as the group-by keys.
- **selected_tensors** (`list[str]`): Group-by tensor names to include in the output. Each selected tensor must also appear in `group_by_tensors`.
- **order_by_tensors** (`list[str]`, optional): Tensor names used to sort the result. Values must be selected tensors or aggregate tensors.
- **aggregate_tensors** (`list[str]`, optional): Tensor names to aggregate. For `method="count"`, `["*"]` counts rows and omitted values default to `["*"]`.
- **function** (`callable`, optional): Row filter applied before aggregation.
- **order_direction** (`str`, optional): `"DESC"` or `"ASC"`. Defaults to `"DESC"`.
- **num_workers** (`int`, optional): Number of compute workers. `0` runs in-place. Defaults to `0`.
- **scheduler** (`str`, optional): Compute scheduler when `num_workers > 0`. Defaults to `"processed"`.
- **progressbar** (`bool`, optional): Show progress during aggregation. Defaults to `True`.
- **method** (`str`, optional): Aggregation method. The current non-vectorized implementation supports `"count"` and `"sum"`. Defaults to `"count"`.

#### Returns

- **numpy.ndarray**: Aggregated result. Columns are selected tensors followed by aggregate columns.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

counts = ds.aggregate(
    group_by_tensors=["categories"],
    selected_tensors=["categories"],
    aggregate_tensors=["*"],
)

high_score_counts = ds.aggregate(
    function=lambda sample: sample.score.data()["value"] >= 4,
    group_by_tensors=["categories"],
    selected_tensors=["categories"],
    aggregate_tensors=["*"],
)
```

---

### ds.aggregate_vectorized()

#### Overview

Aggregate rows using NumPy vectorized operations. This path supports more aggregate methods and does not accept a row-level filter function.

#### Signature

```python
ds.aggregate_vectorized(
    group_by_tensors,
    selected_tensors,
    order_by_tensors=None,
    aggregate_tensors=None,
    order_direction="DESC",
    method="count",
)
```

#### Parameters

- **group_by_tensors** (`list[str]`): Tensor names used as the group-by keys.
- **selected_tensors** (`list[str]`): Tensor names included in the output.
- **order_by_tensors** (`list[str]`, optional): Tensor names used to sort the result. Values can refer to selected tensors or aggregate tensors.
- **aggregate_tensors** (`list[str]`, optional): Tensor names to aggregate. For `method="count"`, use `["*"]` to append row counts.
- **order_direction** (`str`, optional): `"DESC"` or `"ASC"`. Defaults to `"DESC"`.
- **method** (`str`, optional): Aggregation method. Supported values are `"count"`, `"sum"`, `"avg"`, `"min"`, and `"max"`. Defaults to `"count"`.

#### Returns

- **numpy.ndarray**: Aggregated result. Columns are selected tensors followed by aggregate columns when `aggregate_tensors` is provided.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

counts = ds.aggregate_vectorized(
    group_by_tensors=["categories"],
    selected_tensors=["categories"],
    aggregate_tensors=["*"],
)

price_sum = ds.aggregate_vectorized(
    group_by_tensors=["categories"],
    selected_tensors=["categories"],
    aggregate_tensors=["price"],
    method="sum",
)
```

---

### ds.query()

#### Overview

Query a single tensor through its inverted index. Create the index before calling this method.

#### Parameters

- **tensor_name** (`str`): Name of the indexed tensor to query.
- **query**: Query value or query string passed to the tensor's inverted index search implementation.

#### Returns

- **set**: Source indices matching the indexed query. If the index stores UUIDs, they are mapped back to source indices.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")
ds.commit()
ds.create_index(["description"])

matching_indices = ds.query("description", "cat")
result = ds[list(matching_indices)]
```

---

### ds.vector_search()

#### Overview

Perform vector similarity search on a loaded vector index. This is useful for finding similar embeddings or features.

`vector_search()` returns the raw nearest-neighbor result arrays from the vector index. It does **not** return a Dataset view; use the returned sample ids/indices to slice the dataset when you need samples.

#### Signature

```python
ds.vector_search(
    query_vector,
    tensor_name,
    index_name,
    **kwargs,
)
```

#### Parameters

- **query_vector** (`np.ndarray` or `Tensor`): Query vector(s) to search for. FAISS-backed indexes expect a 2-D array shaped `(num_queries, dimension)`.
- **tensor_name** (`str`): Name of the tensor to search in.
- **index_name** (`str`): Name of the vector index to use.
- **topk** (`int`, optional): Number of nearest neighbors to return per query. Defaults to `1` in the backend. The implementation reads `topk`; `k` is not a documented parameter for this API.
- **refine_factor** (`float`, optional): For approximate indexes, search `topk * refine_factor` candidates and re-rank by exact distance when greater than `1`.
- **nprobe** (`int`, optional): `IVFPQ` search parameter. Defaults to `8`.
- **ef_search** (`int`, optional): `HNSWFLAT` search parameter. Defaults to `16`.
- **complexity** (`int`, optional): `DISKANN` search parameter. Defaults to `8`.
- **beam_width** (`int`, optional): `DISKANN` search parameter. Defaults to `1`.
- **num_threads** (`int`, optional): `DISKANN` batch-search parameter. Defaults to `0`.

#### Returns

- **tuple[np.ndarray, np.ndarray]**: `(dist_list, id_list)`. `dist_list` contains distances/scores for each query and `id_list` contains the matching sample ids/positional indices stored in the index. MULLER builds vector indexes with positional ids (`0..len(ds)-1`).

#### Examples

```python
import muller
import numpy as np

ds = muller.load("./my_dataset")

# Create vector index first
ds.create_vector_index("embeddings", index_name="emb_idx")

# Perform vector search
query_vec = np.random.rand(1, 512)
distances, indices = ds.vector_search(
    query_vector=query_vec,
    tensor_name="embeddings",
    index_name="emb_idx",
    topk=10,
)

# Access matching samples explicitly
matches = ds[indices[0].tolist()]
for rank, (distance, sample_idx) in enumerate(zip(distances[0], indices[0]), start=1):
    print(f"Rank {rank}: sample={sample_idx}, distance={distance}")

# Search with additional parameters
distances, indices = ds.vector_search(
    query_vector=query_vec,
    tensor_name="embeddings",
    index_name="emb_idx",
    topk=20,
    ef_search=64,
)
```

---

## Indexing

### ds.create_index()

#### Overview

Create the legacy inverted index for one or more tensor columns. The dataset must be committed first; if there are uncommitted changes, the method warns and does nothing.

This API indexes text-like, class-label, list, string, `int64`, and `float64` tensor data for `ds.query()` and indexed query paths. For the newer sharded/vectorized inverted index used by `filter_vectorized()`, prefer `ds.create_index_vectorized()`.

#### Signature

```python
ds.create_index(
    columns,
    use_uuid=False,
    batch_size=INVERTED_INDEX_BATCH_SIZE,
)
```

#### Parameters

- **columns** (`list[str]`): Tensor names to index. The implementation iterates over `columns`, so pass a list even for a single tensor.
- **use_uuid** (`bool`, optional): Store tensor UUIDs in the index instead of positional sample indices. Defaults to `False`.
- **batch_size** (`int`, optional): Batch size used to split legacy index files. Defaults to `INVERTED_INDEX_BATCH_SIZE`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")
ds.commit()

# Create a legacy index on one tensor
ds.create_index(["labels"])

# Create legacy indexes on multiple tensors
ds.create_index(["labels", "categories", "user_id"])
```

---

### ds.create_index_vectorized()

#### Overview

Create a **vectorized inverted index** on a tensor column to power fast full-text
(`CONTAINS` / `LIKE`) and exact-match queries through `ds.filter_vectorized()` and
`ds.query()`. The index is built in parallel and automatically optimized (shard files
are merged) at the end of creation, so you do **not** need to call `ds.optimize_index()`
separately after a successful build.

If an index already exists for the column, calling this method again updates it in an
**append-only** manner (only the newly added rows are indexed) unless you pass
`force_create=True` to rebuild from scratch.

> **Note:** The dataset must be committed (no uncommitted head changes) before building an
> index; otherwise the call warns and does nothing.

#### Signature

```python
ds.create_index_vectorized(
    tensor_column,
    index_type="fuzzy_match",
    use_uuid=False,
    force_create=False,
    delete_old_index=True,
    use_cpp=False,
    **kwargs,
)
```

#### Parameters

- **tensor_column** (`str`): Name of the tensor column to index.
- **index_type** (`str`, optional): `"fuzzy_match"` (tokenized full-text search, the
  default) or `"exact_match"` (whole-value match). `"exact_match"` is **not** supported
  together with `use_cpp=True`.
- **use_uuid** (`bool`, optional): Index by the row UUIDs instead of positional indices.
  Defaults to `False`.
- **force_create** (`bool`, optional): If `True`, delete any existing index and rebuild it
  from scratch instead of appending. Defaults to `False`.
- **delete_old_index** (`bool`, optional): Whether to delete the previous index folder once
  the new one is built and promoted. Defaults to `True`.
- **use_cpp** (`bool`, optional): Use the native C++ indexing engine instead of the pure
  Python implementation. Defaults to `False`. See
  [Using the C++ engine](#using-the-c-engine-use_cpptrue) below.

The following tuning options are accepted as keyword arguments (`**kwargs`):

- **num_of_shards** (`int`, optional): Number of hash-partitioned shards the postings are
  split across. Defaults to `1`. Controls **search-time** parallelism and storage layout,
  and is **fixed at creation time** (changing it later requires `ds.reshard_index()`).
- **num_of_batches** (`int`, optional): Number of independent batches the row range is split
  into during construction. Defaults to `1`. Controls **build-time** parallelism.
- **max_workers** (`int`, optional): Upper bound on the number of parallel
  processes/threads used during build, optimize, and search. Defaults to `16`.
- **tokenizer** (`str`, optional): Tokenizer for `fuzzy_match`. Defaults to `"jieba"`.
- **cut_all** (`bool`, optional): Use jieba's full-cut mode. Defaults to `False`.
- **stop_words_list** (`list[str]`, optional): Extra stop words to ignore during tokenization.
- **compulsory_words** (`str`, optional): Words that must be kept as a single token.
- **case_sensitive** (`bool`, optional): Whether matching is case sensitive. Defaults to `False`.

#### Returns

- **None**

#### Choosing `num_of_shards`, `num_of_batches`, and `max_workers`

These three options control different things and are commonly misunderstood. The key point:
**`max_workers` is only an upper bound** — the parallelism that actually happens is capped by
how many work units exist.

| Option | Governs | Effective parallelism | Default |
|---|---|---|---|
| `num_of_batches` | **Build** (index creation) | `min(num_of_batches, max_workers)` | `1` |
| `num_of_shards` | **Search & optimize** + storage layout | `min(num_of_shards, max_workers)` | `1` |
| `max_workers` | Cap on all of the above | — | `16` |

- **`num_of_batches`** splits the dataset row range into independent construction tasks.
  Because the default is `1`, **raising `max_workers` alone does not parallelize the build** —
  you must increase `num_of_batches` first. Larger values also give finer restart granularity:
  if a build is interrupted, only the unfinished batches need to be re-run by calling
  `create_index_vectorized()` again.
- **`num_of_shards`** hash-partitions the inverted index. It is persisted in the index
  metadata at creation time and determines how many shards search and optimization can run in
  parallel. More shards means more query-time parallelism and smaller per-shard files (good for
  large datasets / large vocabularies), but too many shards on a small dataset only adds file
  overhead. To change it after the fact, use `ds.reshard_index()`.
- **`max_workers`** simply caps parallelism everywhere; set it close to the machine's CPU core
  count.

**Rules of thumb:**

- **Small datasets:** keep the defaults (`num_of_shards=1`, `num_of_batches=1`, `max_workers=16`).
- **Large datasets, faster build:** set `num_of_batches` to roughly `max_workers` (or a small
  multiple) so every worker stays busy, and set `max_workers` near the CPU core count.
- **High query concurrency / large vocabulary:** increase `num_of_shards` (e.g. into the tens)
  to parallelize search and keep per-shard files small.

#### Using the C++ engine (`use_cpp=True`)

MULLER ships a native C++ inverted-index engine that is significantly faster than the pure
Python path for both construction and search. Enable it with `use_cpp=True`. Important
constraints:

- **Requires the compiled C++ extension.** It is built automatically during a standard
  `pip install .` (via `muller/util/sparsehash/build_proj.sh`). If MULLER was installed with
  `BUILD_CPP=false`, the C++ engine is unavailable.
- **Local datasets only.** The C++ engine reads and writes the local filesystem directly
  (bypassing MULLER's storage abstraction), so it raises `UnsupportedMethod` on remote-backed
  datasets (e.g. S3/object storage). Use `use_cpp=False` for those.
- **`fuzzy_match` only.** Combining `index_type="exact_match"` with `use_cpp=True` raises
  `UnsupportedMethod`; use the Python engine (`use_cpp=False`) for exact-match indexes.

```python
import muller

ds = muller.load("./my_dataset")
ds.commit()  # an index requires a clean (committed) head

# Build a C++ full-text index on a local dataset
ds.create_index_vectorized("description", use_cpp=True)

# Parallel build on a large local dataset
ds.create_index_vectorized(
    "description",
    use_cpp=True,
    num_of_batches=16,   # parallelize construction
    num_of_shards=16,    # parallelize search / shard the postings
    max_workers=16,      # cap parallelism near CPU core count
)
```

#### Examples

```python
import muller

ds = muller.load("./my_dataset")
ds.commit()

# Create a vectorized full-text index (Python engine, defaults)
ds.create_index_vectorized("description")

# Query through the index
res = ds.filter_vectorized([("description", "CONTAINS", "cat")])

# Parallelize the build on a larger dataset
ds.create_index_vectorized("description", num_of_batches=8, max_workers=8)

# Shard the index for query-time parallelism
ds.create_index_vectorized("description", num_of_shards=16, max_workers=16)

# Exact-match index (Python engine only)
ds.create_index_vectorized("categories", index_type="exact_match")

# Rebuild an existing index from scratch instead of appending
ds.create_index_vectorized("description", force_create=True)
```

---

### ds.optimize_index()

#### Overview

Optimize a vectorized inverted index by merging shard files. `create_index_vectorized()` already calls this after a successful create/update, so you usually only need this when recovering or manually finishing an interrupted vectorized index build.

#### Signature

```python
ds.optimize_index(
    tensor,
    use_uuid=None,
    optimize_mode="create",
    max_workers=16,
    delete_old_index=True,
    use_cpp=False,
)
```

#### Parameters

- **tensor** (`str`): Name of the tensor whose vectorized inverted index to optimize.
- **use_uuid** (`bool`, optional): Whether the index uses UUIDs. Defaults to the index loader default when `None`.
- **optimize_mode** (`str`, optional): `"create"` to merge only temporary build shards, or `"update"` to merge temporary update shards together with the existing index. Defaults to `"create"`.
- **max_workers** (`int`, optional): Maximum parallel workers. Defaults to `16`.
- **delete_old_index** (`bool`, optional): Delete the previous index folder/prefix after promotion. Defaults to `True` at the Dataset API layer.
- **use_cpp** (`bool`, optional): Use the native C++ index files. Defaults to `False` at the Dataset API layer.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Normally create_index_vectorized() optimizes automatically.
ds.create_index_vectorized("description", num_of_shards=8)

# Manual recovery/maintenance path
ds.optimize_index("description", optimize_mode="create", max_workers=8)
```

---

### ds.create_vector_index()

#### Overview

Create a vector index for similarity search on embedding tensors. The dataset must be committed first; if there are uncommitted changes, the method warns and does nothing.

When the index is created, it is also cached in memory for immediate search. If you load the dataset in a later process/session, call `ds.load_vector_index()` before `ds.vector_search()`.

#### Signature

```python
ds.create_vector_index(
    tensor_name,
    index_name,
    index_type="FLAT",
    metric="l2",
    **kwargs,
)
```

#### Parameters

- **tensor_name** (`str`): Name of the tensor containing vectors/embeddings.
- **index_name** (`str`): Name for the vector index.
- **index_type** (`str`, optional): Type of vector index. Supported values in code are `"FLAT"`, `"IVFPQ"`, `"HNSWFLAT"`, and `"DISKANN"`. Defaults to `"FLAT"`.
- **metric** (`str`, optional): Distance metric. FAISS-backed indexes support `"l2"`, `"cosine"`, and `"inner_product"`. Defaults to `"l2"`.
- **overwrite** (`bool`, optional): If `True`, replace an existing vector index with the same name. Defaults to `False`; otherwise an existing index raises `IndexExistsError`.
- **nlist** (`int`, optional): `IVFPQ` create parameter. Defaults to `128`.
- **m** (`int`, optional): `IVFPQ` product-quantizer segments, default `1`; `HNSWFLAT` graph degree parameter, default `32`.
- **ef_construction** (`int`, optional): `HNSWFLAT` build parameter. Defaults to `40`.
- **complexity** (`int`, optional): `DISKANN` build parameter. Defaults to `5`.
- **graph_degree** (`int`, optional): `DISKANN` build parameter. Defaults to `5`.
- **num_nodes_to_cache** (`int`, optional): `DISKANN` build parameter. Defaults to `1`.
- **search_memory_maximum** (`float`, optional): `DISKANN` build parameter in GB. Defaults to `0.01`.
- **build_memory_maximum** (`float`, optional): `DISKANN` build parameter in GB. Defaults to `0.01`.
- **num_threads** (`int`, optional): `DISKANN` build parameter. Defaults to `4`.
- **pq_disk_bytes** (`int`, optional): `DISKANN` build parameter. Defaults to `0`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")
ds.commit()

# Create vector index with default settings
ds.create_vector_index("embeddings", index_name="flat_idx")

# Create with specific metric
ds.create_vector_index(
    "embeddings",
    index_name="cosine_idx",
    metric="cosine",
)

# Create with custom parameters
ds.create_vector_index(
    "embeddings",
    index_name="hnsw_idx",
    index_type="HNSWFLAT",
    metric="l2",
    m=16,
    ef_construction=200,
)
```

---

### ds.drop_vector_index()

#### Overview

Delete a vector index from storage and remove it from the in-memory index map.

Call this when an index is no longer needed, or before recreating it with different parameters if you do not want to use `overwrite=True`.

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
ds.create_vector_index("embeddings", index_name="new_idx", metric="cosine")
```

---

### ds.update_vector_index()

#### Overview

Update a vector index after adding new samples and committing the dataset. The implementation compares the index commit id with the dataset commit id and only updates when they differ.

Call this after append-only changes. The current implementation is designed for added rows; updates/deletes are not a general incremental vector-index path.

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
ds.commit()
ds.update_vector_index("embeddings", index_name="emb_idx")
```

---

### ds.load_vector_index()

#### Overview

Load a vector index into memory. `ds.vector_search()` requires the target index to be loaded; newly created indexes are cached immediately, but indexes from a freshly loaded dataset/session need an explicit load.

#### Parameters

- **tensor_name** (`str`): Name of the tensor.
- **index_name** (`str`): Name of the vector index to load.
- **kwargs**: Backend load parameters. FAISS indexes accept `device` (`"cpu"` by default, `"gpu"` when supported). `DISKANN` accepts `num_threads` (default `16`) and `num_nodes_to_cache` (default `10`).

#### Returns

- **None**

#### Examples

```python
import muller
import numpy as np

ds = muller.load("./my_dataset")

# Load an existing vector index into memory after loading a dataset/session
ds.load_vector_index("embeddings", index_name="emb_idx")

# Now searches can run
query_vec = np.random.rand(1, 512)
distances, indices = ds.vector_search(query_vec, "embeddings", "emb_idx", topk=10)
```

---

### ds.unload_vector_index()

#### Overview

Unload a vector index from memory to free up resources while keeping the persisted index on disk/storage.

Call this when you are done with searches in a long-running process and want to release memory. Call `ds.load_vector_index()` again before the next `ds.vector_search()`.

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

Create a hot shard file for a vectorized inverted index by collecting the most frequent terms from existing shards.

Call this after creating a vectorized inverted index when repeated queries are dominated by common terms and you want a cached hot shard.

#### Signature

```python
ds.create_hot_shard_index(
    tensor,
    use_uuid=None,
    max_workers=16,
    n=100000,
)
```

#### Parameters

- **tensor** (`str`): Name of the tensor whose vectorized inverted index should receive a hot shard.
- **use_uuid** (`bool`, optional): Whether the index uses UUIDs. Defaults to the index loader default when `None`.
- **max_workers** (`int`, optional): Maximum parallel workers. Defaults to `16`.
- **n** (`int`, optional): Number of most frequent terms to include in the hot shard. Defaults to `100000`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Create hot shard index
ds.create_hot_shard_index("description")

# Create with a smaller hot-term set
ds.create_hot_shard_index("description", n=1000, max_workers=8)
```

---

### ds.reshard_index()

#### Overview

Re-shard an existing vectorized inverted index from one shard count to another. This is the maintenance path for changing `num_of_shards` after creation.

The method rewrites shard contents but does not update the public `create_index_vectorized()` call that originally chose `num_of_shards`; keep track of the old and new shard counts you use.

#### Signature

```python
ds.reshard_index(
    tensor,
    old_shard_num,
    new_shard_num,
    max_workers=16,
    use_uuid=None,
)
```

#### Parameters

- **tensor** (`str`): Name of the tensor whose vectorized inverted index should be re-sharded.
- **old_shard_num** (`int`): Current number of shards.
- **new_shard_num** (`int`): Target number of shards.
- **max_workers** (`int`, optional): Maximum parallel workers. Defaults to `16`.
- **use_uuid** (`bool`, optional): Whether the index uses UUIDs. Defaults to the index loader default when `None`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Reshard index
ds.reshard_index("description", old_shard_num=4, new_shard_num=16)

# Limit parallelism
ds.reshard_index("description", old_shard_num=4, new_shard_num=16, max_workers=8)
```

---

## Views

### ds.load_view()

#### Overview

Load a saved view by its view id. This is equivalent to `ds.get_view(view_id).load()`.

#### Signature

```python
ds.load_view(
    view_id,
    optimize=False,
    tensors=None,
    num_workers=0,
    scheduler="threaded",
    progressbar=True,
)
```

#### Parameters

- **view_id** (`str`): Id of the saved view to load.
- **optimize** (`bool`, optional): If `True`, optimize the view by copying and rechunking the required data before loading. Defaults to `False`.
- **tensors** (`list[str]`, optional): Tensor names to copy when `optimize=True`. If omitted, all tensors are copied.
- **num_workers** (`int`, optional): Number of workers used for optimization. Only applies when `optimize=True`. Defaults to `0`.
- **scheduler** (`str`, optional): Scheduler used for optimization. Supported values include `"serial"`, `"threaded"`, `"processed"`, and `"distributed"`. Defaults to `"threaded"`.
- **progressbar** (`bool`, optional): Whether to show progress during optimization. Only applies when `optimize=True`. Defaults to `True`.

#### Returns

- **Dataset**: The loaded view.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Load a saved view by id
view = ds.load_view("high_quality_samples")

# Access view data
print(f"View has {len(view)} samples")
for sample in view:
    print(sample)

# Load and optimize for faster streaming
optimized_view = ds.load_view("category_a", optimize=True, tensors=["images", "labels"])
```

---

### ds.save_view()

#### Overview

Save the current dataset view as a virtual dataset (VDS). The saved view can be loaded later by `view_id` if it is saved inside the parent dataset, or loaded by path with `muller.load()` when saved to an external path.

#### Signature

```python
ds.save_view(
    message=None,
    path=None,
    view_id=None,
    optimize=False,
    tensors=None,
    num_workers=0,
    scheduler="threaded",
    ignore_errors=False,
    **ds_args,
)
```

#### Parameters

- **message** (`str`, optional): Custom message stored with the view. If omitted, the dataset query string is used when available.
- **path** (`str` or `pathlib.Path`, optional): External path where the VDS is saved. If omitted, the VDS is saved under the source dataset's `.queries` directory.
- **view_id** (`str`, optional): Unique id for this view. If omitted, MULLER generates a deterministic hash-based id for the view.
- **optimize** (`bool`, optional): If `True`, copy and rechunk the required data into the VDS. Defaults to `False`.
- **tensors** (`list[str]`, optional): Tensor names to copy when `optimize=True`. If omitted, all tensors are copied.
- **num_workers** (`int`, optional): Number of workers used for optimization. Only applies when `optimize=True`. Defaults to `0`.
- **scheduler** (`str`, optional): Scheduler used for optimization. Supported values include `"serial"`, `"threaded"`, `"processed"`, and `"distributed"`. Defaults to `"threaded"`.
- **ignore_errors** (`bool`, optional): Skip samples that fail while saving optimized views. Only applies when `optimize=True`. Defaults to `False`.
- **ds_args**: Additional dataset creation arguments used when `path` is specified.

#### Returns

- **str**: Path to the saved VDS.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Create and save a filtered view
filtered = ds.filter("labels == 5")
vds_path = filtered.save_view(view_id="label_5_samples")

# Add a message
view = ds.filter("score > 80")
view.save_view(view_id="high_scores", message="Samples with score > 80")

# Save to an external dataset path
external_path = view.save_view(path="./views/high_scores", view_id="high_scores")
external_view = muller.load(external_path)
```

#### Notes

- The public parameter is `view_id`; older examples that use `id=` or `view_name` do not match the current code signature.
- Saving an in-place view requires a committed source dataset head. If the dataset has uncommitted HEAD changes, `save_view()` raises `DatasetViewSavingError`.
- External views saved with `path=` are not listed by the parent dataset's `get_views()` and cannot be loaded through `load_view()` on the parent dataset.

---

### ds.delete_view()

#### Overview

Delete a saved in-place view from the dataset by view id.

#### Signature

```python
ds.delete_view(view_id)
```

#### Parameters

- **view_id** (`str`): Id of the view to delete.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Delete a view
ds.delete_view("old_view")

# Delete multiple views
for view_id in ["temp_view1", "temp_view2", "test_view"]:
    ds.delete_view(view_id)
```

---

### ds.get_views()

#### Overview

Get saved view entries for the dataset.

#### Signature

```python
ds.get_views(commit_id=None)
```

#### Parameters

- **commit_id** (`str`, optional): If provided, return only views whose source dataset version matches this commit id. If omitted, return views from all commits.

#### Returns

- **list[ViewEntry]**: View entry objects. Each entry exposes properties such as `id`, `message`, `commit_id`, `virtual`, `query`, `tql_query`, and `source_dataset_path`.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Get all views
views = ds.get_views()
print([view.id for view in views])

# Load each view
for entry in views:
    view = entry.load()
    print(f"{entry.id}: {len(view)} samples")

# Check if view exists
if any(entry.id == "my_view" for entry in ds.get_views()):
    view = ds.load_view("my_view")
```

---

### ds.get_view()

#### Overview

Get the metadata entry for a saved view by id. Call `.load()` on the returned entry to load the view dataset.

#### Signature

```python
ds.get_view(view_id)
```

#### Parameters

- **view_id** (`str`): Id of the saved view to retrieve.

#### Returns

- **ViewEntry**: View metadata entry for the requested view.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

entry = ds.get_view("label_5_samples")
print(entry.id, entry.message, entry.commit_id)

view = entry.load()
```
