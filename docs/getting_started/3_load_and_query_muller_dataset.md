# Basic Dataset Operations (Load, Insert, Update, Delete, Query)

## 1. Loading an Existing Dataset

By default, the **`main`** branch of the dataset is loaded:

```python
>>> ds = muller.load(path="my_muller_dataset")
```

To load a specific branch (if available), append `@{branch_name}` to the path:

```python
>>> ds = muller.load(path="my_muller_dataset@dev")
```

To load a specific commit (if available), append `@{commit_id}` to the path:

```python
>>> ds = muller.load(path="my_muller_dataset@3e49cded62b6b335c74ff07e97f8451a37aca7b2")
```

* The usage of `path` is the same as described in [Creating an Empty Dataset]. Please add the appropriate prefix based on the storage backend.
* In addition to `muller.load()`, you may also use `muller.dataset()` to load an existing dataset (do not set overwrite=True).
* For details on branches and commit versions, see [Section 5: Version Management].
* Additional parameters and advanced usage are documented in the API reference: [muller.load()].

## 2. Inspecting Dataset Metadata
To view schema-level information for all columns in a dataset:

```python
>>> import muller
>>> ds = muller.load("my_muller_dataset")
>>> ds.summary()
ensor     htype          shape           dtype  compression
 -------   -------        -------         -------  ------- 
my_label   generic         (4, 1)          int32    None   
my_photos   image   (4, 400, 300:500, 3)   uint8    jpeg   
 my_text    text           (4, 1)           str     None
```

- Usage reference: [`muller.summary()`]

Additional APIs for inspecting dataset properties include:

- List all columns: [`dataset.tensors()`]
- Get the number of samples (rows): [`dataset.num_samples()`]
- Get the maximum and minimum lengths across columns:  
  [`dataset.max_len()`], [`dataset.min_len()`]
- Dataset-level statistics (e.g., per-column min / max / median / variance):  
  [`dataset.statistics()`]

APIs for inspecting detailed tensor (column) metadata include:

- Column type: [`tensor.htype`]
- Column data type: [`tensor.dtype`]
- Tensor shape (column-level or per-sample):  
  [`tensor.shape_interval`], [`tensor.shape`]
- Tensor dimensionality: [`tensor.ndim`]
- Number of samples (rows) in a column:  
  [`tensor.num_samples`], [`tensor.__len__()`]
- Source file metadata for a specific sample:  
  [`tensor.sample_info`]  
  *(Effective for recovering image/video/audio metadata; requires  
  `create_sample_info_tensor=True` when calling `create_tensor()`.)*

## 3. Adding Data

Data can be appended to tensor columns using the same APIs and workflows
described in Section 3.1 (Step 3: Adding Data to Tensor Columns).

## 4. Deleting Samples or Datasets

#### Removing a sample by index:

```python
>>> ds.pop(2)  # Removes the sample at index 2
```

#### Removing multiple samples by index list:

```python
>>> ds.pop([1, 2, 4, 5])  # Removes samples at indices [1, 2, 4, 5]
```

#### Deleting a tensor column and all of its data:

```python
>>> ds.delete_tensor(<tensor_name>)
# Note: `large_ok` defaults to False. When deleting a column with a large
# number of samples, set `large_ok=True` to explicitly confirm the operation
# and prevent accidental data loss.
```
- For detailed usage, see [`pop()`] and [`delete_tensor()`].
- **Note:** To ensure dataset integrity, data can only be deleted **by entire rows or entire columns**.

#### Deleting a dataset (Method 1):
For an already loaded dataset, invoke the delete operation directly on the dataset object.
```python
>>> ds.delete()
```

#### Deleting a dataset (Method 2):
For a dataset that is not currently loaded, you can delete it using the `muller` library directly.
```python
>>> gtn_f.delete(path="/your/data/path/", creds={'optional'})
```
- For detailed usage, see [`muller.core.dataset.delete()`] and [`muller.delete()`].

## 5. Inspecting Data by random access (via row id and column name) and full scan

#### Access the i-th row in a tensor column (first access the column, then the row):

```python
>>> ds.tensors
{'my_label': Tensor(key='my_label'),
 'my_photos': Tensor(key='my_photos'),
 'my_text': Tensor(key='my_text')}

>>> ds.my_label[0].numpy()       # Returns the sample as a NumPy array
array([1], dtype=int32)

>>> ds.my_label[0].data()['value']  # Returns the sample value
array([1], dtype=int32)

>>> ds.my_label[0].tobytes()      # Returns the sample as bytes
b'\x01\x00\x00\x00'
```

#### Access multiple rows of in tensor column (first access the column, then the row):
```python
>>> ds.my_label[1:4].numpy()   # Returns the sample as a NumPy array
array([[2],
       [3],
       [4]], dtype=int32)
>>> ds.my_label[1:4].numpy(aslist=True)   # Returns the sample as a list of NumPy arrays
[array([2], dtype=int32), array([3], dtype=int32), array([4], dtype=int32)]
```
#### Access the value of a column of the i-th row (first access the row, then the column):

```python
>>> ds[0].my_label.numpy()
array([1], dtype=int32)
```
#### Access the value of columns in multiple rows (first access the row, then the column):

```python
>>> ds[1:4].my_label.numpy()  # You may alsi use .data()['value']
array([[2],
       [3],
       [4]], dtype=int32)
```
Lazy loading for rows and columns involves several `Tensor`-related APIs, each with additional optional parameters. Refer to the API documentation for more details
- [[tensor.numpy()]]. Note that there are three important parameters:
> * `aslist` (`bool`):  
>   If `True`, returns data as a list of `np.ndarray`s. Recommended for dynamic-shape tensors.  
>   If `False`, returns a single `np.ndarray`. May raise an error if samples have dynamic shapes.  
>   Default: `False`.

> * `fetch_chunks` (`bool`):  
>   If `True`, reads the entire chunk containing the sample.  
>   If `False`, only reads the bytes needed for the sample.  
>   **Exceptions:** Even if `False`, it will be automatically set to `True` if:  
>     1. The tensor is `ChunkCompressed`.  
>     2. The chunk being accessed contains more than 128 samples.  
>   Default can be adjusted as needed.

> * `asrow` (`bool`):  
>   If `True`, returns samples in **row-oriented** format as a list of dictionaries, one dict per row.  
>   If samples have inconsistent lengths, an error will be raised (use `False` to avoid this).  
>   If `False`, returns data in **column-oriented** format as a dictionary, where each key maps to a list containing the column data.

- [[tensor.data()]()]
- [[tensor.tobytes()]()]
- [[tensor.text()]()]
- [[tensor.dict()]()]
- [[tensor.list()]()]

## 6. Update Data
#### Method 1: Directly update the i-th row in a given column
```python
ds.my_tensor[i] = gtn_f.read("image.jpg")
```

#### Method 2: Update data by specifying the column name and update values via the `update` API
```python
ds[i].update({"my_tensor": gtn_f.read("image.jpg")})
```

* Method 1 API: [[tensor.\__setitem\__()]()]
* Method 2 API: [[dataset.update()]()]

## 7. Query Data
MULLER provides a comprehensive suite of query functionalities tailored for AI data lakes:
* Comparison Operators: Supports exact and range matching using `>`,`<`, `>=`, and `<=` for numerical types (`int`/`float`) where the tensor htype is generic.
* Equality and Inequality: Supports `==` and `!=` for `int`, `float`, `str`, and `bool` types (`generic` or `text` htypes). Users can optionally build inverted indexes to significantly accelerate retrieval performance.
* Full-Text Search: Supports the `CONTAINS` operator for `str` types (`text` htype), backed by an inverted index. For Chinese text, tokenization is handled by the open-source Jieba tokenizer.
* Pattern Matching: Supports `LIKE` for regular expression matching on `str` types (`text` htype).
* Boolean Logic: Supports complex query compositions using `AND`, `OR`, and `NOT` logical connectors.
* Pagination: Supports query results with `OFFSET` and `LIMIT` clauses for efficient data sampling.
* Data Aggregation: Supports standard SQL-like aggregation workflows, including `SELECT`, `GROUP BY`, and `ORDER BY`, alongside aggregate functions such as `COUNT`, `AVG`, `MIN`, `MAX`, and `SUM`.
* Vector Similarity Search: Supports high-dimensional vector similarity retrieval based on IVFPQ, HNSW and DISKANN for AI-centric embedding analysis.

#### Example 1: Exact Match Query (without index acceleration)

Retrieve all values in the `test1` column (of type `generic`) that are greater than 2.
```python
>>> ds = gtn_f.dataset("temp_test", overwrite=True)
>>> ds.create_tensor(name="test1", htype="generic")   
>>> ds.test1.extend(np.random.randint(5, size=10000))
>>> ds_1 = ds.filter_vectorized([("test1", ">", 2)]) #Note: If the fourth parameter is not specified, it defaults to `False`, which is equivalent to the usage in the next line.
>>> ds_1 = ds.filter_vectorized([("test1", ">", 2, False)])
>>> ds_1.test1.numpy()
array([[4],
       [3],
       [4],
       ...,
       [4],
       [3],
       [4]], shape=(3922, 1))
>>> ds_1.filtered_index
[np.int64(1),
 np.int64(3),
 np.int64(5),
 np.int64(7), ...]
```

#### Example 2: Exact Match Query (without index acceleration) — Negation (NOT)

Retrieve all values in the `test1` column (of type `generic`) that **do NOT satisfy** the condition `> 2`.
```python
>>> ds = gtn_f.dataset("temp_test", overwrite=True)
>>> ds.create_tensor(name="test1", htype="generic")   
>>> ds.test1.extend(np.random.randint(5, size=10000))
>>> ds_1 = ds.filter_vectorized([("test1", ">", 2, False, "NOT")])
>>> ds_1.test1.numpy()
array([[2],
       [0],
       [0],
       ...,
       [1],
       [2],
       [0]], shape=(6078, 1))
```

#### Example 3: Exact Match Query (with Index Acceleration)

Retrieve all values in the `test1` column (of type `generic`) that are equal to 2.

> **Note:** Index creation only applies to data that has been recorded in a commit.  
> Before creating an index, **ensure that `ds.commit()` is executed** to consolidate all pending changes into a commit version.
```python
>>> ds = gtn_f.dataset("temp_test", overwrite=True)
>>> ds.create_tensor(name="test1", htype="generic")   
>>> ds.test1.extend(np.random.randint(5, size=10000))
>>> ds.commit()
>>> ds.create_index(["test1"]) # Create inverted index
>>> ds_1 = ds.filter_vectorized([("test1", "==", 2, True)])
>>> ds_1.test1.numpy()
array([[2],
       [2],
       [2],
       ...,
       [2],
       [2],
       [2]], shape=(2006, 1))
```

#### Example 4: Exact Match Query (without Index Acceleration)

Retrieve all values in the `test2` column (of type `text`) that are equal to `"hi"`.
```python
>>> ds = gtn_f.dataset("temp_test", overwrite=True)
>>> ds.create_tensor(name="test2",htype="text")
>>> with ds:
         ds.test2.extend(["hi", "bye", "oops", "hello", "world"]*2000)
>>> ds_2 = ds.filter_vectorized([("test2", "==", "hi" )])
>>> ds_2.test2.numpy()
array([['hi'],
       ['hi'],
       ['hi'],
       ...,
       ['hi'],
       ['hi'],
       ['hi']], shape=(2000, 1), dtype='<U2')
```

#### Example 5: Exact Match Query (with Index Acceleration)

Retrieve all values in the `test2` column (of type `text`) that are equal to `"hi"`.
```python
>>> ds = gtn_f.dataset("temp_test", overwrite=True)
>>> ds.create_tensor(name="test2",htype="text")
>>> with ds:
         ds.test2.extend(["hi", "bye", "oops", "hello", "world"]*2000)
>>> ds.commit()
>>> ds.create_index(["test2"])
>>> ds_2 = ds.filter_vectorized([("test2", "==", "hi", True)])
>>> ds_2.test2.numpy()
array([['hi'],
       ['hi'],
       ['hi'],
       ...,
       ['hi'],
       ['hi'],
       ['hi']], shape=(2000, 1), dtype='<U2')
```

#### Example 6: Exact Match Query with Logical Connectors (AND, OR, NOT)

Perform an exact-match query combining multiple conditions using logical connectors.  

> **Note:** The number of elements in `connector_list` must be **one less** than the number of elements in `condition_list`.
```python
>>> ds = gtn_f.dataset("temp_test", overwrite=True)
>>> ds.create_tensor(name="test3", htype="generic")
>>> ds.test3.extend(np.random.randint(5, size=10000))
>>> ds.create_tensor(name="test4", htype="generic")
>>> ds.test4.extend(np.random.randint(100, size=10000))
>>> ds_3 = ds.filter_vectorized(condition_list=[("test3", ">", 2), ("test3", "<=", 4), ("test4", "<", 60, False, "NOT")], connector_list=["AND", "OR"]) 
>>> len(ds_3)
6319
```

#### Example 7: Exact Match Query with Offset and Limit

Perform an exact-match query while applying **`offset`** and **`limit`** to control the subset of returned results.
```python
>>> ds = gtn_f.dataset("temp_test", overwrite=True)
>>> ds.create_tensor(name="test5", htype="generic")
>>> ds.test5.extend(np.arange(0, 100))
>>> ds_5 = ds.filter_vectorized([("test5", "<", 50), ("test5", ">=", 20)], ["AND"], offset=30, limit=10)  # Start the query from row 60 and return only 10 results.
>>> len(ds_5)
10
```

#### Example 8: Text Keyword Search (Index Required)

Perform a keyword-based search on text columns.  

(1) English keyword query

```python
>>> ds = gtn_f.dataset("temp_test", overwrite=True)
>>> ds.create_tensor(name="test6", htype="text")
>>> with ds:
         ds.test6.extend(["A majestic long-haired Maine Coon cat perched on a wooden bookshelf, staring intently at a tree outside with its bright amber eyes.",
                          "A soft, white lop-eared rabbit with bright eyes nestled in a patch of clover, twitching its pink nose while nibbling on a fresh garden carrot.",
                          "A focused German Shepherd sitting patiently on a cobblestone street, wearing a professional service harness and looking up at its handler for the next command.",
                          "A domestic short-hair cat with a distinctive tuxedo pattern stretching lazily across a velvet sofa in a dimly lit living room."]*2500)
>>> ds.commit()
>>> ds.create_index(["test6"]) 
>>> ds_6 = ds.filter_vectorized([("test6", "CONTAINS", "bright eyes")])
>>> ds_6.test6[:2].data()["value"]
[np.str_('A majestic long-haired Maine Coon ... with its bright amber eyes.'),
 np.str_('A soft, white lop-eared rabbit with bright eyes...'), ...]
# Jointly query
>>> ds_7 = ds.filter_vectorized([("test5", "<", 50), ("test5", ">=", 20), ("test6", "CONTAINS", "中山大学")], ["AND", "OR"], limit=10)
```

(2) Chinese keywrod query - The tokenization is implemented using the Chinese word segmentation tool **jieba**.

```python
>>> ds = gtn_f.dataset("temp_test", overwrite=True)
>>> ds.create_tensor(name="test6", htype="text")
>>> with ds:
         ds.test6.extend(["我毕业于香港科技大学。", "我毕业于中山大学。"]*5000)
>>> ds.commit()
>>> ds.create_index(["test6"]) #倒排索引
>>> ds_6 = ds.filter_vectorized([("test6", "CONTAINS", "中山大学")])
>>> ds_6.test6[:2].data()["value"]
[np.str_('我毕业于中山大学。'),
 np.str_('我毕业于中山大学。'), ...]
# Jointly query
>>> ds_7 = ds.filter_vectorized([("test5", "<", 50), ("test5", ">=", 20), ("test6", "CONTAINS", "中山大学")], ["AND", "OR"], limit=10)
```

#### Example 9: Regex-Based Text Matching (Using `LIKE`)

Perform text matching using regular expressions, with `LIKE` specified as the query operator.
```python
>>> ds = gtn_f.dataset("temp_test", overwrite=True)
>>> ds.create_tensor(name="test7", htype="text")
>>> ds.test7.extend(['A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'C0'])
>>> ds_8 = ds.filter_vectorized([("test7", "LIKE", "A[0-2]")])
>>> ds_8.test7.numpy()
array([['A0'],
       ['A1'],
       ['A2']], dtype='<U2')
```

#### Example 11: Aggregation and Group-By Statistics

Perform aggregation operations equivalent to the SQL statement:

```sql
select ori_query, ori_response, count(*)
from dataset
group by ori_query, ori_response
order by ori_query;
```
```python
>>> ds = gtn_f.dataset("temp_test", overwrite=True)
>>> tensors = ["ori_query", "ori_response", "query_analysis", "result", "score", "type"]
>>> ds.create_tensor("ori_query", htype="text", exist_ok=True)
>>> ds.create_tensor("ori_response", htype="text", exist_ok=True)
>>> ds.create_tensor("query_analysis", htype="text", exist_ok=True)
>>> ds.create_tensor("result", htype="text", exist_ok=True)
>>> ds.create_tensor("score", htype="generic", exist_ok=True, dtype="float64")
>>> ds.create_tensor("type", htype="generic", exist_ok=True, dtype="float64")
>>> np_data = np.array([
  [
    "1. Use if-else statements to implement comparison logic",
    "if in1 > in2:",
    "{\"Fluency Score\": 5.0, \"Completeness Score\": 3.0}",
    "Overall Quality Score: 6",
    6.0,
    5.0
  ],
  [
    "Write a Python program that uses regular expressions to match mobile phone numbers.",
    "First, we need to import the re module.",
    "{\"Fluency Score\": 5.0, \"Completeness Score\": 2.0}",
    "Overall Quality Score: 0",
    0.0,
    4.0
  ],
  [
    "Compile in C language: calculate the surface area and volume of a cylinder",
    "Four variables are defined: `r`, `h`, `surface_area`, `volume`",
    "{\"Fluency Score\": 5.0, \"Completeness Score\": 4.0}",
    "Overall Quality Score: 6",
    6.0,
    6.0
  ],
  [
    "How to combine the three RGB channel values into a single RGB value",
    "In Java, the extracted R, G, and B channel values are added together to form an RGB value.",
    "{\"Fluency Score\": 5.0, \"Completeness Score\": 4.0}",
    "Overall Quality Score: 7",
    7.0,
    4.0
  ],
  [
    "How to combine the three RGB channel values into a single RGB value",
    "In Java, the extracted R, G, and B channel values are added together to form an RGB value.",
    "{\"Fluency Score\": 5.0, \"Completeness Score\": 4.0}",
    "Overall Quality Score: 7",
    7.0,
    4.0
  ],
  [
    "Design an intelligent customer service system based on deep learning",
    "A deep learning–based intelligent customer service system includes steps such as data preprocessing, model construction, model training, and prediction.",
    "{\"Fluency Score\": 5.0, \"Completeness Score\": 5.0}",
    "Overall Quality Score: 6",
    6.0,
    3.0
  ]
])
>>> for i, item in enumerate(tensors):
         ds[item].extend(np_data[:, i].astype(ds[item].dtype))
>>> result2 = ds.aggregate_vectorized(
        group_by_tensors=['ori_query', 'ori_response'],
        selected_tensors=['ori_query', 'ori_response'],
        order_by_tensors=['ori_query'],
        aggregate_tensors=["*"],
        )
>>> result2
array([["Write a Python program that uses regular expressions to match mobile phone numbers.", "First, we need to import the re module.", '1'],
       ["Compile in C language: calculate the surface area and volume of a cylinder", "Four variables are defined: `r`, `h`, `surface_area`, `volume`", '1'],
       ["How to combine the three RGB channel values into a single RGB value", "In Java, the extracted R, G, and B channel values are added together to form an RGB value.", '2'],
       ["Design an intelligent customer service system based on deep learning", "A deep learning–based intelligent customer service system includes steps such as data preprocessing, model construction, model training, and prediction.", '1'],
       ["1. Use if-else statements to implement comparison logic", "if in1 > in2:", '1']], dtype='<U38')
```
* For detailed usage, please refer to the following API documentation:

> * filter_vectorized(): vectorized query interface for conditional filtering
> * aggregate_vectorized(): vectorized aggregation interface (e.g., group by + count(*))
> * create_index(): interface for creating inverted indexes

#### Example 12: Vector Search

```python
# Create a sample dataset
import numpy as np
ds_vec = muller.dataset(path="test_data_vec/", overwrite=True)
ds_vec.create_tensor(name="embeddings", htype="vector", dtype="float32", dimension=32)
ds_vec.embeddings.extend(np.random.rand(320000).reshape(10000, 32).astype(np.float32))

# Create index
ds_vec.commit()
ds_vec.create_vector_index("embeddings", index_name="flat", index_type="FLAT", metric="l2")
ds_vec.create_vector_index("embeddings", index_name="hnsw", index_type="HNSWFLAT", metric="l2", ef_construction=40, m=32)

# Vector search
q = np.random.rand(1000, 32)
ds_vec.load_vector_index("embeddings", index_name="flat")
res_7 = ds_vec.vector_search(query_vector=q, tensor_name="embeddings", index_name="flat", topk=1)
_, ground_truth = res_7

ds_vec.load_vector_index("embeddings", index_name="hnsw")
res_8 = ds_vec.vector_search(query_vector=q, tensor_name="embeddings", index_name="hnsw", ef_search=16)
_, res_id = res_8

# Compute the recall
recall = np.ones(len(res_id))[(ground_truth==res_id).flatten()].sum() / len(res_id)
```

## 8. Saving and Loading Dataset Views
Each query result can be persisted as a materialized view and assigned a unique ID. For subsequent identical queries, the corresponding materialized view can be loaded directly using this ID, avoiding redundant computation.
```python
my_view = ds.filter([("lable", "==", 1)])
my_view.save_view(id='my_query_id')  # Note: need to specify the `id`!
my_view = ds.load_view(id='my_query_id') # Note: need to specify the `id`!
```

Materialized view–related APIs support additional parameters to optimize read and write performance, such as `optimize` and `num_workers`. For details, please refer to the view-related API documentation.

## 9. Other Dataset-Related APIs

For the complete API reference, see [[Datasets]()] and its subpages. Key APIs include:

* Dataset copying:
> * Full copy including all branches: [dataset.deepcopy()]
> * Copy only the latest commit on the main branch: [dataset.copy()]
* Dataset rechunking (optimize chunk sizes for each tensor): [dataset.rechunk()]
* Export dataset to MindRecord: [dataset.to_mindrecord()]
* Export dataset to JSON: [dataset.to_json()]
* Export dataset to Arrow: [dataset.to_arrow()]
* Convert dataset to a pandas DataFrame (for inspection and visualization): [dataset.to_dataframe()]
