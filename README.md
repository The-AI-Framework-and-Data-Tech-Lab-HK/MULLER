<div align="center">
    <img src="figures/logo.png" width="500">
</div>

## MULLER: A Multimodal Data Lake Format for Collaborative AI Data Workflows
<div align="center">
    <img src="figures/motivation-github.pdf" width="700">
</div>

At modern training scales, AI datasets are curated collaboratively by multiple data engineers rather than a single user.
In a typical workflow, data engineers independently check out branches from a main dataset, perform LLM-assisted data annotation and exploration, and commit their changes. Some branches can be merged via fast-forward (e.g., merging _Branch 1_ at t<sub>2</sub>), while others require three-way merges with conflict detection as the main branch evolves (e.g., merging _Branch 2_ at t<sub>3</sub>). The merged dataset is then exported for downstream model fine-tuning.

However, existing data lake formats or systems such as Parquet, Lance, Iceberg, and Deep Lake do not unify these workflows or support collaborative three-way merges. To address this challenge, we propose MULLER, which is a novel Multimodal data lake format designed for collaborative AI data workflows, with the following key features:
* **Mutimodal data support** with than 12 data types of different modalities, including scalars, vectors, text, images, videos, and audio, with 20+ compression formats (e.g., LZ4, JPG, PNG, MP3, MP4, AVI, WAV).
* **Data sampling, exploration, and Analysis** through low-latency random access and fast scan.
* **Array-oriented hybrid search engine** that jointly queries vector, text, and scalar data.
* **Git-like data versioning** with support for commit, checkout, diff, conflict detection and resolution, as well as merge. Specifically, to the best of our knowledge, MULLER is the first data lake format to support _fine-grained row-level updates and three-way merges_ across multiple coexisting data branches.
* **Seamless integration with LLM/MLLM data training and processing pipelines**.

Here is a [video demo](https://www.youtube.com/watch?v=okHzhbp7an0) of MULLER to demonstrate the basic functions.


## Getting Started

#### Prerequisites

- Python >= 3.11
- CMake >= 3.22.1 (required for building C++ extensions)
- A C++17 compatible compiler (tested with gcc 11.4.0)
- Linux or macOS (tested on Ubuntu 22.04)

#### 1. (Recommended) Create a new Conda environment
```bash
conda create -n muller python=3.11
conda activate muller
```
#### 2. Installation
* First, clone the MULLER repository.
```bash
git clone https://github.com/The-AI-Framework-and-Data-Tech-Lab-HK/MULLER.git
cd MULLER
chmod 777 muller/util/sparsehash/build_proj.sh  # You may need to modify the script permissions.
```
* [Dafault] Install from code
```
pip install .   # Use `pip install . -v` to view detailed build logs
```
* [Optional] Development (editable) installation
```bash
pip install -e .
```
* [Optional] Skip building C++ modules

The Python implementation provides the same core functionality as the C++ modules.
If you only need the basic features of MULLER, you may skip building the C++ extensions:
```bash
BUILD_CPP=false pip install .
```
#### 3. Verify the Installation
```python
import muller
print(muller.__version__)
```

## Examples
#### 1. Create a MULLER Dataset
* Note: MULLER support 12+ data types of different modalities, including scalars, vectors, text, images, videos, and audio, with 20+ compression formats (e.g., LZ4, JPG, PNG, MP3, MP4, AVI, WAV).

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

```python
import muller

# Create an empty MULLER datatset
ds = muller.dataset(path='test_dataset/', overwrite=True)

# Create Columns
ds.create_tensor(name='my_images', htype='image', sample_compression='jpg')
ds.create_tensor('labels', htype='generic', dtype='int')
ds.create_tensor('categories', htype='text')
ds.create_tensor('description', htype='text')

# Append data
with ds:
    ds.my_images.extend([muller.read(img_path_0), muller.read(img_path_1), muller.read(img_path_2), muller.read(img_path_3), muller.read(img_path_4)])
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.categories.extend(["cat", "cat", "dog", "dog", "rabbit"])
    ds.description.extend(["A majestic long-haired Maine Coon cat perched on a wooden bookshelf, staring intently at a tree outside with its bright amber eyes.", 
                           "A domestic short-hair cat with a distinctive tuxedo pattern stretching lazily across a velvet sofa in a dimly lit living room.", 
                           "An energetic Golden Retriever with bright amber eyes sprinting across a vibrant green meadow, its fur glistening under the afternoon sun as it chases a bright yellow tennis ball.", 
                           "A focused German Shepherd sitting patiently on a cobblestone street, wearing a professional service harness and looking up at its handler for the next command.", 
                           "A soft, white lop-eared rabbit with bright eyes nestled in a patch of clover, twitching its pink nose while nibbling on a fresh garden carrot."])

```

#### 2. Data Exploration and Analysis
```python
# Check the metadata and schema of the dataset
ds.summary()

# Investigate the details of data
ds.lables[2:4].numpy()
ds.my_images[3].numpy()
```

#### 3. Data Query
MULLER provides a comprehensive suite of query functionalities tailored for AI data lakes:
* Comparison Operators: Supports exact and range matching using `>`,`<`, `>=`, and `<=` for numerical types (`int`/`float`) where the tensor htype is generic.
* Equality and Inequality: Supports `==` and `!=` for `int`, `float`, `str`, and `bool` types (`generic` or `text` htypes). Users can optionally build inverted indexes to significantly accelerate retrieval performance.
* Full-Text Search: Supports the `CONTAINS` operator for `str` types (`text` htype), backed by an inverted index. For Chinese text, tokenization is handled by the open-source Jieba tokenizer.
* Pattern Matching: Supports `LIKE` for regular expression matching on `str` types (`text` htype).
* Boolean Logic: Supports complex query compositions using `AND`, `OR`, and `NOT` logical connectors.
* Pagination: Supports query results with `OFFSET` and `LIMIT` clauses for efficient data sampling.
* Data Aggregation: Supports standard SQL-like aggregation workflows, including `SELECT`, `GROUP BY`, and `ORDER BY`, alongside aggregate functions such as `COUNT`, `AVG`, `MIN`, `MAX`, and `SUM`.
* Vector Similarity Search: Supports high-dimensional vector similarity retrieval based on IVFPQ, HNSW and DISKANN for AI-centric embedding analysis.

```python
# Example 1: Full-text search (inverted index construction needed)
ds.commit() # Before creating index, you must commit the data modification.
ds.create_index_vectorized("description")
res = ds.filter_vectorized([("description", "CONTAINS", "bright eyes")])
```

```python
# Example 2: Exact macth and complex query compositions
res_1 = ds.filter_vectorized([("labels", ">", 1)])
res_2 = ds.filter_vectorized([("description", "LIKE", "ca[t]")])
res_3 = ds.filter_vectorized([("labels", "<", 2)], ["NOT"])
res_4 = ds.filter_vectorized([("description", "CONTAINS", "cat"), ("labels", "<", 4)],  ["AND"])
res_5 = ds.filter_vectorized([("description", "CONTAINS", "bright eyes"), ("labels", "<", 4)],  ["OR"], offset=2, limit=1)
```

```python
# Example 3: Aggregation
res_6 = ds.aggregate_vectorized(
        group_by_tensors=['categories'],
        selected_tensors=['labels', 'categories'],
        order_by_tensors=['labels'],
        aggregate_tensors=["*"],)
```

```python
# Example 4: Vector search

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

#### 4. Collaborative Data Annotation based on Git-like versioning

Step 1. Checkout a new branch (dev-1), and conduct data annotations. Note: The annotation may be assisted by LLMs.
```python
ds = muller.load(path="test_dataset@main") # You can use `@` to denote the target branch
ds.checkout("dev-1", create=True)
# Append rows
ds.labels.extend([50, 60, 70])
ds.categories.extend(["cat", "bird", "cat"])
ds.description.extend(["An inquisitive ginger tabby cat standing on its hind legs, reaching out with its paws toward a feathered toy during an active play session.", 
                       "A sleek all-black cat curled up in a deep sleep on a soft, cream-colored fleece blanket, enjoying a quiet afternoon nap in a cozy indoor setting.", 
                       "A vibrant Macaw with brilliant red, blue, and yellow plumage, perched firmly on a weathered wooden branch while looking curiously into the distance."])
# Update rows
ds.labels[3] = 30
# Delete rows
ds.pop(1)
ds.commit('commit on dev-1')
```

## Reproduction steps for the experiment results in our paper

Please refer to [exp_scripts/README.md](https://github.com/spencerr221/MULLER/blob/main/exp_scripts/README.md) for the detailed steps.
