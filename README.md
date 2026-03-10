<div align="center">
    <img src="docs/figures/logo.png" width="500">
</div>

## MULLER: A Multimodal Data Lake Format for Collaborative AI Data Workflows
<div align="center">
    <img src="docs/figures/motivation-github.png" width="700">
</div>

## Overview

Modern AI datasets require collaborative workflows where multiple engineers work on parallel branches, perform LLM-assisted annotation, and merge changes—similar to Git workflows for code. However, existing data lake formats (Parquet, Lance, Iceberg, Deep Lake) lack native support for such collaborative patterns.

**MULLER** is a multimodal data lake format designed for collaborative AI data workflows with:

- **Multimodal data support**: 12+ data types (scalars, vectors, text, images, videos, audio) with 20+ compression formats (LZ4, JPG, PNG, MP3, MP4, etc.)
- **Hybrid search engine**: Joint queries across vector, text, and scalar data with low-latency random access
- **Git-like versioning**: Commit, branch, merge with fine-grained row-level updates and three-way merge conflict resolution
- **Seamless integration**: Works with PyTorch, TensorFlow, and LLM/MLLM training pipelines

📺 [Video demo](https://www.youtube.com/watch?v=okHzhbp7an0) | 📖 [Documentation](https://the-ai-framework-and-data-tech-lab-hk.github.io/MULLER/) | 🔗 [API Reference](https://the-ai-framework-and-data-tech-lab-hk.github.io/MULLER/api/top-level-functions/)

## Quick Start

### Choose Your Interface

MULLER offers two ways to work with your data:

**🤖 Natural Language** (via Agent Skills)
- Use Claude Code or compatible AI assistants
- Describe operations in plain English
- Best for: Interactive exploration, rapid prototyping
- [Jump to Natural Language guide →](#natural-language-interface)

**🐍 Python API**
- Direct programmatic control
- Full feature access
- Best for: Production pipelines, automation
- [Jump to Python API guide →](#python-api)

### Installation

#### Prerequisites

- Python >= 3.11
- CMake >= 3.22.1 (required for building C++ extensions)
- A C++17 compatible compiler (tested with gcc 11.4.0)
- Linux or macOS (tested on Ubuntu 22.04)

#### Install MULLER

1. (Recommended) Create a new Conda environment:
```bash
conda create -n muller python=3.11
conda activate muller
```

2. Clone and install:
```bash
git clone https://github.com/The-AI-Framework-and-Data-Tech-Lab-HK/MULLER.git
cd MULLER
chmod 777 muller/util/sparsehash/build_proj.sh
pip install .   # Use `pip install . -v` to view detailed build logs
```

3. Verify installation:
```python
import muller
print(muller.__version__)
```

**Optional installations:**
- Development mode: `pip install -e .`
- Skip C++ modules: `BUILD_CPP=false pip install .`

## Natural Language Interface

MULLER includes [Agent Skills](https://agentskills.io) that let you manage datasets through natural language when using **Claude Code** or compatible AI assistants.

**Compatibility note:** Skills are located in `.claude/skills/` and optimized for Claude Code. Other IDEs (Cursor, Windsurf) may require different integration approaches.

### Available Operations

**Dataset Management** (muller-dataset)
- "Create an image classification dataset at ./my_photos with jpg compression"
- "Add all images from the ./data/ folder to my dataset"
- "Show me all samples where label equals 5"
- "Get statistics and summary for my dataset"

**Version Control** (muller-version-control)
- "Commit my dataset changes with message 'Added 100 new samples'"
- "Create a new branch called dev-1"
- "Merge the dev-1 branch into main"
- "Show me the commit history"

**Advanced Queries** (muller-advanced-query)
- "Create a vector index for embeddings using HNSW"
- "Find the top 10 most similar vectors to my query"
- "Create an inverted index for text search on descriptions"
- "Aggregate data by category and count samples"

**Export & Integration** (muller-export)
- "Export my dataset to Parquet format"
- "Convert the embeddings tensor to NumPy array"
- "Export dataset to JSON for my web API"

For detailed skill documentation, see [.claude/skills/](.claude/skills/).

### Using Skills in Other AI Coding Agents

The MULLER skills are **platform-agnostic** and can be used with other AI coding assistants beyond Claude Code, including Cursor, Windsurf, Codex, OpenClaw, and similar tools.

**How to use with other agents:**

1. Copy the skills directory to your project:
   ```bash
   cp -r .claude/skills /path/to/your/project/.claude/skills
   ```

2. Ensure your AI coding agent can:
   - Discover and read SKILL.md files in `.claude/skills/*/SKILL.md`
   - Execute Python scripts via command line
   - Parse JSON output from the scripts

3. (Optional) Adapt the path if your agent uses a different convention:
   ```bash
   # For Cursor
   cp -r .claude/skills /path/to/your/project/.cursor/skills

   # For a platform-agnostic location
   cp -r .claude/skills ~/.ai-skills/muller/
   ```

The skills will work as long as your AI agent can execute the documented Python commands and parse the JSON responses.

## Python API

### Create a Dataset

MULLER supports 12+ data types across different modalities with 20+ compression formats:

| htype | sample_compression | dtype |
| --- | --- | --- |
| image | bmp, gif, jpg, jpeg, png, tiff, webp, etc. | `uint8` |
| video | mp4, mkv, avi | `uint8` |
| audio | flac, mp3, wav | `float64` |
| class_label | None (null) or lz4 | `uint32` |
| bbox | None (null) or lz4 | `float32` |
| text | None (null) or lz4 | `str` |
| vector | None (null) or lz4 | `float32` |
| generic | None (null) or lz4 | `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `float32`, `float64`, `bool` |

```python
import muller

# Create dataset
ds = muller.dataset(path='test_dataset/', overwrite=True)

# Create tensors (columns)
ds.create_tensor('my_images', htype='image', sample_compression='jpg')
ds.create_tensor('labels', htype='generic', dtype='int')
ds.create_tensor('categories', htype='text')
ds.create_tensor('description', htype='text')

# Append data
with ds:
    ds.my_images.extend([muller.read(img_path_0), muller.read(img_path_1), muller.read(img_path_2)])
    ds.labels.extend([0, 1, 2])
    ds.categories.extend(["cat", "cat", "dog"])
    ds.description.extend([
        "A majestic Maine Coon cat perched on a wooden bookshelf",
        "A tuxedo cat stretching lazily across a velvet sofa",
        "A Golden Retriever sprinting across a green meadow"
    ])

# Explore data
ds.summary()
ds.labels[0:2].numpy()
ds.my_images[1].numpy()
```

### Query Data

MULLER provides comprehensive query capabilities:

```python
# Full-text search (requires inverted index)
ds.commit()
ds.create_index_vectorized("description")
res = ds.filter_vectorized([("description", "CONTAINS", "cat")])

# Comparison and complex queries
res_1 = ds.filter_vectorized([("labels", ">", 1)])
res_2 = ds.filter_vectorized([("description", "LIKE", "ca[t]")])
res_3 = ds.filter_vectorized([("description", "CONTAINS", "cat"), ("labels", "<", 4)], ["AND"])

# Aggregation
res_4 = ds.aggregate_vectorized(
    group_by_tensors=['categories'],
    selected_tensors=['labels', 'categories'],
    aggregate_tensors=["*"]
)

# Vector similarity search
import numpy as np
ds_vec = muller.dataset(path="test_vec/", overwrite=True)
ds_vec.create_tensor("embeddings", htype="vector", dtype="float32", dimension=32)
ds_vec.embeddings.extend(np.random.rand(10000, 32).astype(np.float32))

ds_vec.commit()
ds_vec.create_vector_index("embeddings", index_name="hnsw", index_type="HNSWFLAT", metric="l2")
ds_vec.load_vector_index("embeddings", index_name="hnsw")

query = np.random.rand(100, 32)
distances, indices = ds_vec.vector_search(query_vector=query, tensor_name="embeddings", index_name="hnsw", topk=10)
```

### Version Control

```python
# Create and switch branches
ds = muller.load("test_dataset@main")
ds.checkout("dev-1", create=True)

# Make changes
ds.my_images.extend([muller.read(img_path_5)])
ds.labels.extend([5])
ds.categories.extend(["bird"])
ds.description.extend(["A vibrant Macaw perched on a wooden branch"])
ds.labels[2] = 20  # Update existing row
ds.pop(0)  # Delete row

ds.commit('Added bird samples')

# Merge branches
ds.checkout('main')
ds.merge('dev-1')

# View history and diff
ds.log()
ds.diff(id_1="main", id_2="dev-1")
```

For complete API documentation, see [MULLER API Reference](https://the-ai-framework-and-data-tech-lab-hk.github.io/MULLER/api/top-level-functions/).

## Advanced Topics

### Collaborative Data Annotation Workflow

This example demonstrates a complete collaborative workflow with multiple branches, conflict detection, and three-way merges:

**1. Create branch dev-1 and make changes:**
```python
ds = muller.load("test_dataset@main")
ds.checkout("dev-1", create=True)

# Append, update, delete
ds.my_images.extend([muller.read(img_path_50), muller.read(img_path_60)])
ds.labels.extend([50, 60])
ds.categories.extend(["cat", "bird"])
ds.description.extend(["An inquisitive ginger tabby cat", "A sleek all-black cat"])
ds.labels[3] = 30  # Update
ds.pop(1)  # Delete
ds.commit('commit on dev-1')
```

**2. Create branch dev-2 with different changes:**
```python
ds = muller.load("test_dataset@main")
ds.checkout("dev-2", create=True)

ds.my_images.extend([muller.read(img_path_500), muller.read(img_path_600)])
ds.labels.extend([500, 600])
ds.categories.extend(["cat", "dog"])
ds.description.extend(["A fluffy orange tabby", "A golden retriever"])
ds.labels[3] = 300  # Conflicting update
ds.pop([1, 2])  # Different deletes
ds.commit('commit on dev-2')
```

**3. Fast-forward merge dev-1:**
```python
ds.checkout('main')
ds.merge('dev-1', pop_resolution='theirs')
```

**4. Three-way merge dev-2 with conflict resolution:**
```python
# Detect conflicts
conflict_cols, conflict_records = ds.detect_merge_conflict("dev-2", show_value=True)

# Merge with resolution strategy
ds.merge("dev-2",
         append_resolution="both",      # Keep appends from both branches
         pop_resolution="ours",          # Keep our deletes
         update_resolution="theirs")     # Use their updates
```

**5. Schema evolution across branches:**
```python
import numpy as np
ds.checkout("dev-3", create=True)
ds.create_tensor("features", htype="generic", dtype="float")
ds.features.extend(np.arange(0, 1.1, 0.1))
ds.commit()

ds.checkout("main")
ds.merge("dev-3")  # Schema propagates to main
```

**6. View history and diffs:**
```python
ds.log()
ds.branches
ds.diff(id_1="dev-1", id_2="dev-2")
ds.direct_diff(id_1="dev-1", id_2="dev-2", as_dataframe=True)
```

## Research & Reproduction

To reproduce the experiment results from our paper, see [exp_scripts/README.md](https://github.com/spencerr221/MULLER/blob/main/exp_scripts/README.md).
