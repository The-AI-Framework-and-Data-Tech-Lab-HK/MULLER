---
name: muller-advanced-query
description: Advanced query operations for MULLER datasets - indexing, vector search, aggregation, and complex filtering. Use when user wants to create indexes, perform vector similarity search, or run aggregation queries.
compatibility: Requires Python 3.11+, muller package installed
---

# MULLER Advanced Query

## IMPORTANT: How to Use This Skill

**DO NOT create new Python files.** Always use the existing script:
- Use `scripts/advanced_query.py` for all advanced query operations

Execute the script directly with `python3` command. Never write new scripts to the project root.

## DEMO MODE: Speed Instructions

When executing operations:
- Run commands immediately without pre-explanation
- Do not summarize or explain results after execution
- Only show the JSON output from the script
- Skip all follow-up suggestions unless user asks

## When to Use This Skill

Use this skill when the user wants to:
- Create inverted indexes for text search
- Create vector indexes for similarity search
- Perform vector similarity search
- Run aggregation queries (GROUP BY, COUNT, AVG, etc.)
- Complex filtering with multiple conditions
- Full-text search with CONTAINS operator

## Available Script

### scripts/advanced_query.py

Handles advanced query and indexing operations.

**Operations:**
- `create-index` - Create vectorized inverted index for text search (supports the C++ engine)
- `create-vector-index` - Create vector index (FLAT, HNSW, DISKANN)
- `load-vector-index` - Load vector index into memory
- `vector-search` - Perform vector similarity search
- `aggregate` - Run aggregation queries
- `filter-advanced` - Complex filtering with multiple conditions

**Usage:**
```bash
# Create vectorized inverted index for text search (Python engine, defaults)
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py create-index \
  --path ./my_dataset --tensors "description,title"

# Use the native C++ engine (local datasets only, fuzzy_match only)
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py create-index \
  --path ./my_dataset --tensors "description" --use-cpp

# Parallelize build/search on a large dataset
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py create-index \
  --path ./my_dataset --tensors "description" --use-cpp \
  --num-of-batches 16 --num-of-shards 16 --max-workers 16

# Create vector index
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py create-vector-index \
  --path ./my_dataset --tensor embeddings --index-name hnsw \
  --index-type HNSWFLAT --metric l2

# Vector search
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py vector-search \
  --path ./my_dataset --tensor embeddings --index-name hnsw \
  --query-file query.npy --topk 10

# Aggregation
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py aggregate \
  --path ./my_dataset --group-by categories --select labels,categories \
  --aggregate-tensors "*"
```

## Inverted Index Options (`create-index`)

`create-index` builds a **vectorized** inverted index via `ds.create_index_vectorized()`.
The dataset must be committed first. Options:

- `--tensors` (required): comma-separated columns; each is indexed separately.
- `--index-type`: `fuzzy_match` (default, tokenized full-text) or `exact_match`
  (whole-value match). `exact_match` is **not** supported together with `--use-cpp`.
- `--use-cpp`: use the native C++ engine. **Local datasets only**, **`fuzzy_match` only**,
  and requires the compiled extension (built during a standard `pip install .`; unavailable
  if installed with `BUILD_CPP=false`). Significantly faster than the Python engine.
- `--force-create`: rebuild from scratch instead of appending to an existing index.

**Choosing the parallelism options** — `--max-workers` is only an upper bound; the real
parallelism is capped by how many work units exist:

| Option | Governs | Effective parallelism | Default |
|---|---|---|---|
| `--num-of-batches` | Build (index creation) | `min(num_of_batches, max_workers)` | `1` |
| `--num-of-shards` | Search & optimize + storage layout | `min(num_of_shards, max_workers)` | `1` |
| `--max-workers` | Cap on the above | — | `16` |

- `--num-of-batches` splits the row range into independent build tasks. Because the default
  is `1`, **raising `--max-workers` alone does not parallelize the build** — increase
  `--num-of-batches` first.
- `--num-of-shards` hash-partitions the index and is **fixed at creation time** (change it
  later only via the Python API `ds.reshard_index()`). More shards means more query-time
  parallelism and smaller per-shard files.
- Rules of thumb: small datasets → keep defaults; large datasets → set `--num-of-batches`
  to roughly `--max-workers` and `--max-workers` near the CPU core count; high query
  concurrency / large vocabulary → raise `--num-of-shards`.

See the [Indexing API reference](https://the-ai-framework-and-data-tech-lab-hk.github.io/MULLER/api/dataset-query/#dscreate_index_vectorized)
for full details.

## Quick Reference

For detailed workflows and examples, check references/ directory.
