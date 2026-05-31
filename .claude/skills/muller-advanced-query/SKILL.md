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
- `create-index` - Create inverted index for text search
- `create-vector-index` - Create vector index (FLAT, HNSW, DISKANN)
- `load-vector-index` - Load vector index into memory
- `vector-search` - Perform vector similarity search
- `aggregate` - Run aggregation queries
- `filter-advanced` - Complex filtering with multiple conditions

**Usage:**
```bash
# Create inverted index for text search
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py create-index \
  --path ./my_dataset --tensors "description,title"

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

## Quick Reference

For detailed workflows and examples, check references/ directory.
