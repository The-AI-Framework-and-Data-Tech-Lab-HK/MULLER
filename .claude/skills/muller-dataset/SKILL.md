---
name: muller-dataset
description: Create, query, and manage MULLER datasets (multimodal data lake with version control). Use when user wants to work with datasets, create columns, append data, query samples, or inspect dataset information.
compatibility: Requires Python 3.11+, muller package installed
---

# MULLER Dataset Management

## IMPORTANT: How to Use This Skill

**DO NOT create new Python files.** Always use the existing scripts provided in this skill:
- Use `scripts/dataset_manager.py` for dataset and column management
- Use `scripts/data_operations.py` for data CRUD operations

Execute these scripts directly with `python3` command. Never write new scripts to the project root.

## DEMO MODE: Speed Instructions

When executing operations:
- Run commands immediately without pre-explanation
- Do not summarize or explain results after execution
- Only show the JSON output from the script
- Skip all follow-up suggestions unless user asks

## When to Use This Skill

Use this skill when the user wants to:
- Create or load MULLER datasets
- Add data to datasets (images, text, labels, vectors, etc.)
- Query or filter dataset samples
- Manage dataset structure (create/delete/rename columns)
- Inspect dataset information (summary, statistics)
- Import data from files or directories

## Available Scripts

### scripts/dataset_manager.py

Manages dataset lifecycle and structure.

**Operations:**
- `create` - Create a new dataset
- `load` - Load existing dataset info
- `delete` - Delete a dataset
- `info` - Get dataset information
- `stats` - Get dataset statistics
- `create-column` - Create a new column
- `delete-column` - Delete a column
- `rename-column` - Rename a column
  - Legacy `create-tensor`, `delete-tensor`, and `rename-tensor` commands remain accepted.

**Usage:**
```bash
# Create dataset
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py create --path ./my_dataset

# Create with columns
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py create --path ./my_dataset \
  --columns "images:image:jpg,labels:class_label:uint32"

# Get info
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py info --path ./my_dataset

# Create column
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py create-column --path ./my_dataset \
  --name embeddings --htype vector --dtype float32
```

### scripts/data_operations.py

Handles data CRUD operations.

**Operations:**
- `append` - Add single sample
- `extend` - Add multiple samples
- `update` - Update existing sample
- `delete` - Delete samples
- `query` - Query and filter samples
- `import` - Import data from files

**Usage:**
```bash
# Append sample
python3 .claude/skills/muller-dataset/scripts/data_operations.py append --path ./my_dataset \
  --data '{"images": "path/to/img.jpg", "labels": 1}'

# Query samples
python3 .claude/skills/muller-dataset/scripts/data_operations.py query --path ./my_dataset \
  --filter "labels > 5" --limit 10

# Import from file
python3 .claude/skills/muller-dataset/scripts/data_operations.py import --path ./my_dataset \
  --source data.jsonl
```

## Quick Reference

For detailed workflows and examples, check references/ directory.
