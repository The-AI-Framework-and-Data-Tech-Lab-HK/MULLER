# MULLER Agent Skills

This directory contains Agent Skills for managing MULLER datasets through natural language.

## Available Skills

### 1. muller-dataset
**Purpose:** Basic dataset operations (CRUD)

**Capabilities:**
- Create and load datasets
- Manage tensors (create, delete, rename)
- Add, update, delete data
- Query and filter samples
- Import data from files
- Inspect dataset information

**Example Commands:**
- "Create an image dataset at ./photos"
- "Add all images from ./data/ folder"
- "Query samples where label > 5"

### 2. muller-version-control
**Purpose:** Git-like version control

**Capabilities:**
- Commit changes
- Create and switch branches
- Merge branches with conflict resolution
- View commit history and logs
- Compare differences between commits/branches

**Example Commands:**
- "Commit changes with message 'Added samples'"
- "Create a new branch called dev-1"
- "Merge dev-1 into main"
- "Show commit history"

### 3. muller-advanced-query
**Purpose:** Advanced querying and indexing

**Capabilities:**
- Create inverted indexes for text search
- Create vector indexes (FLAT, HNSW, DISKANN)
- Perform vector similarity search
- Run aggregation queries (GROUP BY, COUNT, AVG, etc.)
- Complex filtering with multiple conditions

**Example Commands:**
- "Create a vector index for embeddings"
- "Find top 10 similar vectors"
- "Create inverted index for text search"
- "Aggregate by category and count"

### 4. muller-export
**Purpose:** Export and integration with other formats

**Capabilities:**
- Export to Apache Arrow format
- Export to Parquet files
- Export to JSON format
- Convert tensors to NumPy arrays
- Export to MindRecord format (MindSpore)
- Integrate with PyTorch, TensorFlow, and other frameworks

**Example Commands:**
- "Export dataset to Parquet format"
- "Convert embeddings tensor to NumPy"
- "Export to JSON for web API"
- "Export to Arrow for data sharing"

## Usage

When using AI coding assistants (Claude Code, Cursor, etc.), simply describe what you want in natural language. The agent will automatically:

1. Identify the appropriate skill
2. Load the skill instructions
3. Execute the corresponding script
4. Report results back to you

## Directory Structure

```
.claude/skills/
├── README.md                           # This file
├── muller-dataset/
│   ├── SKILL.md                        # Skill definition
│   ├── scripts/
│   │   ├── dataset_manager.py          # Dataset lifecycle
│   │   └── data_operations.py          # CRUD operations
│   └── references/
│       ├── quick-start.md
│       ├── api-cheatsheet.md
│       └── htypes-guide.md
├── muller-version-control/
│   ├── SKILL.md
│   ├── scripts/
│   │   └── version_control.py          # Version control ops
│   └── references/
├── muller-advanced-query/
│   ├── SKILL.md
│   ├── scripts/
│   │   └── advanced_query.py           # Advanced queries
│   └── references/
└── muller-export/
    ├── SKILL.md
    ├── scripts/
    │   └── export.py                   # Export operations
    └── references/
```

## Design Principles

1. **No new files:** Skills use existing scripts, never create new files in project root
2. **Progressive disclosure:** Keep SKILL.md concise, detailed docs in references/
3. **JSON output:** All scripts output structured JSON for agent parsing
4. **Clear errors:** Error messages include suggestions for resolution
5. **Self-documenting:** Scripts provide comprehensive --help output

## Token Budget

Each skill is designed to stay within reasonable token limits:
- muller-dataset: ~6000 tokens
- muller-version-control: ~4000 tokens  
- muller-advanced-query: ~5000 tokens
- muller-export: ~4000 tokens

## Complete Workflow Example

```
User: "Create an image dataset, add photos, commit changes, and export to Parquet"

Agent uses:
1. muller-dataset: Create dataset with image tensors
2. muller-dataset: Import images from directory
3. muller-version-control: Commit changes
4. muller-export: Export to Parquet format
```

## Future Enhancements

Potential additional skills:
- Data validation and quality checks
- Performance optimization and profiling
- Multi-modal data processing pipelines
- Automated data augmentation
