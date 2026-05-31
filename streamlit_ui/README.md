# MULLER Streamlit Demo

Interactive demonstration of MULLER's multimodal data lake capabilities for SIGMOD 2026 Demo Track.

## Overview

This Streamlit application showcases MULLER's key features:

- **Dataset Management**: Create datasets, add/edit/delete samples with multimodal data
- **Query & Search**: Conditional filtering, full-text search, and vector similarity search
- **Version Control**: Git-like branching, merging, and conflict resolution
- **Performance Benchmarks**: Compare MULLER vs Parquet for query latency and storage efficiency

## Installation

### Prerequisites

- Python 3.11+
- MULLER package installed

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MULLER
```

2. Install dependencies:
```bash
pip install -e .[demo]
```

This will install:
- Core MULLER dependencies (numpy, pandas, pillow, lz4, etc.)
- Demo-specific dependencies (streamlit, plotly)

## Running the Demo

> **Tip:** If the page shows "localhost refused to connect", first kill any leftover processes with `pkill -f streamlit`, then restart.

### Option 1: Using the launcher script (recommended)

```bash
cd streamlit_ui
./run_demo.sh
```

### Option 2: Direct command

```bash
# 1. Kill any leftover Streamlit processes
pkill -f streamlit

# 2. Confirm port 8501 is free
lsof -i :8501

# 3. Launch from the streamlit_ui directory
cd streamlit_ui
python3 -m streamlit run demo_streamlit.py --server.headless true

# 4. Wait until you see "Local URL: http://localhost:8501", then open the URL in your browser
```

The application will open in your default browser at `http://localhost:8501`.

## Usage Guide

### 1. Dataset Management

**Create a New Dataset:**
1. Navigate to "📊 Dataset Management" → "Create Dataset" tab
2. Enter dataset name and root directory
3. Click "Create Dataset"
4. Default schema includes: `labels` (int), `categories` (text), `description` (text)

**Add Samples:**
- **Manual Entry**: Use "Add Single Sample" form
- **Batch Upload**: Upload CSV file with matching column names

**View & Edit:**
- View dataset summary (total samples, tensors, current branch)
- Browse data in table format
- Delete samples by index

### 2. Query & Search

**Conditional Filtering:**
1. Select field, operator, and value
2. Click "Run Query"
3. View filtered results in table

Supported operators:
- Comparison: `==`, `!=`, `>`, `<`, `>=`, `<=`
- Text: `CONTAINS`, `LIKE`

**Vector Search:**
- Requires embeddings tensor with vector index
- Feature available when dataset includes vector data

### 3. Version Control

**Branch Management:**
- **Create Branch**: Enter branch name and click "Create Branch"
- **Switch Branch**: Select branch from dropdown and click "Checkout"
- **View Branches**: See all branches with current branch highlighted

**Merge Workflow:**
1. Select source branch to merge from
2. Click "Detect Conflicts" to preview conflicts
3. Choose merge strategy:
   - **Append Resolution**: `ours` | `theirs` | `both`
   - **Delete Resolution**: `ours` | `theirs`
   - **Update Resolution**: `ours` | `theirs`
4. Click "Merge" to complete

**Commit Log:**
- View full commit history with timestamps and messages

### 4. Performance Benchmarks

**Run Benchmark:**
1. Configure query (field, operator, value)
2. Click "Run Benchmark"
3. View comparison chart:
   - Query time (seconds)
   - Storage size (MB)

The benchmark exports dataset to Parquet and compares:
- Query execution time
- File size on disk

## Demo Workflow Example

### Scenario: Collaborative Data Annotation

1. **Create Dataset** (`main` branch)
   - Add initial samples with labels and descriptions

2. **Create Branch** (`dev-alice`)
   - Alice adds new samples
   - Alice updates existing labels
   - Commit changes

3. **Create Branch** (`dev-bob`)
   - Bob adds different samples
   - Bob updates same labels as Alice (conflict!)
   - Commit changes

4. **Merge Branches**
   - Merge `dev-alice` → `main` (fast-forward)
   - Merge `dev-bob` → `main` (three-way merge)
   - Detect conflicts on label updates
   - Resolve with merge strategy
   - View final merged dataset

5. **Query & Benchmark**
   - Filter samples by category
   - Compare performance vs Parquet

## Project Structure

```
MULLER/
│
├─ streamlit_ui/
│   ├─ demo_streamlit.py      # Main Streamlit application
│   └─ utils.py               # MULLER API wrapper functions
├─ pyproject.toml             # Dependencies (demo group)
└─ README.md                  # This file
```

## Key Features Demonstrated

### Git-like Versioning
- Create branches for parallel development
- Merge with automatic conflict detection
- Three merge strategies: append, delete, update
- Commit history tracking

### Multimodal Data Support
- Text (descriptions, categories)
- Numeric (labels, IDs)
- Images (via file upload)
- Extensible to video, audio, embeddings

### Efficient Storage
- Chunk-based storage with lazy loading
- LRU cache for fast access
- Multiple compression formats
- Smaller storage footprint than Parquet

### Query Capabilities
- SQL-like conditional filtering
- Full-text search with inverted index
- Vector similarity search (HNSW, FAISS)
- Pagination with offset/limit

## Troubleshooting

### Dataset Not Loading
- Ensure dataset path is correct
- Check file permissions
- Verify MULLER installation: `python -c "import muller; print(muller.__version__)"`

### Streamlit Errors
- Clear cache: Click "Clear cache" in Streamlit menu (top-right)
- Restart app: `Ctrl+C` and re-run `./run_demo.sh` or `streamlit run demo_streamlit.py`

### Performance Issues
- Large datasets may take time to load
- Use pagination (offset/limit) for queries
- Enable lazy loading in MULLER config

## Technical Details

### Dependencies

**Core (from pyproject.toml):**
- numpy, pandas, pillow, lz4
- pyarrow (for Parquet export)
- faiss-cpu (for vector search)

**Demo-specific:**
- streamlit >= 1.30
- plotly (for charts)

### Session State Management

Streamlit session state stores:
- `dataset`: Current MULLER dataset object
- `dataset_path`: Path to dataset on disk
- `current_branch`: Active branch name

On page refresh, dataset is reloaded from `dataset_path`.

### API Wrapper Functions

`streamlit_ui/utils.py` provides:
- `create_dataset()`: Initialize new dataset
- `add_samples()`: Append data with auto-commit
- `run_query()`: Execute filter queries
- `branch_ops()`: Version control operations
- `benchmark_parquet_vs_muller()`: Performance comparison

## Citation

If you use MULLER in your research, please cite:

```bibtex
@inproceedings{muller2026,
  title={MULLER: A Multimodal Data Lake Format for Collaborative AI Data Workflows},
  author={Lin, Xueling and Liu, Bingyu},
  booktitle={SIGMOD},
  year={2026}
}
```

## License

Mozilla Public License 2.0 (MPL-2.0)

## Contact

- Xueling Lin: heathersherrylin@gmail.com
- Bingyu Liu: liubingyu62@gmail.com

---

**SIGMOD 2026 Demo Track Submission**
