"""
MULLER Streamlit Demo - Interactive Multimodal Data Lake Management

This demo showcases MULLER's capabilities:
- Dataset creation and CRUD operations
- Conditional filtering and vector search
- Git-like version control (branch, merge, conflict resolution)
- Performance benchmarking vs Parquet
"""
import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import muller` works
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import (
    create_dataset, create_tensors, add_samples, update_sample, delete_sample,
    run_query, dataset_to_dataframe, branch_ops, benchmark_parquet_vs_muller,
    load_dataset, commit_dataset, get_dataset_info,
)
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MULLER Demo",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_defaults = {
    "dataset": None,
    "dataset_path": None,
    "current_branch": "main",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helper: reload dataset from path (survives Streamlit reruns)
# ---------------------------------------------------------------------------
def _ensure_dataset():
    """Reload the dataset object from path if it was lost across reruns."""
    if st.session_state.dataset is None and st.session_state.dataset_path:
        ds, err = load_dataset(st.session_state.dataset_path)
        if err is None:
            st.session_state.dataset = ds


_ensure_dataset()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🗄️ MULLER Demo")
st.sidebar.markdown("**Multimodal Data Lake with Git-like Versioning**")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dataset Management", "🔍 Query & Search",
     "🌿 Version Control", "⚡ Benchmarks", "ℹ️ About"],
)

st.sidebar.markdown("---")
if st.session_state.dataset is not None:
    info = get_dataset_info(st.session_state.dataset)
    st.sidebar.success(f"Dataset loaded ({info.get('num_samples', '?')} samples)")
    st.sidebar.info(f"Branch: `{info.get('branch', '?')}`")
    if info.get("has_uncommitted"):
        st.sidebar.warning("Uncommitted changes")
else:
    st.sidebar.warning("No dataset loaded")

# ============================================================================
# PAGE 1: Dataset Management
# ============================================================================
if page == "📊 Dataset Management":
    st.title("📊 Dataset Management")

    tab_create, tab_add, tab_view = st.tabs(["Create / Load", "Add Samples", "View & Edit"])

    # --- Tab 1: Create / Load ---
    with tab_create:
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Create New Dataset")
            ds_name = st.text_input("Dataset Name", value="demo_dataset", key="create_name")
            ds_root = st.text_input("Root Directory", value=str(Path.home() / "muller_datasets"), key="create_root")
            overwrite = st.checkbox("Overwrite if exists", value=False)

            if st.button("Create Dataset", type="primary"):
                with st.spinner("Creating dataset..."):
                    ds, error = create_dataset(ds_name, ds_root, overwrite=overwrite)
                    if error:
                        st.error(error)
                    else:
                        schema = {
                            "labels":      {"htype": "generic", "dtype": "int64"},
                            "categories":  {"htype": "text"},
                            "description": {"htype": "text"},
                        }
                        err = create_tensors(ds, schema)
                        if err:
                            st.error(err)
                        else:
                            ds.commit(message="Initial schema creation")
                            st.session_state.dataset = ds
                            st.session_state.dataset_path = str(Path(ds_root) / ds_name)
                            st.session_state.current_branch = "main"
                            st.success(f"Dataset created at: {st.session_state.dataset_path}")
                            st.rerun()

        with col_right:
            st.subheader("Load Existing Dataset")
            load_path = st.text_input("Dataset Path", value="", key="load_path")
            if st.button("Load Dataset"):
                if not load_path:
                    st.error("Please enter a path")
                else:
                    ds, error = load_dataset(load_path)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.dataset = ds
                        st.session_state.dataset_path = load_path
                        st.session_state.current_branch = ds.branch
                        st.success(f"Dataset loaded from: {load_path}")
                        st.rerun()

    # --- Tab 2: Add Samples ---
    with tab_add:
        st.subheader("Add Samples")
        if st.session_state.dataset is None:
            st.warning("Please create or load a dataset first.")
        else:
            ds = st.session_state.dataset
            tensor_names = list(ds.tensors.keys())

            with st.expander("Add Single Sample", expanded=True):
                sample_data = {}
                for tname in tensor_names:
                    t = ds.tensors[tname]
                    if t.htype == "text":
                        sample_data[tname] = [st.text_input(f"{tname}", key=f"add_{tname}")]
                    else:
                        val = st.text_input(f"{tname} (number)", value="0", key=f"add_{tname}")
                        try:
                            sample_data[tname] = [int(val)]
                        except ValueError:
                            try:
                                sample_data[tname] = [float(val)]
                            except ValueError:
                                sample_data[tname] = [val]

                if st.button("Add Sample", type="primary"):
                    err = add_samples(ds, sample_data, auto_commit=True)
                    if err:
                        st.error(err)
                    else:
                        st.success("Sample added and committed.")
                        st.rerun()

            with st.expander("Batch Upload (CSV)"):
                uploaded = st.file_uploader("Choose CSV file", type=["csv"])
                if uploaded is not None:
                    df_up = pd.read_csv(uploaded)
                    st.dataframe(df_up.head(), use_container_width=True)
                    if st.button("Import CSV Data"):
                        data = {col: df_up[col].tolist() for col in df_up.columns if col in tensor_names}
                        if not data:
                            st.error(f"CSV columns must match tensors: {tensor_names}")
                        else:
                            err = add_samples(ds, data, auto_commit=True)
                            if err:
                                st.error(err)
                            else:
                                st.success(f"Imported {len(df_up)} samples.")
                                st.rerun()

    # --- Tab 3: View & Edit ---
    with tab_view:
        st.subheader("View & Edit Dataset")
        if st.session_state.dataset is None:
            st.warning("Please create or load a dataset first.")
        else:
            ds = st.session_state.dataset
            n = len(ds)

            col1, col2, col3 = st.columns(3)
            col1.metric("Samples", n)
            col2.metric("Tensors", len(ds.tensors))
            col3.metric("Branch", ds.branch)

            if n == 0:
                st.info("Dataset is empty. Add samples first.")
            else:
                preview_limit = min(n, 200)
                df, err = dataset_to_dataframe(ds, end=preview_limit)
                if err:
                    st.error(err)
                else:
                    st.dataframe(df, use_container_width=True)

                with st.expander("Delete Sample"):
                    del_idx = st.number_input("Sample Index", min_value=0, max_value=max(n - 1, 0), value=0)
                    if st.button("Delete", type="secondary"):
                        err = delete_sample(ds, del_idx)
                        if err:
                            st.error(err)
                        else:
                            commit_dataset(ds, message=f"Deleted sample {del_idx}")
                            st.success(f"Deleted sample {del_idx}")
                            st.rerun()

                with st.expander("Update Sample"):
                    upd_idx = st.number_input("Sample Index to Update", min_value=0, max_value=max(n - 1, 0), value=0, key="upd_idx")
                    upd_tensor = st.selectbox("Tensor", list(ds.tensors.keys()), key="upd_tensor")
                    upd_val = st.text_input("New Value", key="upd_val")
                    if st.button("Update", type="secondary"):
                        try:
                            parsed = int(upd_val)
                        except ValueError:
                            try:
                                parsed = float(upd_val)
                            except ValueError:
                                parsed = upd_val
                        err = update_sample(ds, upd_tensor, upd_idx, parsed)
                        if err:
                            st.error(err)
                        else:
                            commit_dataset(ds, message=f"Updated {upd_tensor}[{upd_idx}]")
                            st.success(f"Updated {upd_tensor}[{upd_idx}]")
                            st.rerun()


# ============================================================================
# PAGE 2: Query & Search
# ============================================================================
elif page == "🔍 Query & Search":
    st.title("🔍 Query & Search")

    if st.session_state.dataset is None:
        st.warning("Please create or load a dataset first.")
    else:
        ds = st.session_state.dataset
        tensor_names = list(ds.tensors.keys())

        tab_filter, tab_vector = st.tabs(["Conditional Filtering", "Vector Search"])

        with tab_filter:
            st.subheader("Conditional Filtering")

            # Allow multiple conditions
            num_conds = st.number_input("Number of conditions", 1, 5, 1, key="num_conds")

            conditions = []
            connectors = []
            for i in range(num_conds):
                cols = st.columns([3, 2, 3, 2] if i > 0 else [3, 2, 3])
                idx = 0
                if i > 0:
                    with cols[0]:
                        conn = st.selectbox("Logic", ["AND", "OR"], key=f"conn_{i}")
                        connectors.append(conn)
                    idx = 1

                with cols[idx]:
                    field = st.selectbox("Field", tensor_names, key=f"field_{i}")
                with cols[idx + 1]:
                    op = st.selectbox("Op", ["==", "!=", ">", "<", ">=", "<=", "CONTAINS", "LIKE"], key=f"op_{i}")
                with cols[idx + 2]:
                    val = st.text_input("Value", key=f"val_{i}")

                conditions.append((field, op, val))

            if st.button("Run Query", type="primary"):
                # Parse values
                parsed_conds = []
                for field, op, val in conditions:
                    if op in (">", "<", ">=", "<="):
                        try:
                            val = float(val)
                        except ValueError:
                            st.error(f"Cannot convert '{val}' to number for operator {op}")
                            st.stop()
                    elif op == "==":
                        try:
                            val = int(val)
                        except ValueError:
                            pass
                    parsed_conds.append((field, op, val))

                result_ds, err = run_query(ds, parsed_conds, connectors if connectors else None)
                if err:
                    st.error(err)
                else:
                    n_results = len(result_ds)
                    st.success(f"Found {n_results} matching samples")
                    if n_results > 0:
                        df, _ = dataset_to_dataframe(result_ds, end=min(n_results, 200))
                        if df is not None:
                            st.dataframe(df, use_container_width=True)

        with tab_vector:
            st.subheader("Vector Similarity Search")
            st.info("Vector search requires an embeddings tensor with a vector index.")
            st.markdown("""
            **How to use:**
            1. Create a tensor with `htype='embedding'`
            2. Build a vector index with `ds.create_index()`
            3. Query with `ds.query(tensor_name, query_vector)`

            *This feature requires pre-computed embeddings.*
            """)


# ============================================================================
# PAGE 3: Version Control
# ============================================================================
elif page == "🌿 Version Control":
    st.title("🌿 Version Control")

    if st.session_state.dataset is None:
        st.warning("Please create or load a dataset first.")
    else:
        ds = st.session_state.dataset

        tab_branch, tab_merge, tab_log = st.tabs(["Branches", "Merge & Conflicts", "Commit Log"])

        # --- Branches ---
        with tab_branch:
            st.subheader("Branch Management")

            branches, err = branch_ops(ds, "list")
            if err:
                st.error(err)
                st.stop()

            branch_names = list(branches.keys()) if isinstance(branches, dict) else branches

            st.markdown("**Current branches:**")
            for bname in branch_names:
                if bname == ds.branch:
                    st.markdown(f"- **{bname}** ← current")
                else:
                    st.markdown(f"- {bname}")

            col_create, col_switch = st.columns(2)

            with col_create:
                st.markdown("**Create Branch**")
                new_branch = st.text_input("New branch name", key="new_branch")
                if st.button("Create Branch", type="primary"):
                    if not new_branch:
                        st.error("Enter a branch name")
                    else:
                        # Commit pending changes before branching
                        if ds.has_head_changes:
                            commit_dataset(ds, "Auto-commit before branch creation")
                        res, err = branch_ops(ds, "create", branch_name=new_branch)
                        if err:
                            st.error(err)
                        else:
                            st.session_state.current_branch = new_branch
                            st.success(res)
                            st.rerun()

            with col_switch:
                st.markdown("**Switch Branch**")
                other_branches = [b for b in branch_names]
                target = st.selectbox("Target branch", other_branches, key="switch_branch")
                if st.button("Checkout"):
                    if ds.has_head_changes:
                        commit_dataset(ds, "Auto-commit before checkout")
                    res, err = branch_ops(ds, "checkout", branch_name=target)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.current_branch = target
                        st.success(res)
                        st.rerun()

        # --- Merge & Conflicts ---
        with tab_merge:
            st.subheader("Merge & Conflict Resolution")

            branches, _ = branch_ops(ds, "list")
            branch_names = list(branches.keys()) if isinstance(branches, dict) else branches
            other = [b for b in branch_names if b != ds.branch]

            if not other:
                st.info("No other branches to merge.")
            else:
                merge_src = st.selectbox("Merge from branch", other, key="merge_src")

                # Detect conflicts
                if st.button("Detect Conflicts"):
                    result, err = branch_ops(ds, "detect_conflict", branch_name=merge_src)
                    if err:
                        st.error(err)
                    else:
                        if result["columns"]:
                            st.warning(f"Conflicts in: {', '.join(result['columns'])}")

                            with st.expander("Conflict Details", expanded=True):
                                for col_name in result["columns"]:
                                    st.markdown(f"#### `{col_name}`")
                                    cdata = result["records"].get(col_name, {})

                                    # Append conflicts
                                    if cdata.get("app_ori_idx"):
                                        st.markdown("**Append conflicts:**")
                                        rows = []
                                        ori_vals = cdata.get("app_ori_values", [])
                                        tar_vals = cdata.get("app_tar_values", [])
                                        for j, idx in enumerate(cdata["app_ori_idx"]):
                                            rows.append({"Index": idx, "Current (ours)": str(ori_vals[j]) if j < len(ori_vals) else "—"})
                                        for j, idx in enumerate(cdata.get("app_tar_idx", [])):
                                            rows.append({"Index": idx, "Source (theirs)": str(tar_vals[j]) if j < len(tar_vals) else "—"})
                                        if rows:
                                            st.dataframe(pd.DataFrame(rows), use_container_width=True)

                                    # Update conflicts
                                    if cdata.get("update_values"):
                                        update_ori = cdata["update_values"].get("update_ori", [])
                                        update_tar = cdata["update_values"].get("update_tar", [])
                                        if update_ori or update_tar:
                                            st.markdown("**Update conflicts:**")
                                            comp = []
                                            all_idx = set()
                                            for d in update_ori:
                                                all_idx.update(d.keys())
                                            for d in update_tar:
                                                all_idx.update(d.keys())
                                            for idx in sorted(all_idx):
                                                ov = next((d[idx] for d in update_ori if idx in d), "—")
                                                tv = next((d[idx] for d in update_tar if idx in d), "—")
                                                comp.append({"Index": idx, "Current (ours)": str(ov), "Source (theirs)": str(tv)})
                                            if comp:
                                                cdf = pd.DataFrame(comp)

                                                def _hl(row):
                                                    if row["Current (ours)"] != "—" and row["Source (theirs)"] != "—":
                                                        return ["background-color: #ffcccc"] * len(row)
                                                    return [""] * len(row)

                                                st.dataframe(cdf.style.apply(_hl, axis=1), use_container_width=True)

                                    # Delete conflicts
                                    if cdata.get("del_ori_idx") or cdata.get("del_tar_idx"):
                                        st.markdown("**Delete conflicts:**")
                                        if cdata.get("del_ori_idx"):
                                            st.markdown(f"- Current deletes: {cdata['del_ori_idx']}")
                                        if cdata.get("del_tar_idx"):
                                            st.markdown(f"- Source deletes: {cdata['del_tar_idx']}")
                        else:
                            st.success("No conflicts detected — safe to merge.")

                st.markdown("---")
                st.markdown("**Merge Strategy**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    append_res = st.radio("Append", ["ours", "theirs", "both"], key="m_app")
                with c2:
                    pop_res = st.radio("Delete", ["ours", "theirs"], key="m_pop")
                with c3:
                    update_res = st.radio("Update", ["ours", "theirs"], key="m_upd")

                if st.button("Merge", type="primary"):
                    strategy = {
                        "append_resolution": append_res,
                        "pop_resolution": pop_res,
                        "update_resolution": update_res,
                    }
                    res, err = branch_ops(ds, "merge", branch_name=merge_src, merge_strategy=strategy)
                    if err:
                        st.error(err)
                    else:
                        st.success(res)
                        st.rerun()

        # --- Commit Log ---
        with tab_log:
            st.subheader("Commit History")
            if st.button("Refresh Log"):
                st.rerun()

            commits, err = branch_ops(ds, "log")
            if err:
                st.error(err)
            else:
                if commits:
                    for c in commits:
                        if isinstance(c, dict):
                            st.markdown(f"- `{c.get('commit_id', '?')[:8]}` — {c.get('message', '(no message)')}")
                        else:
                            st.markdown(f"- `{str(c)[:8]}`")
                else:
                    st.info("No commits yet.")


# ============================================================================
# PAGE 4: Benchmarks
# ============================================================================
elif page == "⚡ Benchmarks":
    st.title("⚡ Performance Benchmarks")

    if st.session_state.dataset is None:
        st.warning("Please create or load a dataset first.")
    elif len(st.session_state.dataset) == 0:
        st.warning("Dataset is empty — add samples before running benchmarks.")
    else:
        ds = st.session_state.dataset
        st.subheader("MULLER vs Parquet: Query Performance & Storage")

        tensor_names = list(ds.tensors.keys())

        col1, col2, col3 = st.columns(3)
        with col1:
            bm_field = st.selectbox("Field", tensor_names, key="bm_field")
        with col2:
            bm_op = st.selectbox("Operator", [">", "<", "==", ">=", "<=", "!="], key="bm_op")
        with col3:
            bm_val = st.text_input("Value", value="0", key="bm_val")

        num_runs = st.slider("Number of runs (for averaging)", 1, 10, 3)

        if st.button("Run Benchmark", type="primary"):
            # Parse value
            try:
                parsed = int(bm_val)
            except ValueError:
                try:
                    parsed = float(bm_val)
                except ValueError:
                    parsed = bm_val

            with st.spinner(f"Running benchmark ({num_runs} runs)..."):
                conditions = [(bm_field, bm_op, parsed)]
                fig, err = benchmark_parquet_vs_muller(ds, conditions, num_runs=num_runs)
                if err:
                    st.error(err)
                else:
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("""
                    **Key Takeaways:**
                    - MULLER uses chunk-based storage with lazy loading for efficient I/O
                    - Parquet requires full table scan for non-indexed queries
                    - MULLER supports Git-like versioning without data duplication
                    """)


# ============================================================================
# PAGE 5: About
# ============================================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About MULLER")

    st.markdown("""
## MULLER: Multimodal Data Lake Format

**MULLER** is a next-generation data lake format designed for collaborative AI workflows.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multimodal** | Images, videos, audio, text, vectors, structured data |
| **Versioning** | Git-like branch, merge, conflict resolution |
| **Lazy Loading** | On-demand data loading with LRU cache |
| **Query Engine** | SQL-like filtering, full-text search, vector similarity |
| **Compression** | 20+ formats (LZ4, JPEG, PNG, MP4, …) |
| **Cloud Storage** | S3, OBS, Roma, and local filesystem |

### Architecture

```
Dataset
├── Tensor (column)
│   ├── ChunkEngine (variable-sized chunks)
│   └── TensorMeta (htype, dtype, compression)
├── VersionControl
│   ├── CommitDAG
│   └── MergeStrategies (ours / theirs / both)
├── LRUCache (memory → local → remote)
└── StorageProvider (local / S3 / OBS)
```

### Demo Workflow

1. **Create Dataset** → define schema, add samples
2. **Query & Search** → conditional filtering, vector similarity
3. **Version Control** → branch, modify, merge with conflict resolution
4. **Benchmark** → compare with Parquet on query latency & storage

---
*SIGMOD 2026 Demo Track Submission*
    """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("MULLER Demo | SIGMOD 2026")
