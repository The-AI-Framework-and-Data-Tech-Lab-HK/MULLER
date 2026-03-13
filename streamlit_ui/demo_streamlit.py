# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

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
import os
import tempfile
from pathlib import Path

# Ensure project root is on sys.path so `import muller` works
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import (
    create_dataset, create_tensors, add_samples, update_sample, delete_sample,
    run_query, dataset_to_dataframe, branch_ops, benchmark_parquet_vs_muller,
    load_dataset, commit_dataset, get_dataset_info,
    build_commit_graph_data, render_commit_graph_html,
)
import streamlit.components.v1 as components
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

    tab_create, tab_view = st.tabs(["Create / Load", "View & Edit"])

    # --- Tab 1: Create / Load ---
    with tab_create:
        col_left, col_right = st.columns(2)

        # ---- Create New Dataset (left column) ----
        with col_left:
            st.subheader("Create New Dataset")
            ds_name = st.text_input("Dataset Name", value="demo_dataset", key="create_name")
            ds_root = st.text_input("Root Directory", value=str(Path.home() / "muller_datasets"), key="create_root")
            overwrite = st.checkbox("Overwrite if exists", value=False)

            # --- Dynamic schema definition ---
            HTYPE_OPTIONS = ["generic", "text", "image", "video", "audio", "embedding", "class_label", "json"]
            DTYPE_OPTIONS = ["(auto)", "int32", "int64", "float32", "float64", "uint8", "bool", "str"]
            COMPRESSION_OPTIONS = ["(none)", "lz4", "jpg", "png", "mp4", "mp3", "wav"]

            st.markdown("**Define Columns (Tensors)**")

            # Each row is tracked by a unique ID so insert/delete anywhere is safe.
            _DEFAULT_ROWS = [
                {"id": 0, "name": "labels",      "htype": "generic", "dtype": "int64",  "comp": "(none)"},
                {"id": 1, "name": "categories",  "htype": "text",    "dtype": "(auto)",  "comp": "(none)"},
                {"id": 2, "name": "description", "htype": "text",    "dtype": "(auto)",  "comp": "(none)"},
            ]
            if "schema_rows" not in st.session_state:
                st.session_state.schema_rows = list(_DEFAULT_ROWS)
            if "schema_next_id" not in st.session_state:
                st.session_state.schema_next_id = len(_DEFAULT_ROWS)

            # Header row
            h_name, h_htype, h_dtype, h_comp, h_btns = st.columns([3, 2, 2, 2, 1])
            h_name.markdown("**Name**")
            h_htype.markdown("**htype**")
            h_dtype.markdown("**dtype**")
            h_comp.markdown("**compress**")
            h_btns.markdown("**+/−**")

            schema_inputs = []
            rows = st.session_state.schema_rows
            for pos, row in enumerate(rows):
                rid = row["id"]
                c_name, c_htype, c_dtype, c_comp, c_btns = st.columns([3, 2, 2, 2, 1])
                with c_name:
                    col_name = st.text_input("n", value=row["name"], key=f"col_name_{rid}",
                                             label_visibility="collapsed")
                with c_htype:
                    col_htype = st.selectbox("h", HTYPE_OPTIONS,
                                             index=HTYPE_OPTIONS.index(row["htype"]) if row["htype"] in HTYPE_OPTIONS else 0,
                                             key=f"col_htype_{rid}", label_visibility="collapsed")
                with c_dtype:
                    col_dtype = st.selectbox("d", DTYPE_OPTIONS,
                                             index=DTYPE_OPTIONS.index(row["dtype"]) if row["dtype"] in DTYPE_OPTIONS else 0,
                                             key=f"col_dtype_{rid}", label_visibility="collapsed")
                with c_comp:
                    col_comp = st.selectbox("c", COMPRESSION_OPTIONS,
                                            index=COMPRESSION_OPTIONS.index(row["comp"]) if row["comp"] in COMPRESSION_OPTIONS else 0,
                                            key=f"col_comp_{rid}", label_visibility="collapsed")
                with c_btns:
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("＋", key=f"add_{rid}", help="Insert a row below"):
                            new_id = st.session_state.schema_next_id
                            st.session_state.schema_next_id += 1
                            rows.insert(pos + 1, {"id": new_id, "name": "", "htype": "generic",
                                                   "dtype": "(auto)", "comp": "(none)"})
                            st.rerun()
                    with b2:
                        if len(rows) > 1 and st.button("−", key=f"del_{rid}", help="Remove this row"):
                            rows.pop(pos)
                            st.rerun()

                schema_inputs.append((col_name.strip(), col_htype, col_dtype, col_comp))

            if st.button("Create Dataset", type="primary"):
                # Validate column names
                col_names = [s[0] for s in schema_inputs if s[0]]
                if not col_names:
                    st.error("Please define at least one column.")
                elif len(col_names) != len(set(col_names)):
                    st.error("Column names must be unique.")
                else:
                    with st.spinner("Creating dataset..."):
                        ds, error = create_dataset(ds_name, ds_root, overwrite=overwrite)
                        if error:
                            st.error(error)
                        else:
                            schema = {}
                            for name, htype, dtype, comp in schema_inputs:
                                if not name:
                                    continue
                                cfg = {"htype": htype}
                                if dtype != "(auto)":
                                    cfg["dtype"] = dtype
                                if comp != "(none)":
                                    cfg["sample_compression"] = comp
                                schema[name] = cfg

                            err = create_tensors(ds, schema)
                            if err:
                                st.error(err)
                            else:
                                ds.commit(message="Initial schema creation")
                                st.session_state.dataset = ds
                                st.session_state.dataset_path = str(Path(ds_root) / ds_name)
                                st.session_state.current_branch = "main"
                                st.success(f"Dataset created with columns: {list(schema.keys())}")
                                st.rerun()

        # ---- Load Existing Dataset (right column) ----
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

    # --- Tab 2: View & Edit ---
    with tab_view:
        st.subheader("View & Edit Dataset")
        if st.session_state.dataset is None:
            st.warning("Please create or load a dataset first.")
        else:
            ds = st.session_state.dataset
            tensor_names = list(ds.tensors.keys())
            n = len(ds)

            # 1) Current Schema (collapsed by default)
            with st.expander("Current Schema", expanded=False):
                schema_rows = []
                for tname, t in ds.tensors.items():
                    schema_rows.append({"Column": tname, "htype": t.htype, "dtype": str(t.dtype)})
                st.dataframe(pd.DataFrame(schema_rows), width="stretch", hide_index=True)

            # 2) Dataset details
            col1, col2, col3 = st.columns(3)
            col1.metric("Samples", n)
            col2.metric("Tensors", len(ds.tensors))
            col3.metric("Branch", ds.branch)

            if n == 0:
                st.info("Dataset is empty. Add samples below.")
            else:
                preview_limit = min(n, 200)
                df, err = dataset_to_dataframe(ds, end=preview_limit)
                if err:
                    st.error(err)
                else:
                    st.dataframe(df, width="stretch")

            # 3) Add Single Sample (collapsed by default)
            with st.expander("Add Single Sample", expanded=False):
                sample_data = {}
                for tname in tensor_names:
                    t = ds.tensors[tname]
                    htype = t.htype
                    dtype_str = str(t.dtype)

                    if htype == "text":
                        sample_data[tname] = [st.text_input(f"{tname} (text)", key=f"add_{tname}")]
                    elif htype in ("image", "video", "audio"):
                        file_types = {"image": ["jpg", "png", "jpeg", "bmp"],
                                      "video": ["mp4", "avi", "mov"],
                                      "audio": ["mp3", "wav", "flac"]}
                        uploaded_file = st.file_uploader(
                            f"{tname} ({htype})", type=file_types.get(htype, []),
                            key=f"add_{tname}")
                        if uploaded_file is not None:
                            sample_data[tname] = [np.frombuffer(uploaded_file.read(), dtype=np.uint8)]
                        else:
                            sample_data[tname] = None
                    elif "int" in dtype_str:
                        val = st.text_input(f"{tname} (integer)", value="0", key=f"add_{tname}")
                        try:
                            sample_data[tname] = [int(val)]
                        except ValueError:
                            sample_data[tname] = [0]
                    elif "float" in dtype_str:
                        val = st.text_input(f"{tname} (float)", value="0.0", key=f"add_{tname}")
                        try:
                            sample_data[tname] = [float(val)]
                        except ValueError:
                            sample_data[tname] = [0.0]
                    elif "bool" in dtype_str:
                        val = st.checkbox(f"{tname} (bool)", key=f"add_{tname}")
                        sample_data[tname] = [val]
                    else:
                        val = st.text_input(f"{tname}", key=f"add_{tname}")
                        try:
                            sample_data[tname] = [int(val)]
                        except ValueError:
                            try:
                                sample_data[tname] = [float(val)]
                            except ValueError:
                                sample_data[tname] = [val]

                if st.button("Add Sample", type="primary"):
                    filtered = {k: v for k, v in sample_data.items() if v is not None}
                    if not filtered:
                        st.error("Please fill in at least one field.")
                    else:
                        err = add_samples(ds, filtered, auto_commit=True)
                        if err:
                            st.error(err)
                        else:
                            st.success("Sample added and committed.")
                            st.rerun()

            # 4) Batch Upload via CSV (collapsed by default)
            with st.expander("Batch Upload via CSV", expanded=False):
                uploaded = st.file_uploader("Choose CSV file", type=["csv"])
                if uploaded is not None:
                    df_up = pd.read_csv(uploaded)
                    st.dataframe(df_up.head(), width="stretch")

                    matched = [col for col in df_up.columns if col in tensor_names]
                    unmatched = [col for col in df_up.columns if col not in tensor_names]
                    if matched:
                        st.info(f"Matched columns: {matched}")
                    if unmatched:
                        st.warning(f"Columns not in dataset (will be skipped): {unmatched}")

                    media_cols = [
                        col for col in matched
                        if ds.tensors[col].htype in ("image", "video", "audio")
                    ]
                    path_columns = {}
                    if media_cols:
                        st.markdown("**Path columns** — these columns have media htypes. "
                                    "If their CSV values are file paths, select how to handle them:")
                        for col in media_cols:
                            mode = st.selectbox(
                                f"`{col}` ({ds.tensors[col].htype})",
                                options=["read", "text", "skip"],
                                help="read: load file via muller.read(); "
                                     "text: store path as text; "
                                     "skip: treat as plain value",
                                key=f"csv_pathcol_{col}",
                            )
                            if mode != "skip":
                                path_columns[col] = mode

                    if st.button("Import CSV Data"):
                        if not matched:
                            st.error(f"CSV columns must match tensors: {tensor_names}")
                        else:
                            tmp_path = None
                            try:
                                with tempfile.NamedTemporaryFile(
                                    suffix=".csv", delete=False, mode="wb"
                                ) as tmp:
                                    tmp.write(uploaded.getvalue())
                                    tmp_path = tmp.name

                                ds.add_data_from_csv(
                                    csv_path=tmp_path,
                                    path_columns=path_columns if path_columns else None,
                                    workers=0,
                                )
                                commit_dataset(ds, message="Import CSV data via Streamlit UI")
                                st.success(f"Imported {len(df_up)} samples.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to import CSV: {e}")
                            finally:
                                if tmp_path and os.path.exists(tmp_path):
                                    os.unlink(tmp_path)

            # 5) Delete Sample (collapsed by default)
            with st.expander("Delete Sample", expanded=False):
                if n == 0:
                    st.info("No samples to delete.")
                else:
                    del_idx = st.number_input("Sample Index", min_value=0, max_value=max(n - 1, 0), value=0)
                    if st.button("Delete", type="secondary"):
                        err = delete_sample(ds, del_idx)
                        if err:
                            st.error(err)
                        else:
                            commit_dataset(ds, message=f"Deleted sample {del_idx}")
                            st.success(f"Deleted sample {del_idx}")
                            st.rerun()

            # 6) Update Sample (collapsed by default)
            with st.expander("Update Sample", expanded=False):
                if n == 0:
                    st.info("No samples to update.")
                else:
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

            # Dynamic condition list tracked by unique IDs
            if "filter_cond_ids" not in st.session_state:
                st.session_state.filter_cond_ids = [0]
                st.session_state.filter_next_id = 1

            conditions = []
            connectors = []
            cond_ids = st.session_state.filter_cond_ids
            for pos, cid in enumerate(cond_ids):
                if pos == 0:
                    c_field, c_op, c_val, c_btn = st.columns([3, 2, 3, 1])
                else:
                    c_conn, c_field, c_op, c_val, c_btn = st.columns([1, 3, 2, 3, 1])
                    with c_conn:
                        conn = st.selectbox("Logic", ["AND", "OR"], key=f"conn_{cid}",
                                            label_visibility="collapsed")
                        connectors.append(conn)

                with c_field:
                    field = st.selectbox("Field" if pos == 0 else "Field",
                                         tensor_names, key=f"field_{cid}",
                                         label_visibility="collapsed" if pos > 0 else "visible")
                with c_op:
                    op = st.selectbox("Op" if pos == 0 else "Op",
                                      ["==", "!=", ">", "<", ">=", "<=", "CONTAINS", "LIKE"],
                                      key=f"op_{cid}",
                                      label_visibility="collapsed" if pos > 0 else "visible")
                with c_val:
                    val = st.text_input("Value" if pos == 0 else "Value",
                                        key=f"val_{cid}",
                                        label_visibility="collapsed" if pos > 0 else "visible")
                with c_btn:
                    if pos == 0:
                        # Spacer to align with label row, then ＋ button
                        st.markdown("<div style='height:29px'></div>", unsafe_allow_html=True)
                        if st.button("＋", key="add_cond", help="Add condition"):
                            new_id = st.session_state.filter_next_id
                            st.session_state.filter_next_id += 1
                            cond_ids.append(new_id)
                            st.rerun()
                    else:
                        if st.button("−", key=f"del_cond_{cid}", help="Remove this condition"):
                            cond_ids.remove(cid)
                            st.rerun()

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

                # Auto-create inverted index for CONTAINS queries if missing
                for field, op, val in parsed_conds:
                    if op == "CONTAINS":
                        branch = ds.version_state["branch"]
                        meta_path = f"inverted_index_dir_vec/{branch}/meta.json"
                        has_index = False
                        try:
                            import json as _json
                            meta = _json.loads(ds.storage[meta_path].decode("utf-8"))
                            has_index = field in meta
                        except KeyError:
                            pass
                        if not has_index:
                            with st.spinner(f"Creating inverted index for '{field}'..."):
                                if ds.has_head_changes:
                                    ds.commit(message="Auto-commit before index creation")
                                ds.create_index_vectorized(field)

                result_ds, err = run_query(ds, parsed_conds, connectors if connectors else None)
                if err:
                    st.error(err)
                else:
                    n_results = len(result_ds)
                    st.success(f"Found {n_results} matching samples")
                    if n_results > 0:
                        df, _ = dataset_to_dataframe(result_ds, end=min(n_results, 200))
                        if df is not None:
                            st.dataframe(df, width="stretch")

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

        tab_branch, tab_merge = st.tabs(["Branches", "Merge & Conflicts"])

        # --- Branches ---
        with tab_branch:
            st.subheader("Branch Management")

            branches, err = branch_ops(ds, "list")
            if err:
                st.error(err)
                st.stop()

            branch_names = list(branches.keys()) if isinstance(branches, dict) else branches

            if st.button("Refresh", key="refresh_graph"):
                st.rerun()

            # Visual commit graph
            try:
                _graph_data = build_commit_graph_data(ds)
                _graph_html = render_commit_graph_html(_graph_data, height=350)
                components.html(_graph_html, height=370, scrolling=True)
            except Exception:
                # Fallback to markdown
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
                                            st.dataframe(pd.DataFrame(rows), width="stretch")

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

                                                st.dataframe(cdf.style.apply(_hl, axis=1), width="stretch")

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
                    st.plotly_chart(fig, width="stretch")
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
