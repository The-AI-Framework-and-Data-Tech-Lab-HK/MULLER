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
import hashlib
import json
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
    run_query, dataset_to_dataframe, dataframe_for_streamlit_display,
    list_image_tensor_names, decode_muller_image_sample, pil_resize_to_height,
    pil_square_thumbnail, pil_fit_inside,
    pil_preview_available,
    is_coco2017_muller_schema, load_coco_category_id_to_name,
    pil_overlay_coco_bboxes, DEFAULT_COCO_INSTANCES_JSON,
    branch_ops, benchmark_parquet_vs_muller,
    load_dataset, release_dataset_lock, commit_dataset, get_dataset_info,
    build_commit_graph_data, render_commit_graph_html,
    classify_deletable_branches,
    list_vector_tensor_names, list_vector_indexes, parse_query_vector,
    run_vector_search, ensure_vector_index, tensor_embedding_dim,
    save_query_view, list_saved_views, load_saved_view, delete_saved_view,
    get_view_entry, current_user, set_current_user,
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


def _dm_on_select_coco_thumb(gi: int) -> None:
    st.session_state["dm_coco_selected_gi"] = int(gi)


def _dm_coco_thumb_page_delta(delta: int, max_page: int) -> None:
    cur = int(st.session_state.get("dm_coco_thumb_page", 1))
    st.session_state["dm_coco_thumb_page"] = max(
        1, min(int(max_page), cur + int(delta))
    )


# ---------------------------------------------------------------------------
# Helper: reload dataset from path (survives Streamlit reruns)
# ---------------------------------------------------------------------------
def _ensure_dataset():
    """Reload the dataset object from path if it was lost across reruns."""
    if st.session_state.dataset is None and st.session_state.dataset_path:
        ds, err, warn = load_dataset(st.session_state.dataset_path)
        if err is None:
            st.session_state.dataset = ds
            if warn:
                st.session_state["_load_warning"] = warn


_ensure_dataset()

# Full-resolution image lightbox (native PIL size; avoids blurry browser zoom on thumbnails).
_DM_PREVIEW_ROWS = 3


def _dm_fullres_dialog_body() -> None:
    img = st.session_state.get("dm_fullres_pil")
    cap = st.session_state.get("dm_fullres_caption", "")
    if img is None:
        return
    st.image(img, use_container_width=False)
    st.caption(cap)


_dm_fullres_dialog = (
    st.dialog("Full resolution", width="large")(_dm_fullres_dialog_body)
    if hasattr(st, "dialog")
    else None
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🗄️ MULLER Demo")
st.sidebar.markdown("**Multimodal Data Lake with Git-like Versioning**")

# -- Current user (for multi-user / permission demos) -----------------------
# SensitiveConfig is a process singleton, so setting it here affects every
# subsequent MULLER call in this Streamlit process — exactly the semantics we
# need to act as different engineers within a single demo session. The demo
# storyline only needs three personas, so we expose them as a fixed dropdown
# rather than a free-form textbox to keep the script copy-paste-free.
DEMO_USERS = ["public", "A", "B"]
_cur_user = current_user()
_default_idx = DEMO_USERS.index(_cur_user) if _cur_user in DEMO_USERS else 0
_user_input = st.sidebar.selectbox(
    "👤 Current user",
    options=DEMO_USERS,
    index=_default_idx,
    key="current_user_select",
    help="MULLER permission checks (branch ownership, delete_view, …) "
    "compare against this uid. Pick one of the three demo personas to "
    "impersonate that engineer and exercise the permission model.",
)
if _user_input and _user_input != _cur_user:
    new_uid = set_current_user(_user_input)
    st.sidebar.success(f"Acting as `{new_uid}`.")
    st.rerun()

st.sidebar.markdown("---")

# Persist the active page across reruns (e.g. after switching the current
# user, which calls st.rerun()). Without an explicit key the radio loses
# its selection on rerun and falls back to the first option, which is
# annoying mid-demo when the operator is on Version Control / Query.
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dataset Management", "🔍 Query & Search",
     "🌿 Version Control", "⚡ Benchmarks", "ℹ️ About"],
    key="nav_page",
)

st.sidebar.markdown("---")
if st.session_state.dataset is not None:
    info = get_dataset_info(st.session_state.dataset)
    st.sidebar.success(f"Dataset loaded ({info.get('num_samples', '?')} samples)")
    st.sidebar.info(f"Branch: `{info.get('branch', '?')}`")
    if info.get("has_uncommitted"):
        st.sidebar.warning("Uncommitted changes")
    # Surface read-only fallback prominently: it silently blocks commits,
    # inverted-index creation, and most write paths.
    if getattr(st.session_state.dataset, "read_only", False):
        st.sidebar.error(
            "Read-only mode — write lock unavailable. "
            "Close other holders and reload."
        )
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
            _default_create_path = str(Path.home() / "research_data" / "muller_coco_demo")
            ds_path_input = st.text_input(
                "Dataset Path",
                value=_default_create_path,
                key="create_dataset_path",
                help="Full path where the dataset folder will be created (parent directory must exist or be creatable).",
            )
            overwrite = st.checkbox("Overwrite if exists", value=False)

            # --- Dynamic schema definition ---
            # Pull htype names from MULLER's canonical configuration table so
            # the dropdown always matches what core accepts (e.g. `bbox`,
            # `polygon`, `keypoints_coco`, `list`, `segment_mask`, …). Any
            # hardcoded short-list here drifts out of sync as soon as someone
            # writes a valid schema JSON that uses one of the long-tail htypes,
            # with a confusing "htype X not in [...]" error.
            try:
                from muller.core.types.htype import HTYPE_CONFIGURATIONS as _HTYPE_CFG
                _CORE_HTYPES = list(_HTYPE_CFG.keys())
            except Exception:
                _CORE_HTYPES = []

            # Promote the htypes we touch most often in the demo to the top
            # of the dropdown (order matters for UX); append any remaining
            # core htypes alphabetically so nothing is hidden.
            _PREFERRED_HTYPES = [
                "generic", "text", "image", "video", "audio",
                "embedding", "class_label", "bbox", "polygon",
                "keypoints_coco", "segment_mask", "binary_mask",
                "instance_label", "list", "json", "vector", "point",
            ]
            _seen = set()
            HTYPE_OPTIONS = []
            for h in _PREFERRED_HTYPES:
                if h in _CORE_HTYPES or not _CORE_HTYPES:
                    HTYPE_OPTIONS.append(h)
                    _seen.add(h)
            for h in sorted(_CORE_HTYPES):
                if h not in _seen:
                    HTYPE_OPTIONS.append(h)

            DTYPE_OPTIONS = [
                "(auto)",
                "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64",
                "float16", "float32", "float64",
                "bool", "str",
            ]
            COMPRESSION_OPTIONS = ["(none)", "lz4", "jpg", "png", "mp4", "mp3", "wav"]

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

            # --- Bulk-load schema from JSON ---
            # Typing a multi-tensor schema cell-by-cell is tedious and error
            # prone, and skipping this step is the #1 reason the demo looks
            # "not like a COCO dataset" afterwards (you silently end up with
            # the 3-tensor placeholder schema, and a subsequent Batch Upload
            # of a 9-column CSV drops 8 of the 9 columns on the floor). So:
            #   - this expander is *open by default* (file uploader visible)
            #   - below, ``Create Dataset`` blocks on the default placeholder
            #     rows with an amber warning-and-confirm
            with st.expander("📥 Load schema from JSON file", expanded=True):
                st.caption(
                    "Upload a JSON describing the tensors. Accepted shapes:  \n"
                    "• `{\"tensors\": [{\"name\": …, \"htype\": …, \"dtype\": …, "
                    "\"sample_compression\": …}, …]}`  \n"
                    "• a plain list of the same objects  \n"
                    "• a `{name: {htype, dtype, sample_compression}}` dict  \n"
                    "See `sigmod_demo_revision/coco_schema.json` for a ready-to-use "
                    "9-tensor COCO2017 example (8 core tensors + `description` text)."
                )
                schema_file = st.file_uploader(
                    "Schema JSON", type=["json"], key="schema_json_upload",
                )

                def _normalize_schema_entries(data):
                    """Return (list-of-dicts, errors)."""
                    if isinstance(data, dict) and "tensors" in data:
                        raw = data.get("tensors") or []
                    elif isinstance(data, list):
                        raw = data
                    elif isinstance(data, dict):
                        raw = [{"name": k, **(v or {})} for k, v in data.items()]
                    else:
                        return [], ["Top-level JSON must be an object or an array."]
                    norm, errs = [], []
                    for i, e in enumerate(raw):
                        if not isinstance(e, dict):
                            errs.append(f"entry #{i} is not a JSON object")
                            continue
                        nm = (e.get("name") or "").strip()
                        if not nm:
                            errs.append(f"entry #{i} missing 'name'")
                            continue
                        ht = e.get("htype") or "generic"
                        dt = e.get("dtype") or "(auto)"
                        comp = e.get("sample_compression") or "(none)"
                        # Only htype is strictly validated here — we already
                        # pull every MULLER-recognized htype from
                        # HTYPE_CONFIGURATIONS above, so a miss here is a real
                        # typo. dtype / compression have many valid values
                        # MULLER accepts (e.g. "Any", "List", "jpeg" alias)
                        # that may not be in the shortlist dropdown, so we
                        # pass them through and let MULLER validate at
                        # tensor-creation time; the UI below dynamically
                        # extends its dropdowns to show whatever we got.
                        if ht not in HTYPE_OPTIONS:
                            errs.append(
                                f"entry '{nm}': htype '{ht}' is not a "
                                f"MULLER htype. Known htypes: {HTYPE_OPTIONS}"
                            )
                            continue
                        norm.append({"name": nm, "htype": ht, "dtype": dt, "comp": comp})
                    return norm, errs

                if schema_file is not None:
                    try:
                        _schema_data = json.load(schema_file)
                    except Exception as _e:
                        st.error(f"Could not parse JSON: {_e}")
                    else:
                        normalized, errs = _normalize_schema_entries(_schema_data)
                        if errs:
                            st.error("Schema JSON has problems:\n- " + "\n- ".join(errs))
                        elif not normalized:
                            st.error("Schema JSON has no valid tensor entries.")
                        else:
                            # Fingerprint the applied schema so streamlit's
                            # auto-reruns don't re-apply on every keystroke.
                            sig = json.dumps(normalized, sort_keys=True)
                            if st.session_state.get("_schema_json_sig") != sig:
                                next_id = int(st.session_state.get("schema_next_id", 0))
                                new_rows = []
                                for item in normalized:
                                    new_rows.append({"id": next_id, **item})
                                    next_id += 1
                                st.session_state.schema_rows = new_rows
                                st.session_state.schema_next_id = next_id
                                st.session_state["_schema_json_sig"] = sig
                                st.success(
                                    f"Loaded **{len(normalized)}** tensor rows "
                                    f"from `{schema_file.name}`."
                                )
                                st.rerun()
                            else:
                                st.caption(
                                    f"`{schema_file.name}` already applied "
                                    f"({len(normalized)} rows) — edit the table "
                                    "below to tweak, or upload a different file."
                                )

                col_reset, _ = st.columns([1, 3])
                with col_reset:
                    if st.button("Reset schema to UI defaults", key="schema_reset_btn"):
                        st.session_state.schema_rows = list(_DEFAULT_ROWS)
                        st.session_state.schema_next_id = len(_DEFAULT_ROWS)
                        st.session_state.pop("_schema_json_sig", None)
                        st.rerun()

            st.markdown("**Define Columns (Tensors)**")

            # Header row
            h_name, h_htype, h_dtype, h_comp, h_btns = st.columns([3, 2, 2, 2, 1])
            h_name.markdown("**Name**")
            h_htype.markdown("**htype**")
            h_dtype.markdown("**dtype**")
            h_comp.markdown("**compress**")
            h_btns.markdown("**+/−**")

            def _opts_with_value(base_opts, value):
                """Return a dropdown-options list that always contains ``value``.

                A schema loaded from JSON may carry a dtype / compression the
                preset shortlist does not include (e.g. ``"Any"``, ``"List"``,
                ``"jpeg"``). Without this, ``selectbox`` silently resets to
                index 0 and the user's explicit choice is lost.
                """
                if value in base_opts:
                    return base_opts
                return list(base_opts) + [value]

            schema_inputs = []
            rows = st.session_state.schema_rows
            for pos, row in enumerate(rows):
                rid = row["id"]
                c_name, c_htype, c_dtype, c_comp, c_btns = st.columns([3, 2, 2, 2, 1])
                with c_name:
                    col_name = st.text_input("n", value=row["name"], key=f"col_name_{rid}",
                                             label_visibility="collapsed")
                with c_htype:
                    _htype_opts = _opts_with_value(HTYPE_OPTIONS, row["htype"])
                    col_htype = st.selectbox("h", _htype_opts,
                                             index=_htype_opts.index(row["htype"]),
                                             key=f"col_htype_{rid}", label_visibility="collapsed")
                with c_dtype:
                    _dtype_opts = _opts_with_value(DTYPE_OPTIONS, row["dtype"])
                    col_dtype = st.selectbox("d", _dtype_opts,
                                             index=_dtype_opts.index(row["dtype"]),
                                             key=f"col_dtype_{rid}", label_visibility="collapsed")
                with c_comp:
                    _comp_opts = _opts_with_value(COMPRESSION_OPTIONS, row["comp"])
                    col_comp = st.selectbox("c", _comp_opts,
                                            index=_comp_opts.index(row["comp"]),
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

            # Detect that the user is about to create a dataset with the
            # *unchanged* placeholder schema. This is almost always a
            # mistake for demo / real data: the dataset ends up with only
            # labels/categories/description, and a subsequent Batch Upload
            # CSV silently drops every column not in that placeholder set.
            _current_rows_sig = [
                {"name": nm, "htype": ht, "dtype": dt, "comp": cm}
                for nm, ht, dt, cm in schema_inputs if nm
            ]
            _default_rows_sig = [
                {k: v for k, v in r.items() if k != "id"} for r in _DEFAULT_ROWS
            ]
            _is_untouched_placeholder = _current_rows_sig == _default_rows_sig

            if _is_untouched_placeholder:
                st.warning(
                    "⚠️ The schema editor below still shows the **default "
                    "placeholder** (`labels`, `categories`, `description`). "
                    "This is just a demo skeleton — it is **not** the COCO "
                    "layout. If you create the dataset now and then Batch "
                    "Upload `public_base_3000.csv`, every column except "
                    "`description` will be silently dropped (no bbox, no "
                    "category_id, no images → the dataset will not be "
                    "recognized as COCO and the 5×5 thumbnail + bbox "
                    "overlay UI will not appear).  \n\n"
                    "**Load `schema.json` in the expander above first**, or "
                    "tick the confirmation below if you really intended the "
                    "placeholder schema."
                )
                _confirm_placeholder = st.checkbox(
                    "I know this is the placeholder schema — create it anyway",
                    value=False,
                    key="create_ds_confirm_placeholder",
                )
            else:
                _confirm_placeholder = True

            if st.button("Create Dataset", type="primary",
                         disabled=_is_untouched_placeholder and not _confirm_placeholder):
                # Validate column names
                col_names = [s[0] for s in schema_inputs if s[0]]
                if not col_names:
                    st.error("Please define at least one column.")
                elif len(col_names) != len(set(col_names)):
                    st.error("Column names must be unique.")
                else:
                    raw = (ds_path_input or "").strip()
                    p = Path(raw).expanduser()
                    if not raw or not p.name:
                        st.error("Please enter a valid dataset path (including a folder name).")
                    else:
                        ds_root = str(p.parent)
                        ds_name = p.name
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
            _default_load_path = "/Users/sherrylin/Documents/research_data/muller_datasetcoco"
            load_path = st.text_input(
                "Dataset Path",
                value=_default_load_path,
                key="load_path",
                help="Full path to an existing MULLER dataset directory.",
            )
            if st.button("Load Dataset"):
                if not load_path:
                    st.error("Please enter a path")
                else:
                    # Release the previous dataset's write lock first; otherwise
                    # re-loading the same path in the same Streamlit session
                    # races itself for the lock and falls back to read-only.
                    release_dataset_lock(st.session_state.dataset)
                    st.session_state.dataset = None

                    ds, error, warn = load_dataset(load_path)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.dataset = ds
                        st.session_state.dataset_path = load_path
                        st.session_state.current_branch = ds.branch
                        st.session_state["_load_warning"] = warn  # may be None
                        st.success(f"Dataset loaded from: {load_path}")
                        if warn:
                            st.warning(warn)
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

            # Per-action "flash" slots — each Add / Import / Delete / Update
            # button writes its result here as a (kind, message) tuple before
            # calling st.rerun(); the matching `_render_flash` call below the
            # button picks it up on the next script run and renders it inline,
            # because the original `st.success()` was wiped by the rerun.
            # Auto-clear them whenever the user loads a different dataset to
            # avoid stale "Deleted sample N" lines on a fresh dataset.
            _flash_keys = ("_flash_add", "_flash_csv", "_flash_del", "_flash_upd")
            _flash_ctx = (st.session_state.get("dataset_path"), ds.branch)
            if st.session_state.get("_flash_ctx") != _flash_ctx:
                st.session_state["_flash_ctx"] = _flash_ctx
                for _k in _flash_keys:
                    st.session_state.pop(_k, None)

            def _render_flash(key: str):
                payload = st.session_state.get(key)
                if not payload:
                    return
                kind, msg = payload
                if kind == "success":
                    st.success(msg)
                elif kind == "error":
                    st.error(msg)
                else:
                    st.info(msg)

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
                _dp = st.session_state.get("dataset_path")
                if st.session_state.get("_dm_view_dataset_path") != _dp:
                    st.session_state["_dm_view_dataset_path"] = _dp
                    st.session_state["dm_view_page_idx"] = 1

                _coco_layout = is_coco2017_muller_schema(ds)
                img_tensors = list_image_tensor_names(ds)

                if not _coco_layout:
                    key_page = "dm_view_page_idx"
                    key_ps = "dm_view_page_size"
                    if key_page not in st.session_state:
                        st.session_state[key_page] = 1
                    pg1, pg2, pg3 = st.columns([1, 1, 3])
                    with pg1:
                        page_size = st.selectbox(
                            "Rows per page",
                            options=[10, 25, 50, 100],
                            index=0,
                            key=key_ps,
                        )
                    total_pages = max(1, (n + page_size - 1) // page_size)
                    if st.session_state[key_page] > total_pages:
                        st.session_state[key_page] = total_pages
                    with pg2:
                        page = st.number_input(
                            "Page",
                            min_value=1,
                            max_value=total_pages,
                            step=1,
                            key=key_page,
                            help="Go to a page (default shows the first page only).",
                        )
                    start_idx = (int(page) - 1) * page_size
                    end_idx = min(start_idx + page_size, n)
                    with pg3:
                        st.caption(
                            f"Showing samples **{start_idx + 1}–{end_idx}** of **{n}** · "
                            f"page **{int(page)}** / **{total_pages}**"
                        )

                    show_details = st.checkbox(
                        "Show details",
                        value=False,
                        key="dm_show_table_details",
                        help="When enabled, loads and shows the full tabular view for this page. "
                        "Turn off for a lighter UI (e.g. image preview only).",
                    )
                else:
                    start_idx = 0
                    end_idx = n
                    show_details = False
                    st.caption(
                        f"**{n}** samples · COCO layout uses a fixed **5×5** thumbnail grid "
                        "with its own paging (no table pagination here)."
                    )

                # Image strip preview (same row order as paginated rows; global # = row index in dataset)
                if img_tensors:
                    if not pil_preview_available():
                        st.info("Install **Pillow** (`pip install pillow`) to enable image preview.")
                    else:
                        st.markdown("##### Image preview")
                        if len(img_tensors) == 1:
                            preview_col = img_tensors[0]
                        else:
                            preview_col = st.selectbox(
                                "Image column",
                                img_tensors,
                                key="dm_img_preview_tensor",
                            )
                        _coco_overlay = False
                        _coco_cat_map = None
                        if _coco_layout:
                            st.caption(
                                "COCO2017-style layout detected — the 8 core tensors "
                                "(area, bbox, category_id, id, image_id, images, iscrowd, segmentation) "
                                "are all present; any extra columns (e.g. `description`) are allowed."
                            )
                            _ann_path = st.text_input(
                                "COCO instances JSON (for category names)",
                                value=DEFAULT_COCO_INSTANCES_JSON,
                                key="dm_coco_ann_json",
                                help="Same file as pycocotools.COCO(...), e.g. instances_val2017.json.",
                            )
                            _coco_overlay = st.checkbox(
                                "Overlay bounding boxes & labels",
                                value=True,
                                key="dm_coco_overlay",
                            )
                            if _coco_overlay:
                                _ap = (_ann_path or "").strip()
                                # Only load when path is non-empty; never overwrite a good map with
                                # a failed load from a transient empty path (e.g. widget timing on rerun).
                                if _ap and (
                                    st.session_state.get("_dm_coco_cat_map_path") != _ap
                                    or st.session_state.get("_dm_coco_cat_map") is None
                                ):
                                    _m, _cerr = load_coco_category_id_to_name(_ap)
                                    st.session_state["_dm_coco_cat_map_path"] = _ap
                                    st.session_state["_dm_coco_cat_map"] = _m
                                    st.session_state["_dm_coco_cat_err"] = _cerr
                                _coco_cat_map = st.session_state.get("_dm_coco_cat_map")
                                _cerr = st.session_state.get("_dm_coco_cat_err")
                                if _cerr and _ap:
                                    st.warning(_cerr)
                        if _coco_layout:
                            _COCO_THUMB_GRID = 5
                            _COCO_THUMB_PER_PAGE = _COCO_THUMB_GRID * _COCO_THUMB_GRID
                            n_thumb_pages = max(
                                1, (n + _COCO_THUMB_PER_PAGE - 1) // _COCO_THUMB_PER_PAGE
                            )
                            if "dm_coco_thumb_page" not in st.session_state:
                                st.session_state["dm_coco_thumb_page"] = 1
                            tp_cur = int(st.session_state.get("dm_coco_thumb_page", 1))
                            if tp_cur > n_thumb_pages:
                                st.session_state["dm_coco_thumb_page"] = n_thumb_pages
                                tp_cur = n_thumb_pages
                            thumb_offset = (tp_cur - 1) * _COCO_THUMB_PER_PAGE
                            thumb_len = min(_COCO_THUMB_PER_PAGE, n - thumb_offset)
                            n_page = thumb_len
                            try:
                                _vimg = ds[thumb_offset : thumb_offset + thumb_len]
                                raw_images = list(
                                    _vimg[preview_col].numpy(aslist=True)
                                )
                            except Exception as _e:
                                st.warning(f"Could not load images: {_e}")
                                raw_images = []
                        else:
                            n_page = end_idx - start_idx
                            try:
                                _vimg = ds[start_idx:end_idx]
                                raw_images = list(
                                    _vimg[preview_col].numpy(aslist=True)
                                )
                            except Exception as _e:
                                st.warning(f"Could not load images: {_e}")
                                raw_images = []
                        if raw_images:
                            if _coco_layout:
                                _coco_view_ctx = (
                                    _dp,
                                    str(st.session_state.get("current_branch")),
                                    n,
                                    preview_col,
                                )
                                if (
                                    st.session_state.get("_dm_coco_view_ctx")
                                    != _coco_view_ctx
                                ):
                                    st.session_state["_dm_coco_view_ctx"] = _coco_view_ctx
                                    st.session_state["dm_coco_thumb_page"] = 1
                                    st.session_state["dm_coco_selected_gi"] = 0

                                sel_gi = int(
                                    st.session_state.get("dm_coco_selected_gi", 0)
                                )
                                if sel_gi < 0 or sel_gi >= n:
                                    sel_gi = 0
                                    st.session_state["dm_coco_selected_gi"] = sel_gi
                                if (
                                    sel_gi < thumb_offset
                                    or sel_gi >= thumb_offset + thumb_len
                                ):
                                    sel_gi = thumb_offset
                                    st.session_state["dm_coco_selected_gi"] = sel_gi

                                st.caption(
                                    "COCO layout: **thumbnail grid** (left) and **full preview** "
                                    "(right). Tap **#index** under a thumbnail to update the right pane "
                                    "(same-page rerun; no URL navigation)."
                                )
                                left_pane, right_pane = st.columns([0.4, 0.6], gap="large")

                                with left_pane:
                                    st.markdown("**Thumbnails** (fixed **5×5** grid, 25 per page)")
                                    nav_a, nav_b, nav_c = st.columns([1, 1, 2])
                                    with nav_a:
                                        st.button(
                                            "◀ Prev",
                                            key="dm_coco_thumb_prev",
                                            disabled=(tp_cur <= 1),
                                            on_click=_dm_coco_thumb_page_delta,
                                            args=(-1, n_thumb_pages),
                                        )
                                    with nav_b:
                                        st.button(
                                            "Next ▶",
                                            key="dm_coco_thumb_next",
                                            disabled=(tp_cur >= n_thumb_pages),
                                            on_click=_dm_coco_thumb_page_delta,
                                            args=(1, n_thumb_pages),
                                        )
                                    with nav_c:
                                        st.number_input(
                                            "Thumbnail page",
                                            min_value=1,
                                            max_value=n_thumb_pages,
                                            step=1,
                                            key="dm_coco_thumb_page",
                                            help=(
                                                f"Showing global samples **{thumb_offset + 1}–"
                                                f"{thumb_offset + thumb_len}** of **{n}**."
                                            ),
                                        )
                                    st.caption(
                                        f"Page **{tp_cur}** / **{n_thumb_pages}** · "
                                        f"samples **{thumb_offset + 1}–{thumb_offset + thumb_len}**"
                                    )
                                    tc = tr = _COCO_THUMB_GRID
                                    thumb_edge = max(36, min(128, 300 // tc))
                                    for rr in range(tr):
                                        grid_cols = st.columns(tc)
                                        for ci in range(tc):
                                            slot = rr * tc + ci
                                            with grid_cols[ci]:
                                                if slot >= thumb_len:
                                                    st.empty()
                                                    continue
                                                j = slot
                                                gi = thumb_offset + j
                                                _pil_dec = decode_muller_image_sample(
                                                    raw_images[j]
                                                )
                                                _sq = (
                                                    pil_square_thumbnail(
                                                        _pil_dec, int(thumb_edge)
                                                    )
                                                    if _pil_dec
                                                    else None
                                                )
                                                if _sq is not None:
                                                    st.image(_sq)
                                                    st.button(
                                                        f"#{gi}",
                                                        key=f"dm_coco_pick_{gi}",
                                                        type=(
                                                            "primary"
                                                            if gi == sel_gi
                                                            else "secondary"
                                                        ),
                                                        use_container_width=True,
                                                        on_click=_dm_on_select_coco_thumb,
                                                        args=(gi,),
                                                        help="Show this sample on the right",
                                                    )
                                                else:
                                                    st.caption("—")

                                with right_pane:
                                    st.markdown("**Full image**")
                                    pv_w = st.slider(
                                        "Preview pane width (px)",
                                        min_value=480,
                                        max_value=1200,
                                        value=920,
                                        step=20,
                                        key="dm_coco_pv_w",
                                    )
                                    pv_h = st.slider(
                                        "Preview pane height (px)",
                                        min_value=400,
                                        max_value=1000,
                                        value=720,
                                        step=20,
                                        key="dm_coco_pv_h",
                                    )
                                    try:
                                        _one = ds[sel_gi : sel_gi + 1]
                                        _raw_sel = list(
                                            _one[preview_col].numpy(aslist=True)
                                        )[0]
                                    except Exception:
                                        _raw_sel = None
                                    _pil_full = decode_muller_image_sample(
                                        _raw_sel
                                    )
                                    if (
                                        _pil_full is not None
                                        and _coco_overlay
                                    ):
                                        try:
                                            _bb = ds.bbox[sel_gi].numpy(aslist=True)
                                            _cid = ds.category_id[sel_gi].numpy(
                                                aslist=True
                                            )
                                            _pil_full = pil_overlay_coco_bboxes(
                                                _pil_full,
                                                _bb,
                                                _cid,
                                                _coco_cat_map,
                                            )
                                        except Exception:
                                            pass
                                    _fitted = (
                                        pil_fit_inside(
                                            _pil_full,
                                            int(pv_w),
                                            int(pv_h),
                                        )
                                        if _pil_full
                                        else None
                                    )
                                    if _fitted is not None:
                                        st.image(_fitted, use_container_width=False)
                                    else:
                                        st.markdown(
                                            f'<div style="width:100%;max-width:{int(pv_w)}px;'
                                            f"height:{int(pv_h)}px;background:#ececf0;"
                                            'border-radius:8px;display:flex;align-items:center;'
                                            'justify-content:center;color:#666;">'
                                            "Decode failed for this sample.</div>",
                                            unsafe_allow_html=True,
                                        )
                                    st.caption(
                                        f"Sample **#{sel_gi}** · overlay "
                                        f"{'on' if _coco_overlay else 'off'} · "
                                        "pane shows native resolution scaled down to fit (no upscaling)."
                                    )
                                    _cap = (
                                        f"Sample index **{sel_gi}** (native resolution)"
                                    )
                                    _zk = f"dm_fullres_coco_{sel_gi}"
                                    if _pil_full is not None:
                                        if _dm_fullres_dialog is not None:
                                            if st.button(
                                                "Open full resolution",
                                                key=_zk,
                                                help="Dialog at native image size",
                                            ):
                                                st.session_state["dm_fullres_pil"] = (
                                                    _pil_full.copy()
                                                )
                                                st.session_state[
                                                    "dm_fullres_caption"
                                                ] = _cap
                                                _dm_fullres_dialog()
                                        elif hasattr(st, "popover"):
                                            with st.popover("Full resolution"):
                                                st.image(
                                                    _pil_full,
                                                    use_container_width=False,
                                                )
                                                st.caption(_cap)
                                        else:
                                            with st.expander("Full resolution"):
                                                st.image(
                                                    _pil_full,
                                                    use_container_width=False,
                                                )
                                                st.caption(_cap)
                            else:
                                _strip_ctx = (
                                    start_idx,
                                    end_idx,
                                    preview_col,
                                    int(page),
                                    page_size,
                                )
                                if (
                                    st.session_state.get("_dm_img_strip_ctx")
                                    != _strip_ctx
                                ):
                                    st.session_state["_dm_img_strip_ctx"] = _strip_ctx
                                    st.session_state["dm_img_strip_pan"] = 0
                                    _max_cols = min(12, max(1, n_page))
                                    st.session_state["dm_img_cols_per_row"] = min(
                                        4, _max_cols
                                    )
                                max_cols = min(12, max(1, n_page))
                                c_a, c_b, c_c = st.columns([1.1, 1.1, 1.0])
                                with c_a:
                                    if max_cols > 1:
                                        st.slider(
                                            "Thumbnails per row",
                                            min_value=1,
                                            max_value=max_cols,
                                            key="dm_img_cols_per_row",
                                            help=f"Grid uses {_DM_PREVIEW_ROWS} rows (up to "
                                            f"{max_cols * _DM_PREVIEW_ROWS} thumbnails visible).",
                                        )
                                    else:
                                        st.caption(
                                            "Single sample on this page — one thumbnail."
                                        )
                                cols_per_row = (
                                    1
                                    if max_cols <= 1
                                    else int(
                                        st.session_state.get(
                                            "dm_img_cols_per_row",
                                            min(4, max_cols),
                                        )
                                    )
                                )
                                cols_per_row = max(1, min(cols_per_row, max_cols))
                                vis = min(cols_per_row * _DM_PREVIEW_ROWS, n_page)
                                pan_max = max(0, n_page - vis)
                                if pan_max > 0:
                                    if (
                                        st.session_state.get("dm_img_strip_pan", 0)
                                        > pan_max
                                    ):
                                        st.session_state["dm_img_strip_pan"] = pan_max
                                    with c_b:
                                        st.slider(
                                            "Pan (this page)",
                                            min_value=0,
                                            max_value=pan_max,
                                            key="dm_img_strip_pan",
                                            help="Same order as this page's samples (and as the details table when shown). "
                                            "**#** is the 0-based sample index in the dataset.",
                                        )
                                else:
                                    st.session_state["dm_img_strip_pan"] = 0
                                    with c_b:
                                        st.caption(
                                            f"All samples on this page fit in the {_DM_PREVIEW_ROWS}-row grid — "
                                            "no panning needed."
                                        )
                                with c_c:
                                    px_h = st.slider(
                                        "Preview height (px)",
                                        min_value=120,
                                        max_value=400,
                                        value=200,
                                        step=10,
                                        key="dm_img_px_h",
                                    )
                                pan = (
                                    0
                                    if pan_max <= 0
                                    else int(
                                        st.session_state.get("dm_img_strip_pan", 0)
                                    )
                                )
                                row_indices = list(
                                    range(pan, min(pan + vis, n_page))
                                )
                                if row_indices:
                                    for row_i in range(_DM_PREVIEW_ROWS):
                                        chunk = row_indices[
                                            row_i
                                            * cols_per_row : (row_i + 1)
                                            * cols_per_row
                                        ]
                                        if not chunk:
                                            break
                                        prev_cols = st.columns(len(chunk))
                                        for slot, j in enumerate(chunk):
                                            with prev_cols[slot]:
                                                _pil_full = decode_muller_image_sample(
                                                    raw_images[j]
                                                )
                                                if _pil_full is not None:
                                                    _thumb = pil_resize_to_height(
                                                        _pil_full, int(px_h)
                                                    )
                                                else:
                                                    _thumb = None
                                                if _thumb is not None:
                                                    st.image(_thumb)
                                                else:
                                                    st.caption("_(decode failed)_")
                                                st.caption(f"**#{start_idx + j}**")
                                                _cap = f"Sample index **{start_idx + j}** (full resolution)"
                                                _zk = f"dm_fullres_{start_idx}_{j}"
                                                if _pil_full is not None:
                                                    if _dm_fullres_dialog is not None:
                                                        if st.button(
                                                            "Full size",
                                                            key=_zk,
                                                            help="Open at native image resolution",
                                                        ):
                                                            st.session_state[
                                                                "dm_fullres_pil"
                                                            ] = _pil_full.copy()
                                                            st.session_state[
                                                                "dm_fullres_caption"
                                                            ] = _cap
                                                            _dm_fullres_dialog()
                                                    elif hasattr(st, "popover"):
                                                        with st.popover("Full size"):
                                                            st.image(
                                                                _pil_full,
                                                                use_container_width=False,
                                                            )
                                                            st.caption(_cap)
                                                    else:
                                                        with st.expander("Full size"):
                                                            st.image(
                                                                _pil_full,
                                                                use_container_width=False,
                                                            )
                                                            st.caption(_cap)

                if show_details:
                    df, err = dataset_to_dataframe(ds, start=start_idx, end=end_idx)
                    if err:
                        st.error(err)
                    else:
                        st.dataframe(dataframe_for_streamlit_display(df), width="stretch")

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

                add_commit_msg = st.text_input(
                    "Commit Message", value="Add sample via Streamlit UI",
                    key="add_sample_commit_msg")
                if st.button("Add Sample", type="primary"):
                    filtered = {k: v for k, v in sample_data.items() if v is not None}
                    if not filtered:
                        # Validation feedback can be inline (no rerun), so write
                        # it directly to the flash slot too for consistency.
                        st.session_state["_flash_add"] = ("error", "Please fill in at least one field.")
                    else:
                        err = add_samples(ds, filtered, auto_commit=True,
                                          commit_message=add_commit_msg)
                        if err:
                            st.session_state["_flash_add"] = ("error", err)
                        else:
                            st.session_state["_flash_add"] = (
                                "success",
                                f"Sample added and committed (commit: `{add_commit_msg}`).",
                            )
                            st.rerun()
                _render_flash("_flash_add")

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

                    # Hard-stop warning for the specific failure mode where
                    # the user tries to import a rich CSV (e.g. 9-column
                    # COCO public_base_3000.csv) into a dataset that was
                    # created with the default placeholder schema — only
                    # a handful of columns match and the rest vanish, which
                    # is almost never what the user wants. Make this loud
                    # and actionable instead of a single-line warning.
                    if matched and len(unmatched) >= 2 * len(matched):
                        st.error(
                            f"**CSV columns barely match the dataset schema** "
                            f"({len(matched)} matched vs {len(unmatched)} "
                            f"skipped). This usually means the dataset was "
                            f"created without loading the correct schema "
                            f"JSON (e.g. you kept the default `labels / "
                            f"categories / description` placeholder). If you "
                            f"import now, every column in the 'skipped' list "
                            f"will be lost.  \n\n"
                            f"**Fix**: go to **Create / Load → Create New "
                            f"Dataset**, tick *Overwrite if exists*, load "
                            f"`schema.json` in the expander, recreate, then "
                            f"come back here and re-upload this CSV."
                        )

                    # Per-column import mode:
                    #  - "read" for media columns (calls muller.read() on the path)
                    #  - "json" for cells whose payload is a JSON array/object
                    #    (per-sample arrays like bbox (N,4), segmentation polygons,
                    #    variable-length area / category_id / id / iscrowd)
                    #  - "skip" (i.e. not in path_columns) for plain scalars
                    #
                    # Why sniff the CSV content instead of the tensor metadata:
                    # at Batch Upload time the target tensor was just created
                    # and holds no samples, so tensor.shape is (0,) and we
                    # cannot tell from the schema alone whether a `generic`
                    # column will hold one number or a length-N array. The
                    # CSV cell itself is the source of truth — a leading `[`
                    # or `{` unambiguously means JSON. We peek a handful of
                    # rows to be robust to blank leading cells.
                    def _first_non_null(series, n=8):
                        try:
                            return [v for v in series.dropna().head(n).tolist()]
                        except Exception:
                            return []

                    def _looks_json(values):
                        for v in values:
                            s = str(v).strip()
                            if not s:
                                continue
                            if s.startswith("[") or s.startswith("{"):
                                return True
                        return False

                    def _infer_mode(col_name, tensor):
                        h = (tensor.htype or "").lower()
                        if h in ("image", "video", "audio"):
                            return "read"
                        # Some htypes are inherently non-scalar regardless of
                        # what the CSV looks like; catch them explicitly so a
                        # malformed / empty first few cells don't regress to
                        # skip and then break on append.
                        if h in ("bbox", "list", "polygon", "keypoints_coco",
                                 "segment_mask", "binary_mask", "embedding",
                                 "json", "vector", "point"):
                            return "json"
                        if _looks_json(_first_non_null(df_up[col_name])):
                            return "json"
                        return "skip"

                    inferred = {col: _infer_mode(col, ds.tensors[col]) for col in matched}
                    media_cols = [c for c in matched if inferred[c] == "read"]
                    array_cols = [c for c in matched if inferred[c] == "json"]
                    needs_selector = media_cols + array_cols

                    path_columns = {}
                    if needs_selector:
                        st.markdown(
                            "**Path / array columns** — the UI inferred these "
                            "from each column's CSV content; override if needed:"
                        )
                        for col in needs_selector:
                            t = ds.tensors[col]
                            is_media = col in media_cols
                            if is_media:
                                opts = ["read", "text", "skip"]
                                col_help = (
                                    "read: load file via muller.read(); "
                                    "text: store path as text; "
                                    "skip: treat as plain value"
                                )
                            else:
                                opts = ["json", "text", "skip"]
                                col_help = (
                                    "json: parse cell with json.loads (use this "
                                    "for per-sample arrays like bbox / "
                                    "segmentation / variable-length labels); "
                                    "text: store as plain string; "
                                    "skip: pass through unchanged"
                                )
                            default_idx = opts.index(inferred[col]) if inferred[col] in opts else 0
                            mode = st.selectbox(
                                f"`{col}` ({t.htype})",
                                options=opts,
                                index=default_idx,
                                help=col_help,
                                key=f"csv_pathcol_{col}",
                            )
                            if mode != "skip":
                                path_columns[col] = mode

                    csv_commit_msg = st.text_input(
                        "Commit Message", value="Import CSV data via Streamlit UI",
                        key="csv_commit_msg")
                    if st.button("Import CSV Data"):
                        if not matched:
                            st.session_state["_flash_csv"] = (
                                "error", f"CSV columns must match tensors: {tensor_names}"
                            )
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
                                commit_dataset(ds, message=csv_commit_msg)
                                st.session_state["_flash_csv"] = (
                                    "success",
                                    f"Imported {len(df_up)} samples (commit: `{csv_commit_msg}`).",
                                )
                                st.rerun()
                            except Exception as e:
                                st.session_state["_flash_csv"] = ("error", f"Failed to import CSV: {e}")
                            finally:
                                if tmp_path and os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                _render_flash("_flash_csv")

            # 5) Delete Sample (collapsed by default)
            with st.expander("Delete Sample", expanded=False):
                if n == 0:
                    st.info("No samples to delete.")
                else:
                    del_idx = st.number_input("Sample Index", min_value=0, max_value=max(n - 1, 0), value=0)
                    del_commit_msg = st.text_input(
                        "Commit Message", value="Delete sample via Streamlit UI",
                        key="del_commit_msg")
                    if st.button("Delete", type="secondary"):
                        err = delete_sample(ds, del_idx)
                        if err:
                            st.session_state["_flash_del"] = ("error", err)
                        else:
                            commit_dataset(ds, message=del_commit_msg)
                            st.session_state["_flash_del"] = (
                                "success",
                                f"Deleted sample {del_idx} (commit: `{del_commit_msg}`).",
                            )
                            st.rerun()
                _render_flash("_flash_del")

            # 6) Update Sample (collapsed by default)
            with st.expander("Update Sample", expanded=False):
                if n == 0:
                    st.info("No samples to update.")
                else:
                    upd_idx = st.number_input("Sample Index to Update", min_value=0, max_value=max(n - 1, 0), value=0, key="upd_idx")
                    upd_tensor = st.selectbox("Tensor", list(ds.tensors.keys()), key="upd_tensor")
                    upd_val = st.text_input("New Value", key="upd_val")
                    upd_commit_msg = st.text_input(
                        "Commit Message", value="Update sample via Streamlit UI",
                        key="upd_commit_msg")
                    if st.button("Update", type="secondary"):
                        # Try JSON first so users can paste array literals like
                        # "[1]" or "[[1,2,3,4]]" (matches the CSV-import "json"
                        # mode for non-scalar tensors); fall back to int / float
                        # / raw string for scalar tensors.
                        parsed = None
                        _stripped = (upd_val or "").strip()
                        if _stripped and _stripped[0] in "[{":
                            try:
                                import json as _json
                                parsed = _json.loads(_stripped)
                            except ValueError:
                                parsed = None
                        if parsed is None:
                            try:
                                parsed = int(upd_val)
                            except ValueError:
                                try:
                                    parsed = float(upd_val)
                                except ValueError:
                                    parsed = upd_val
                        err = update_sample(ds, upd_tensor, upd_idx, parsed)
                        if err:
                            st.session_state["_flash_upd"] = ("error", err)
                        else:
                            commit_dataset(ds, message=upd_commit_msg)
                            st.session_state["_flash_upd"] = (
                                "success",
                                f"Updated `{upd_tensor}[{upd_idx}]` -> `{parsed}` (commit: `{upd_commit_msg}`).",
                            )
                            st.rerun()
                _render_flash("_flash_upd")


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

        # Result of the most recent query/view load lives across reruns so the
        # user can scroll, tweak display options, and then save it as a view
        # without losing it. Invalidated whenever the dataset identity changes.
        _qs_ctx = (st.session_state.get("dataset_path"), ds.branch, len(ds))
        if st.session_state.get("_qs_result_ctx") != _qs_ctx:
            st.session_state["_qs_result_ctx"] = _qs_ctx
            st.session_state["qs_result_ds"] = None
            st.session_state["qs_result_desc"] = None
            st.session_state["qs_result_meta"] = None

        # ── Saved views (expander at top) ──────────────────────────────────
        _me = current_user()
        with st.expander(f"📁 Saved Views · you are `{_me}`", expanded=False):
            col_refresh, _ = st.columns([1, 6])
            with col_refresh:
                if st.button("Refresh", key="qs_views_refresh"):
                    st.rerun()
            view_rows, view_err = list_saved_views(ds)
            if view_err:
                st.error(view_err)
            if view_rows:
                # Decorate each row so the table makes ownership obvious at a glance.
                _display_rows = [
                    {
                        "owner": f"👤 you ({r['owner']})" if r["owner"] == _me
                        else f"{r['owner']} (read-only)",
                        "id": r["id"],
                        "message": r["message"],
                        "commit": r["commit_id"],
                        "virtual": r["virtual"],
                    }
                    for r in view_rows
                ]
                st.dataframe(pd.DataFrame(_display_rows), width="stretch", hide_index=True)
            else:
                st.caption("No views saved yet. Run a query below and save the result.")

            # owner_of[view_id] → owner uid, so the Delete button can gate itself.
            owner_of = {r["id"]: r["owner"] for r in view_rows}
            existing_ids = list(owner_of.keys())

            col_sel, col_input = st.columns(2)
            with col_sel:
                picked = st.selectbox(
                    "Pick from saved views",
                    options=["(type an ID instead)"] + existing_ids,
                    key="qs_view_pick",
                )
            with col_input:
                typed = st.text_input(
                    "…or type a view ID",
                    key="qs_view_typed",
                    help="Takes precedence over the dropdown when non-empty.",
                )
            target_vid = typed.strip() or (picked if picked != "(type an ID instead)" else "")
            _target_owner = owner_of.get(target_vid) if target_vid else None
            _is_mine = (_target_owner == _me) if _target_owner else False

            col_load, col_del = st.columns(2)
            with col_load:
                if st.button("Load View", type="primary", key="qs_view_load",
                             disabled=(not target_vid)):
                    view_ds, view_entry, err = load_saved_view(ds, target_vid)
                    if err:
                        st.error(err)
                    else:
                        st.session_state["qs_result_ds"] = view_ds
                        st.session_state["qs_result_desc"] = (
                            f"View `{target_vid}` (length = {len(view_ds)})"
                        )
                        _owner = str(view_entry.get("uid") or "-") if view_entry else "-"
                        st.session_state["qs_result_meta"] = {
                            "source": "view",
                            "view_id": target_vid,
                            "owner": _owner,
                            "is_mine": _owner == _me,
                            "source_commit": getattr(view_entry, "commit_id", "") or "",
                            "message": getattr(view_entry, "message", "") or "",
                            "query": getattr(view_entry, "query", None),
                            "tql_query": getattr(view_entry, "tql_query", None),
                        }
                        st.rerun()
            with col_del:
                # Owner-only: the backend would raise UnAuthorizationError for
                # non-owners anyway; gating here avoids a scary stack trace
                # during the live demo and makes the rule visually obvious.
                _can_delete = bool(target_vid) and _is_mine
                _del_help = None
                if target_vid and not _is_mine:
                    _del_help = (
                        f"View `{target_vid}` is owned by `{_target_owner}`. "
                        "Only its owner can delete it."
                    )
                _confirm_del = st.checkbox(
                    f"Confirm delete `{target_vid}`" if target_vid else "Confirm delete",
                    key="qs_view_del_confirm",
                    disabled=not _can_delete,
                    help=_del_help,
                )
                if st.button("Delete View", key="qs_view_delete",
                             disabled=not _can_delete, help=_del_help):
                    if not _confirm_del:
                        st.warning("Please tick the confirmation box before deleting.")
                    else:
                        err = delete_saved_view(ds, target_vid)
                        if err:
                            st.error(err)
                        else:
                            st.success(f"Deleted view '{target_vid}'.")
                            st.rerun()
                if target_vid and not _is_mine:
                    st.caption(f"🔒 Owned by `{_target_owner}` — read-only to you.")

        st.markdown("---")

        # ── Mode switch: a single unified "Run Query" form ─────────────────
        mode = st.radio(
            "Query mode",
            ["Conditional filter", "Vector search"],
            key="qs_mode",
            horizontal=True,
            help="Pick one approach per run. The two modes share the result "
            "pane below and the 'Save as view' step.",
        )

        if mode == "Conditional filter":
            st.subheader("Conditional Filtering")

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

            if st.button("Run Query", type="primary", key="qs_run_filter"):
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

                def _idx_progress(field: str) -> None:
                    st.info(
                        f"No inverted index found for `{field}` — building it "
                        "now (one-time cost; reused on later CONTAINS queries)."
                    )

                with st.spinner("Running query..."):
                    result_ds, err = run_query(
                        ds,
                        parsed_conds,
                        connectors if connectors else None,
                        progress_cb=_idx_progress,
                    )
                if err:
                    st.error(err)
                else:
                    n_results = len(result_ds)
                    st.session_state["qs_result_ds"] = result_ds
                    st.session_state["qs_result_desc"] = (
                        f"Conditional filter → {n_results} matching samples"
                    )
                    st.session_state["qs_result_meta"] = {
                        "source": "filter",
                        "conditions": parsed_conds,
                        "connectors": list(connectors),
                    }

        else:  # Vector search
            st.subheader("Vector Similarity Search")

            vec_tensors = list_vector_tensor_names(ds)
            idx_map = list_vector_indexes(ds)

            if not vec_tensors and not idx_map:
                st.info(
                    "No `embedding`/`vector` tensor declared on this dataset. "
                    "Add one with `htype='embedding'` (float32), populate it, "
                    "then build an index via `ds.create_vector_index(...)`."
                )
            else:
                # Allow choosing any tensor that *has* an index, even if its
                # declared htype is not literally 'embedding' — so users with
                # generic float tensors + an index are not blocked.
                selectable = sorted(set(vec_tensors) | set(idx_map.keys()))
                col_t, col_i, col_k = st.columns([2, 2, 1])
                with col_t:
                    v_tensor = st.selectbox(
                        "Embedding tensor", selectable, key="qs_vec_tensor"
                    )
                indexes_for_tensor = idx_map.get(v_tensor, [])
                with col_i:
                    if indexes_for_tensor:
                        v_index = st.selectbox(
                            "Vector index", indexes_for_tensor, key="qs_vec_index"
                        )
                    else:
                        v_index = st.text_input(
                            "Vector index (will be created)",
                            value="demo_flat",
                            key="qs_vec_index_new",
                            help="No existing index on this tensor; one will be "
                            "built on the current commit when you run the search.",
                        )
                with col_k:
                    topk = st.number_input(
                        "top-k", min_value=1, max_value=1000, value=10, step=1,
                        key="qs_vec_topk",
                    )

                dim = tensor_embedding_dim(ds, v_tensor)
                qv_help = (
                    f"Comma/space-separated floats. Expected dimension: {dim}"
                    if dim else "Comma/space-separated floats."
                )
                qv_text = st.text_area(
                    "Query vector", key="qs_vec_text", height=100, help=qv_help,
                    placeholder="e.g. 0.12, -0.03, 0.88, ..." if not dim else
                    ", ".join(["0.0"] * min(int(dim), 8)) + (", ..." if dim and dim > 8 else ""),
                )

                col_metric, col_type = st.columns(2)
                with col_metric:
                    metric = st.selectbox("Metric", ["l2", "cosine", "ip"], key="qs_vec_metric",
                                          disabled=bool(indexes_for_tensor),
                                          help="Ignored when reusing an existing index "
                                          "(that index was built with its own metric).")
                with col_type:
                    index_type = st.selectbox("Index type", ["FLAT", "IVF", "IVFPQ"],
                                              key="qs_vec_idx_type",
                                              disabled=bool(indexes_for_tensor),
                                              help="Used only when creating a new index.")

                if st.button("Run Vector Search", type="primary", key="qs_run_vec"):
                    qvec, perr = parse_query_vector(qv_text, expected_dim=dim)
                    if perr:
                        st.error(perr)
                    else:
                        if not indexes_for_tensor:
                            with st.spinner(
                                f"Building vector index '{v_index}' on '{v_tensor}'…"
                            ):
                                ierr = ensure_vector_index(
                                    ds, v_tensor, v_index,
                                    index_type=index_type, metric=metric,
                                )
                            if ierr:
                                st.error(ierr)
                                st.stop()
                        with st.spinner("Searching…"):
                            result_ds, positions, dists, verr = run_vector_search(
                                ds, v_tensor, v_index, qvec, topk=int(topk),
                            )
                        if verr:
                            st.error(verr)
                        else:
                            n_hits = len(result_ds) if result_ds is not None else 0
                            st.session_state["qs_result_ds"] = result_ds
                            st.session_state["qs_result_desc"] = (
                                f"Vector search on `{v_tensor}` (index `{v_index}`, "
                                f"top-{int(topk)}) → {n_hits} hits"
                            )
                            st.session_state["qs_result_meta"] = {
                                "source": "vector",
                                "tensor": v_tensor,
                                "index": v_index,
                                "topk": int(topk),
                                "positions": list(positions or []),
                                "distances": dists.tolist() if dists is not None else [],
                            }

        # ── Shared result pane ─────────────────────────────────────────────
        st.markdown("---")
        result_ds = st.session_state.get("qs_result_ds")
        result_desc = st.session_state.get("qs_result_desc")
        result_meta = st.session_state.get("qs_result_meta") or {}

        if result_ds is None:
            st.caption("No active query result. Run a query above, or load a saved view.")
        else:
            n_results = len(result_ds)

            # ── Origin card for loaded views ─────────────────────────────
            # Views are the collaboration handoff primitive: seeing *who*
            # created the view and *what question* they asked is what turns
            # an opaque sample list into actionable context.
            if result_meta.get("source") == "view":
                _vid = result_meta.get("view_id", "")
                _owner = result_meta.get("owner", "-")
                _is_mine = bool(result_meta.get("is_mine"))
                _source_commit = (result_meta.get("source_commit") or "")[:8]
                _msg = result_meta.get("message") or ""
                _query = result_meta.get("query") or result_meta.get("tql_query") or ""

                if _is_mine:
                    _owner_badge = f"👤 **you** (`{_owner}`) · editable"
                else:
                    _owner_badge = f"🔒 **read-only** · owned by `{_owner}`"

                st.markdown(
                    f"### 📁 View `{_vid}`  \n{_owner_badge}"
                )
                meta_cols = st.columns(3)
                meta_cols[0].metric("Samples", n_results)
                meta_cols[1].metric("Source commit", _source_commit or "—")
                meta_cols[2].metric("Virtual", "yes" if result_meta.get("virtual", True) else "no")

                if _msg:
                    st.caption(f"**Message:** {_msg}")
                if _query:
                    st.markdown("**Original query:**")
                    st.code(_query, language="text")
                else:
                    st.caption(
                        "_No query string recorded — this view was saved from "
                        "a programmatic sub-view rather than a filter._"
                    )
                st.markdown("")
            else:
                st.success(result_desc or f"{n_results} samples in current result")

            # Vector-search hit list
            if result_meta.get("source") == "vector" and result_meta.get("positions"):
                pos = result_meta["positions"]
                dists = result_meta.get("distances", [])
                if len(dists) == len(pos):
                    hits_df = pd.DataFrame({
                        "rank": list(range(1, len(pos) + 1)),
                        "sample_idx": pos,
                        "distance": dists,
                    })
                else:
                    hits_df = pd.DataFrame({
                        "rank": list(range(1, len(pos) + 1)),
                        "sample_idx": pos,
                    })
                with st.expander("Nearest-neighbor hit list", expanded=False):
                    st.dataframe(hits_df, width="stretch", hide_index=True)

            # ── Image grid inspector ─────────────────────────────────────
            # Mostly matters for loaded views (engineer wants to visually
            # confirm what Alice's query produced before reusing it), but we
            # make it available for any image-bearing result.
            img_tensors_r = list_image_tensor_names(result_ds) if n_results > 0 else []
            if img_tensors_r:
                if not pil_preview_available():
                    st.info(
                        "Install **Pillow** (`pip install pillow`) to enable "
                        "the image grid preview."
                    )
                else:
                    with st.expander("🖼️ Inspect images", expanded=(result_meta.get("source") == "view")):
                        preview_col = (
                            img_tensors_r[0]
                            if len(img_tensors_r) == 1
                            else st.selectbox(
                                "Image column", img_tensors_r, key="qs_view_img_col"
                            )
                        )

                        # COCO overlay support — only bother auto-detecting on
                        # datasets that obviously follow that schema.
                        _is_coco = is_coco2017_muller_schema(result_ds)
                        _coco_cat_map = None
                        _coco_overlay = False
                        if _is_coco:
                            _coco_overlay = st.checkbox(
                                "Overlay bounding boxes & labels",
                                value=True, key="qs_view_coco_overlay",
                            )
                            if _coco_overlay:
                                _ann = st.text_input(
                                    "COCO instances JSON (for category names)",
                                    value=DEFAULT_COCO_INSTANCES_JSON,
                                    key="qs_view_coco_ann",
                                )
                                if _ann.strip():
                                    _coco_cat_map, _cerr = load_coco_category_id_to_name(_ann.strip())
                                    if _cerr:
                                        st.warning(_cerr)

                        per_page = 12
                        total_pages = max(1, (n_results + per_page - 1) // per_page)
                        grid_page = st.number_input(
                            "Page", min_value=1, max_value=total_pages, step=1,
                            key="qs_view_grid_page",
                        )
                        lo = (int(grid_page) - 1) * per_page
                        hi = min(lo + per_page, n_results)
                        st.caption(
                            f"Showing samples **{lo + 1}–{hi}** of **{n_results}** · "
                            f"page **{int(grid_page)}** / **{total_pages}**"
                        )
                        # Bulk-fetch per-page: this is the ONLY access pattern
                        # that works on a filter-view sub-dataset. Accessing
                        # `_chunk.bbox[k]` one-by-one returns an empty [] on
                        # filter views for non-image tensors (only the bulk
                        # `.numpy(aslist=True)` path respects the filter's
                        # sample-uuid remap), so the overlay used to silently
                        # render every image without boxes.
                        try:
                            _chunk = result_ds[lo:hi]
                            _raw_imgs = list(_chunk[preview_col].numpy(aslist=True))
                        except Exception as _e:
                            st.warning(f"Could not load images: {_e}")
                            _raw_imgs = []
                            _chunk = None

                        _chunk_bboxes = None
                        _chunk_cats = None
                        if _chunk is not None and _coco_overlay and _is_coco:
                            try:
                                _chunk_bboxes = list(_chunk.bbox.numpy(aslist=True))
                                _chunk_cats = list(_chunk.category_id.numpy(aslist=True))
                            except Exception as _e:
                                st.warning(
                                    "Could not load bbox/category_id for overlay: "
                                    f"{_e}"
                                )

                        if _raw_imgs:
                            cols_per_row = 4
                            for row_start in range(0, len(_raw_imgs), cols_per_row):
                                row_cells = st.columns(cols_per_row)
                                for j, cell in enumerate(row_cells):
                                    k = row_start + j
                                    if k >= len(_raw_imgs):
                                        continue
                                    with cell:
                                        _pil = decode_muller_image_sample(_raw_imgs[k])
                                        if (
                                            _pil is not None
                                            and _coco_overlay
                                            and _is_coco
                                            and _chunk_bboxes is not None
                                            and _chunk_cats is not None
                                            and k < len(_chunk_bboxes)
                                            and k < len(_chunk_cats)
                                        ):
                                            _pil = pil_overlay_coco_bboxes(
                                                _pil,
                                                _chunk_bboxes[k],
                                                _chunk_cats[k],
                                                _coco_cat_map,
                                            )
                                        _thumb = (
                                            pil_square_thumbnail(_pil, 180)
                                            if _pil is not None else None
                                        )
                                        if _thumb is not None:
                                            st.image(_thumb)
                                        else:
                                            st.caption("_(decode failed)_")
                                        st.caption(f"**#{lo + k}**")
                                        if _pil is not None and _dm_fullres_dialog is not None:
                                            _key = f"qs_view_fs_{lo + k}"
                                            if st.button(
                                                "Full size", key=_key,
                                                help="Open at native resolution",
                                            ):
                                                st.session_state["dm_fullres_pil"] = _pil.copy()
                                                st.session_state["dm_fullres_caption"] = (
                                                    f"Sample index **{lo + k}** (full resolution)"
                                                )
                                                _dm_fullres_dialog()

            # Tabular preview
            if n_results > 0:
                with st.expander("📊 Table preview", expanded=(not bool(img_tensors_r))):
                    df, derr = dataset_to_dataframe(result_ds, end=min(n_results, 200))
                    if derr:
                        st.error(derr)
                    elif df is not None:
                        st.dataframe(dataframe_for_streamlit_display(df), width="stretch")

            # Save as view
            with st.expander("💾 Save result as view", expanded=False):
                # Note on semantics: even if the active result came from a
                # read-only borrow of another engineer's view, saving it here
                # creates a NEW view owned by the current user (``uid`` comes
                # from ``obtain_current_user()`` inside MULLER's save_view).
                # This is the collaboration handoff: "I reuse Alice's query
                # result as my starting point, and snapshot it as my own view."
                if result_meta.get("source") == "view" and not result_meta.get("is_mine"):
                    st.caption(
                        f"ℹ️ This result was loaded from `{result_meta.get('owner', '-')}`'s view. "
                        "Saving here creates a **new** view owned by "
                        f"`{current_user()}`."
                    )
                _default_vid = ""
                if result_meta.get("source") == "view" and result_meta.get("is_mine"):
                    # Only pre-fill when the active view is already ours —
                    # otherwise we'd invite the user to stomp on a foreign id.
                    _default_vid = str(result_meta.get("view_id", ""))
                new_vid = st.text_input(
                    "View ID", value=_default_vid, key="qs_save_vid",
                    help="Required. Letters/numbers/underscore recommended; "
                    "must be unique within this dataset.",
                )
                # Tie the widget's key to a short hash of ``result_desc`` so
                # that when the user runs a new query (producing a different
                # description like "Conditional filter → 59 matching samples")
                # Streamlit treats this as a *new* text_input and picks up the
                # new default. A stable ``key="qs_save_msg"`` would stay pinned
                # to the first query's value forever — Streamlit's ``value=``
                # is only consulted on the widget's first render.
                _rd = result_desc or ""
                _msg_key = "qs_save_msg::" + hashlib.md5(
                    _rd.encode("utf-8")
                ).hexdigest()[:12]
                new_msg = st.text_input(
                    "Message (optional)", value=_rd,
                    key=_msg_key,
                )
                if st.button("Save as View", type="primary", key="qs_save_btn"):
                    vid_clean = new_vid.strip()
                    if not vid_clean:
                        st.error("Please enter a non-empty view ID.")
                    else:
                        path, serr = save_query_view(
                            result_ds, view_id=vid_clean, message=new_msg or None
                        )
                        if serr:
                            st.error(serr)
                        else:
                            st.success(
                                f"Saved view `{vid_clean}` "
                                f"({n_results} samples). Stored at: `{path}`"
                            )
                            st.rerun()


# ============================================================================
# PAGE 3: Version Control
# ============================================================================
elif page == "🌿 Version Control":
    st.title("🌿 Version Control")

    if st.session_state.dataset is None:
        st.warning("Please create or load a dataset first.")
    else:
        ds = st.session_state.dataset

        # --- Branches ---
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
            _graph_h = 22 + _graph_data["lane_count"] * 46 + 20
            _graph_html = render_commit_graph_html(_graph_data, height=_graph_h)
            components.html(_graph_html, height=_graph_h, scrolling=False)
        except Exception:
            # Fallback to markdown
            st.markdown("**Current branches:**")
            for bname in branch_names:
                if bname == ds.branch:
                    st.markdown(f"- **{bname}** ← current")
                else:
                    st.markdown(f"- {bname}")

        col_create, col_switch, col_delete = st.columns(3)

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

        with col_delete:
            st.markdown("**Delete Branch**")
            # Pre-compute which branches are actually deletable (mirrors backend rules:
            # not `main`, not current, no sub-branches, never merged into another branch).
            deletable_map = classify_deletable_branches(ds)
            deletable = [b for b in branch_names if deletable_map.get(b) is None]
            blocked = [(b, deletable_map.get(b)) for b in branch_names
                       if deletable_map.get(b) is not None]

            if not deletable:
                st.selectbox("Branch to delete", ["(none)"],
                             key="delete_branch_none", disabled=True)
                st.caption("No deletable branches.")
            else:
                del_target = st.selectbox("Branch to delete", deletable,
                                          key="delete_branch_target")
                confirm_del = st.checkbox(f"Confirm delete `{del_target}`",
                                          key="confirm_delete_branch")
                if st.button("Delete Branch"):
                    if not confirm_del:
                        st.warning("Please tick the confirmation box before deleting.")
                    else:
                        res, err = branch_ops(ds, "delete", branch_name=del_target)
                        if err:
                            st.error(err)
                        else:
                            st.success(res)
                            st.rerun()

            if blocked:
                with st.expander(f"Non-deletable branches ({len(blocked)})", expanded=False):
                    for bname, reason in blocked:
                        st.markdown(f"- `{bname}` — _{reason}_")

        # --- Merge & Conflicts ---
        st.markdown("---")
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
                    # Check tensor-level conflicts (renames/deletes)
                    has_tensor_conflicts = bool(result["columns"])

                    # Check sample-level conflicts across ALL common tensors.
                    # MULLER's `detect_merge_conflict` returns *differences vs. LCA*
                    # in `del_ori_idx / del_tar_idx / app_ori_idx / app_tar_idx`,
                    # NOT conflicts. A real conflict requires divergence on the
                    # same uuid:
                    #   - append: both branches added new uuids → both append-lists
                    #     non-empty (new uuids are always disjoint, so any double
                    #     append needs an `append_resolution`).
                    #   - update: MULLER pre-filters `update_values` to only carry
                    #     true update-vs-update (or update-vs-pop) divergence, so
                    #     "either side non-empty" is the right test here.
                    #   - delete: only a conflict when both sides popped the SAME
                    #     LCA index (`pop_resolution` is needed). One-sided deletes
                    #     merge cleanly without a strategy.
                    tensors_with_sample_conflicts = []
                    for col_name, cdata in result.get("records", {}).items():
                        has_append = bool(cdata.get("app_ori_idx") and cdata.get("app_tar_idx"))
                        has_update = False
                        if cdata.get("update_values"):
                            uv = cdata["update_values"]
                            has_update = bool(uv.get("update_ori") or uv.get("update_tar"))
                        ori_del = set(cdata.get("del_ori_idx") or [])
                        tar_del = set(cdata.get("del_tar_idx") or [])
                        has_delete = bool(ori_del & tar_del)
                        if has_append or has_update or has_delete:
                            tensors_with_sample_conflicts.append(col_name)

                    if has_tensor_conflicts or tensors_with_sample_conflicts:
                        conflict_summary = []
                        if has_tensor_conflicts:
                            conflict_summary.append(f"Tensor conflicts: {', '.join(result['columns'])}")
                        if tensors_with_sample_conflicts:
                            conflict_summary.append(f"Sample conflicts in: {', '.join(tensors_with_sample_conflicts)}")
                        st.warning(" | ".join(conflict_summary))

                        # Show details for all tensors that have any conflict
                        all_conflict_tensors = set(tensors_with_sample_conflicts)
                        if has_tensor_conflicts:
                            all_conflict_tensors.update(result["columns"])

                        with st.expander("Conflict Details", expanded=True):
                            for col_name in sorted(all_conflict_tensors):
                                st.markdown(f"#### `{col_name}`")
                                cdata = result["records"].get(col_name, {})

                                # Append conflicts (both branches appended)
                                if cdata.get("app_ori_idx") and cdata.get("app_tar_idx"):
                                    st.markdown("**Append conflicts (both branches added samples):**")
                                    rows = []
                                    ori_vals = cdata.get("app_ori_values", [])
                                    tar_vals = cdata.get("app_tar_values", [])
                                    for j, idx in enumerate(cdata["app_ori_idx"]):
                                        rows.append({"Side": "Current (ours)", "Index": idx,
                                                     "Value": str(ori_vals[j]) if j < len(ori_vals) else "—"})
                                    for j, idx in enumerate(cdata["app_tar_idx"]):
                                        rows.append({"Side": "Source (theirs)", "Index": idx,
                                                     "Value": str(tar_vals[j]) if j < len(tar_vals) else "—"})
                                    if rows:
                                        st.dataframe(pd.DataFrame(rows), width="stretch")
                                elif cdata.get("app_ori_idx"):
                                    st.markdown(f"**Appended in current only:** {len(cdata['app_ori_idx'])} samples")
                                elif cdata.get("app_tar_idx"):
                                    st.markdown(f"**Appended in source only:** {len(cdata['app_tar_idx'])} samples")

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
