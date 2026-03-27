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
    run_query, dataset_to_dataframe, dataframe_for_streamlit_display,
    list_image_tensor_names, decode_muller_image_sample, pil_resize_to_height,
    pil_square_thumbnail, pil_fit_inside,
    pil_preview_available,
    is_coco2017_muller_schema, load_coco_category_id_to_name,
    pil_overlay_coco_bboxes, DEFAULT_COCO_INSTANCES_JSON,
    branch_ops, benchmark_parquet_vs_muller,
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
        ds, err = load_dataset(st.session_state.dataset_path)
        if err is None:
            st.session_state.dataset = ds


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
            _default_create_path = str(Path.home() / "muller_datasets" / "demo_dataset")
            ds_path_input = st.text_input(
                "Dataset Path",
                value=_default_create_path,
                key="create_dataset_path",
                help="Full path where the dataset folder will be created (parent directory must exist or be creatable).",
            )
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
                                "COCO2017-style layout detected (8 columns: area, bbox, category_id, id, "
                                "image_id, images, iscrowd, segmentation)."
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
                        st.error("Please fill in at least one field.")
                    else:
                        err = add_samples(ds, filtered, auto_commit=True,
                                          commit_message=add_commit_msg)
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

                    csv_commit_msg = st.text_input(
                        "Commit Message", value="Import CSV data via Streamlit UI",
                        key="csv_commit_msg")
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
                                commit_dataset(ds, message=csv_commit_msg)
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
                    del_commit_msg = st.text_input(
                        "Commit Message", value="Delete sample via Streamlit UI",
                        key="del_commit_msg")
                    if st.button("Delete", type="secondary"):
                        err = delete_sample(ds, del_idx)
                        if err:
                            st.error(err)
                        else:
                            commit_dataset(ds, message=del_commit_msg)
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
                    upd_commit_msg = st.text_input(
                        "Commit Message", value="Update sample via Streamlit UI",
                        key="upd_commit_msg")
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
                            commit_dataset(ds, message=upd_commit_msg)
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
                            st.dataframe(dataframe_for_streamlit_display(df), width="stretch")

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
                        # Check tensor-level conflicts (renames/deletes)
                        has_tensor_conflicts = bool(result["columns"])

                        # Check sample-level conflicts across ALL common tensors
                        tensors_with_sample_conflicts = []
                        for col_name, cdata in result.get("records", {}).items():
                            has_append = bool(cdata.get("app_ori_idx") and cdata.get("app_tar_idx"))
                            has_update = False
                            if cdata.get("update_values"):
                                uv = cdata["update_values"]
                                has_update = bool(uv.get("update_ori") or uv.get("update_tar"))
                            has_delete = bool(cdata.get("del_ori_idx") or cdata.get("del_tar_idx"))
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
