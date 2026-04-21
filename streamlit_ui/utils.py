# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""
Utility functions for Streamlit demo - wraps MULLER API calls with error handling.
"""
import muller
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import time
import os
import io
import plotly.graph_objects as go

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
except ImportError:
    PILImage = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

# COCO2017 layout produced by official_demo.ipynb (muller_datasetcoco-style).
COCO2017_MULLER_SCHEMA = frozenset(
    {
        "area",
        "bbox",
        "category_id",
        "id",
        "image_id",
        "images",
        "iscrowd",
        "segmentation",
    }
)
DEFAULT_COCO_INSTANCES_JSON = (
    "/Users/sherrylin/Documents/research_data/coco2017/annotations/instances_val2017.json"
)

# Matplotlib tab20 base colors (RGB 0–255) for box / label styling.
_TAB20_RGB: List[Tuple[int, int, int]] = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
]


def pil_preview_available() -> bool:
    return PILImage is not None


def _is_image_htype(ht: Any) -> bool:
    if ht is None:
        return False
    h = str(ht).lower()
    if h in ("image", "image.rgb", "image.gray", "dicom", "nifti"):
        return True
    if h.startswith("image."):
        return True
    if h.startswith("link[") and "image" in h:
        return True
    return False


def list_image_tensor_names(ds: Any) -> List[str]:
    """Tensor names whose htype stores image samples (for Streamlit preview)."""
    return [n for n, t in ds.tensors.items() if _is_image_htype(t.htype)]


def decode_muller_image_sample(v: Any) -> Optional[Any]:
    """Decode one image sample to a PIL Image, or None if unsupported."""
    if PILImage is None or v is None:
        return None
    if not isinstance(v, np.ndarray):
        try:
            if pd.isna(v) and not isinstance(v, (str, bytes)):
                return None
        except (TypeError, ValueError):
            pass
    try:
        img = None
        if isinstance(v, bytes):
            img = PILImage.open(io.BytesIO(v))
        elif isinstance(v, np.ndarray):
            if v.size == 0 or v.dtype == object:
                return None
            if v.ndim == 1 and v.dtype == np.uint8:
                try:
                    img = PILImage.open(io.BytesIO(v.tobytes()))
                except Exception:
                    return None
            elif v.ndim >= 2:
                arr = v
                if arr.dtype != np.uint8:
                    if np.issubdtype(arr.dtype, np.floating):
                        arr = np.clip(arr, 0, 1)
                        arr = (arr * 255).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8, copy=False)
                img = PILImage.fromarray(arr)
        if img is None:
            return None
        return img
    except Exception:
        return None


def pil_resize_to_height(img: Any, target_h: int) -> Optional[Any]:
    """Resize to fixed height; width scales with aspect ratio."""
    if img is None or PILImage is None or target_h <= 0:
        return None
    try:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        w, h = img.size
        if h <= 0:
            return None
        new_w = max(1, int(round(w * target_h / float(h))))
        try:
            resample = PILImage.Resampling.LANCZOS
        except AttributeError:
            resample = PILImage.LANCZOS  # type: ignore[attr-defined]
        return img.resize((new_w, target_h), resample)
    except Exception:
        return None


def pil_square_thumbnail(img: Any, edge: int) -> Optional[Any]:
    """Center-crop to a square, then resize to edge×edge (for uniform thumbnail grids)."""
    if img is None or PILImage is None or edge <= 0:
        return None
    try:
        im = img if img.mode in ("RGB", "RGBA") else img.convert("RGB")
        if im.mode == "RGBA":
            im = im.convert("RGB")
        w, h = im.size
        if w <= 0 or h <= 0:
            return None
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        im = im.crop((left, top, left + s, top + s))
        try:
            resample = PILImage.Resampling.LANCZOS
        except AttributeError:
            resample = PILImage.LANCZOS  # type: ignore[attr-defined]
        return im.resize((edge, edge), resample)
    except Exception:
        return None


def pil_fit_inside(
    img: Any,
    max_w: int,
    max_h: int,
    pad_color: Tuple[int, int, int] = (238, 238, 242),
) -> Optional[Any]:
    """Uniform scale to fit inside max_w×max_h (no upscaling); letterbox on pad_color."""
    if img is None or PILImage is None or max_w <= 0 or max_h <= 0:
        return None
    try:
        im = img if img.mode in ("RGB", "RGBA") else img.convert("RGB")
        if im.mode == "RGBA":
            im = im.convert("RGB")
        w, h = im.size
        if w <= 0 or h <= 0:
            return None
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        try:
            resample = PILImage.Resampling.LANCZOS
        except AttributeError:
            resample = PILImage.LANCZOS  # type: ignore[attr-defined]
        im2 = im.resize((nw, nh), resample)
        canvas = PILImage.new("RGB", (max_w, max_h), pad_color)
        ox = (max_w - nw) // 2
        oy = (max_h - nh) // 2
        canvas.paste(im2, (ox, oy))
        return canvas
    except Exception:
        return None


def is_coco2017_muller_schema(ds: Any) -> bool:
    """True if public tensors match the 8-column COCO2017 MULLER layout (ignores _uuid)."""
    names = set(ds.tensors.keys())
    names.discard("_uuid")
    return names == COCO2017_MULLER_SCHEMA


def load_coco_category_id_to_name(instances_json: str) -> Tuple[Optional[Dict[int, str]], Optional[str]]:
    """Load COCO category id → name from instances JSON (pycocotools if installed, else JSON)."""
    path = Path(instances_json).expanduser()
    if not path.is_file():
        return None, f"COCO annotations file not found: {path}"
    try:
        from pycocotools.coco import COCO  # type: ignore

        coco = COCO(str(path))
        return {int(k): v["name"] for k, v in coco.cats.items()}, None
    except ImportError:
        pass
    except Exception as e:
        return None, f"pycocotools could not load annotations: {e}"
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cats = data.get("categories") or []
        m = {int(c["id"]): str(c["name"]) for c in cats if "id" in c and "name" in c}
        if not m:
            return None, "No categories found in JSON"
        return m, None
    except Exception as e:
        return None, str(e)


def _unwrap_muller_aslist_sample(x: Any) -> Any:
    """MULLER often returns ``aslist=True`` as a one-element list wrapping the real ndarray."""
    if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], np.ndarray):
        return x[0]
    return x


def _normalize_bboxes_for_overlay(b: Any) -> np.ndarray:
    b = _unwrap_muller_aslist_sample(b)
    a = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if a.ndim == 1 and a.shape[0] == 4:
        a = a.reshape(1, 4)
    if a.ndim != 2 or a.shape[1] != 4:
        return np.zeros((0, 4), dtype=np.float64)
    return a


def _normalize_category_ids_for_overlay(c: Any, n: int) -> np.ndarray:
    c = _unwrap_muller_aslist_sample(c)
    a = np.asarray(c)
    if a.ndim == 0:
        a = np.array([a.item()], dtype=np.int64)
    else:
        a = a.reshape(-1).astype(np.int64, copy=False)
    if a.shape[0] != n:
        if a.shape[0] == 0 and n > 0:
            return np.zeros(n, dtype=np.int64)
        m = min(a.shape[0], n)
        out = np.zeros(n, dtype=np.int64)
        out[:m] = a[:m]
        return out
    return a


def pil_overlay_coco_bboxes(
    img: Any,
    bbox_sample: Any,
    category_id_sample: Any,
    cat_id_to_name: Optional[Dict[int, str]] = None,
) -> Any:
    """Draw COCO-style [x, y, w, h] boxes and labels on a PIL image (mutates a copy)."""
    if PILImage is None or ImageDraw is None or img is None:
        return img
    _names = cat_id_to_name or {}
    try:
        out = img.copy()
        if out.mode not in ("RGB", "RGBA"):
            out = out.convert("RGB")
        else:
            out = out.convert("RGB")
        draw = ImageDraw.Draw(out)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        bboxes = _normalize_bboxes_for_overlay(bbox_sample)
        n = int(bboxes.shape[0])
        cats = _normalize_category_ids_for_overlay(category_id_sample, n)
        for i in range(n):
            x, y, w, h = (float(bboxes[i, j]) for j in range(4))
            if w <= 1e-6 or h <= 1e-6:
                continue
            x0, y0 = int(round(x)), int(round(y))
            x1, y1 = int(round(x + w)), int(round(y + h))
            rgb = _TAB20_RGB[i % len(_TAB20_RGB)]
            draw.rectangle([x0, y0, x1, y1], outline=rgb, width=max(2, int(round(min(out.size) / 400))))
            cid = int(cats[i]) if i < len(cats) else 0
            label = _names.get(cid, str(cid))
            ty = max(0, y0 - 12)
            if font is not None:
                _bb = draw.textbbox((0, 0), label, font=font)
                tw, th = _bb[2] - _bb[0], _bb[3] - _bb[1]
            else:
                tw, th = (len(label) * 6, 11)
            draw.rectangle([x0, ty, x0 + tw + 4, ty + th + 2], fill=rgb)
            draw.text((x0 + 2, ty + 1), label, fill=(255, 255, 255), font=font)
        return out
    except Exception:
        return img


def create_dataset(name: str, root: str, overwrite: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """Create a new MULLER dataset."""
    try:
        ds_path = Path(root) / name
        ds = muller.dataset(path=str(ds_path), overwrite=overwrite)
        return ds, None
    except Exception as e:
        return None, f"Failed to create dataset: {e}"


def create_tensors(ds: Any, schema: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Create tensors (columns) in the dataset.

    schema: {tensor_name: {htype, dtype, sample_compression}}
    """
    try:
        for tensor_name, config in schema.items():
            kwargs = {"htype": config.get("htype", "generic")}
            if config.get("dtype"):
                kwargs["dtype"] = config["dtype"]
            if config.get("sample_compression"):
                kwargs["sample_compression"] = config["sample_compression"]
            ds.create_tensor(tensor_name, **kwargs)
        return None
    except Exception as e:
        return f"Failed to create tensors: {e}"


def add_samples(
    ds: Any,
    data: Dict[str, List],
    auto_commit: bool = True,
    commit_message: Optional[str] = None,
) -> Optional[str]:
    """Add samples to dataset using per-tensor extend."""
    try:
        with ds:
            for tensor_name, values in data.items():
                ds[tensor_name].extend(values)
        if auto_commit:
            ds.commit(message=commit_message or "Add samples via Streamlit UI")
        return None
    except Exception as e:
        return f"Failed to add samples: {e}"


def update_sample(ds: Any, tensor_name: str, index: int, value: Any) -> Optional[str]:
    """Update a single sample value."""
    try:
        ds[tensor_name][index] = value
        return None
    except Exception as e:
        return f"Failed to update sample: {e}"


def delete_sample(ds: Any, index: int) -> Optional[str]:
    """Delete a sample by index."""
    try:
        ds.pop(index)
        return None
    except Exception as e:
        return f"Failed to delete sample: {e}"


def run_query(ds: Any, conditions: List[Tuple[str, str, Any]],
              connectors: Optional[List[str]] = None,
              offset: int = 0, limit: Optional[int] = None) -> Tuple[Optional[Any], Optional[str]]:
    """Run a filter query on the dataset."""
    try:
        kwargs = {"offset": offset, "limit": limit}
        if connectors:
            kwargs["connector_list"] = connectors
        result = ds.filter_vectorized(conditions, **kwargs)
        return result, None
    except Exception as e:
        return None, f"Query failed: {e}"


def dataset_to_dataframe(ds: Any, tensor_list: Optional[List[str]] = None,
                         start: int = 0, end: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Convert dataset (or view) to pandas DataFrame.

    Uses the sliced *view* for the fallback path so only ``end - start`` rows are
    materialized (avoids loading entire tensors for pagination).
    """
    try:
        if end is not None:
            view = ds[start:end]
        elif start > 0:
            view = ds[start:]
        else:
            view = ds
        df = view.to_dataframe(tensor_list=tensor_list)
        return df, None
    except Exception:
        try:
            if end is not None:
                view = ds[start:end]
            elif start > 0:
                view = ds[start:]
            else:
                view = ds
            if tensor_list is None:
                tensor_list = list(view.tensors.keys())
            data = {}
            for tname in tensor_list:
                tensor = view[tname]
                vals = tensor.numpy(aslist=True)
                data[tname] = vals
            return pd.DataFrame(data), None
        except Exception as e2:
            return None, f"Failed to convert to DataFrame: {e2}"


def dataframe_for_streamlit_display(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe for ``st.dataframe`` (PyArrow round-trip).

    Object columns with nested lists, multi-dimensional arrays, or other
    non-scalar cells (e.g. COCO ``bbox``) otherwise raise ArrowInvalid.
    """
    if df is None or df.empty:
        return df

    def _cell(v: Any) -> Any:
        if v is None:
            return None
        try:
            if pd.isna(v) and not isinstance(v, (str, bytes)):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(v, (str, bytes)):
            return v
        if isinstance(v, (bool, int, float, np.integer, np.floating, np.bool_)):
            return v
        if isinstance(v, np.ndarray):
            if v.ndim == 0 and v.dtype != object:
                try:
                    return v.item()
                except Exception:
                    return str(v)
            return str(v)
        if isinstance(v, (list, tuple, dict, set)):
            return str(v)
        return str(v)

    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(_cell)
    return out


def branch_ops(ds: Any, action: str, branch_name: Optional[str] = None,
               merge_strategy: Optional[Dict[str, str]] = None) -> Tuple[Optional[Any], Optional[str]]:
    """Perform version control operations."""
    try:
        if action == "create":
            ds.checkout(branch_name, create=True)
            return f"Branch '{branch_name}' created and checked out", None

        elif action == "checkout":
            ds.checkout(branch_name)
            return f"Switched to branch '{branch_name}'", None

        elif action == "merge":
            strategy = merge_strategy or {}
            ds.merge(
                branch_name,
                append_resolution=strategy.get("append_resolution", "ours"),
                pop_resolution=strategy.get("pop_resolution", "ours"),
                update_resolution=strategy.get("update_resolution", "ours"),
            )
            return f"Merged '{branch_name}' into current branch", None

        elif action == "detect_conflict":
            conflict_cols, conflict_records = ds.detect_merge_conflict(branch_name, show_value=True)
            return {"columns": conflict_cols, "records": conflict_records}, None

        elif action == "list":
            return ds.branches, None

        elif action == "commit":
            cid = ds.commit(message=merge_strategy.get("message", "Commit from UI") if merge_strategy else "Commit from UI")
            return f"Committed: {cid}", None

        elif action == "log":
            return ds.commits(ordered_by_date=True), None

        elif action == "diff":
            diff = ds.diff(as_dict=True, show_value=True)
            return diff, None

        else:
            return None, f"Unknown action: {action}"

    except Exception as e:
        return None, f"Branch operation failed: {e}"


def build_commit_graph_data(ds: Any) -> Dict[str, Any]:
    """Extract the full commit DAG from dataset version state for visualization.

    Returns a JSON-serializable dict with commits, branches, and lane assignments.
    """
    from muller.constants import FIRST_COMMIT_ID
    from collections import deque

    commit_node_map = ds.version_state.get("commit_node_map", {})
    branch_commit_map = ds.version_state.get("branch_commit_map", {})
    branch_info = ds.version_state.get("branch_info", {})
    current_branch = ds.version_state.get("branch", "main")
    current_node = ds.version_state.get("commit_node")
    current_commit_id = current_node.commit_id if current_node else ""

    # Assign lanes: main=0, others sorted by create_time
    all_branch_names = set(branch_commit_map.keys())
    for node in commit_node_map.values():
        all_branch_names.add(node.branch)

    def _branch_sort_key(name):
        if name == "main":
            return (0, "")
        info = branch_info.get(name, {})
        ct = info.get("create_time")
        if ct is None:
            return (2, name)
        if hasattr(ct, "timestamp"):
            return (1, ct.timestamp())
        return (1, str(ct))

    sorted_branches = sorted(all_branch_names, key=_branch_sort_key)
    branch_lanes = {name: i for i, name in enumerate(sorted_branches)}

    # BFS to collect all reachable nodes
    root = commit_node_map.get(FIRST_COMMIT_ID)
    if not root:
        return {"commits": [], "branches": [], "lane_count": 0}

    all_nodes = {}
    visited = set()
    queue = deque([root])
    visited.add(root.commit_id)
    while queue:
        node = queue.popleft()
        all_nodes[node.commit_id] = node
        for child in node.children:
            if child.commit_id not in visited:
                visited.add(child.commit_id)
                queue.append(child)

    # Assign depth by commit_time order so that left-to-right = chronological.
    # Nodes with commit_time are sorted by time; uncommitted heads go to the end.
    committed = [n for n in all_nodes.values() if n.commit_time is not None]
    uncommitted = [n for n in all_nodes.values() if n.commit_time is None]
    committed.sort(key=lambda n: n.commit_time)
    uncommitted.sort(key=lambda n: branch_lanes.get(n.branch, 0))

    depth = {}
    d = 0
    prev_time = None
    for n in committed:
        if prev_time is not None and n.commit_time != prev_time:
            d += 1
        depth[n.commit_id] = d
        prev_time = n.commit_time
    # Uncommitted heads each get their own depth at the end
    if uncommitted:
        d += 1
        for n in uncommitted:
            depth[n.commit_id] = d

    sorted_nodes = sorted(all_nodes.values(),
                          key=lambda n: (depth[n.commit_id], branch_lanes.get(n.branch, 0)))
    topo_order = list(reversed(sorted_nodes))
    max_depth = max(depth.values()) if depth else 0

    # Build commit records
    commits = []
    for row_idx, node in enumerate(topo_order):
        lane = branch_lanes.get(node.branch, 0)
        commits.append({
            "id": node.commit_id,
            "short_id": node.commit_id[:8],
            "branch": node.branch,
            "lane": lane,
            "depth": depth[node.commit_id],
            "message": node.commit_message or "",
            "time": str(node.commit_time)[:-7] if node.commit_time else "",
            "author": node.commit_user_name or "",
            "parent_id": node.parent.commit_id if node.parent else None,
            "merge_parent_id": node.merge_parent if node.merge_parent else None,
            "is_merge": node.is_merge_node,
            "is_head": node.is_head_node,
            "is_checkout": getattr(node, "checkout_node", False),
            "is_current": (node.commit_id == current_commit_id),
            "row": row_idx,
        })

    branches_list = []
    for name in sorted_branches:
        branches_list.append({
            "name": name,
            "lane": branch_lanes[name],
            "is_current": name == current_branch,
            "head_commit_id": branch_commit_map.get(name, ""),
        })

    return {
        "commits": commits,
        "branches": branches_list,
        "lane_count": len(branch_lanes),
        "max_depth": max_depth,
    }


def render_commit_graph_html(graph_data: Dict[str, Any], height: int = 500) -> str:
    """Render the commit graph as self-contained horizontal HTML with SVG + vanilla JS."""
    import json
    json_data = json.dumps(graph_data)

    return f'''<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: "SF Mono", "Cascadia Code", "Fira Code", Menlo, monospace; background: #fafbfc; }}
  .graph-wrap {{ position: relative; overflow-x: auto; overflow-y: auto; max-height: {height}px; padding: 0; }}
  .tooltip {{
    position: absolute; display: none; background: #24292e; color: #e1e4e8;
    padding: 8px 12px; border-radius: 6px; font-size: 11.5px; line-height: 1.6;
    pointer-events: none; z-index: 100; white-space: pre-line; max-width: 300px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
  }}
  .tooltip b {{ color: #79b8ff; }}
  .tooltip .t-branch {{ color: #85e89d; }}
  .tooltip .t-msg {{ color: #d1d5da; }}
  .tooltip .t-meta {{ color: #959da5; font-size: 10.5px; }}
  .tooltip .t-tag {{ display: inline-block; padding: 0 5px; border-radius: 3px; font-size: 10px;
    margin-left: 4px; font-weight: 600; }}
  .tooltip .t-merge {{ background: #b3920020; color: #ffdf5d; border: 1px solid #ffdf5d50; }}
  .tooltip .t-cur   {{ background: #34d05820; color: #85e89d; border: 1px solid #85e89d50; }}
</style>
</head>
<body>
<div class="graph-wrap" id="wrap">
  <svg id="graph" xmlns="http://www.w3.org/2000/svg"></svg>
  <div class="tooltip" id="tip"></div>
</div>
<script>
(function() {{
  const D = {json_data};
  if (!D.commits || !D.commits.length) {{
    document.getElementById("wrap").innerHTML = '<p style="padding:16px;color:#8b949e;font-size:13px;">No commits yet.</p>';
    return;
  }}

  /* ── Layout constants ── */
  const COL  = 70;          /* horizontal spacing between depth levels */
  const LANE = 46;          /* vertical spacing between branch lanes */
  const R    = 5;           /* node radius */
  const LM   = 90;          /* left margin for branch labels */
  const TM   = 22;          /* top margin */

  /* ── Palette: muted, professional tones ── */
  const PAL = [
    "#2ea043",  /* green  – main */
    "#388bfd",  /* blue   */
    "#d29922",  /* amber  */
    "#a371f7",  /* purple */
    "#f85149",  /* red    */
    "#3fb950",  /* lime   */
    "#56d4dd",  /* cyan   */
    "#db6d28",  /* orange */
    "#e275ad",  /* pink   */
    "#768390",  /* gray   */
  ];
  function clr(lane) {{ return PAL[lane % PAL.length]; }}

  /* ── SVG helpers ── */
  const NS = "http://www.w3.org/2000/svg";
  function el(tag, a) {{
    const e = document.createElementNS(NS, tag);
    for (const k in a) e.setAttribute(k, a[k]);
    return e;
  }}
  function g() {{ return document.createElementNS(NS, "g"); }}

  const svg = document.getElementById("graph");
  const tip = document.getElementById("tip");

  /* ── Compute node positions ── */
  const pos = {{}};
  D.commits.forEach(c => {{
    pos[c.id] = {{ x: LM + c.depth * COL, y: TM + c.lane * LANE }};
  }});

  /* ── Nudge nodes that sit on cross-lane bezier edges ──
   * For bezier M sx,sy C mx,sy mx,dy dx,dy the y-parametric is:
   *   y(t) = sy·(1-t)²·(1+2t) + dy·t²·(3-2t)
   * We solve for t at each intermediate lane's y via bisection,
   * then check if any unrelated node's x is within OV_THRESH of the
   * curve's x(t). If so, shift that node right by NUDGE_PX.
   */
  const NUDGE_PX = Math.round(COL * 0.22);
  const OV_THRESH = R + 4;
  function _solveT(sy, dy, tgt) {{
    let lo = 0, hi = 1;
    for (let i = 0; i < 25; i++) {{
      const t = (lo + hi) / 2, u = 1 - t;
      const yt = sy * u * u * (1 + 2 * t) + dy * t * t * (3 - 2 * t);
      if ((dy > sy) === (yt < tgt)) lo = t; else hi = t;
    }}
    return (lo + hi) / 2;
  }}
  function _bezX(t, sx, mx, dx) {{
    const u = 1 - t;
    return u * u * u * sx + 3 * u * t * mx + t * t * t * dx;
  }}
  const nudged = new Set();
  D.commits.forEach(c => {{
    ["parent_id", "merge_parent_id"].forEach(key => {{
      const pid = c[key];
      if (!pid || !pos[pid]) return;
      const s = pos[pid], d = pos[c.id];
      if (s.y === d.y) return;
      const emx = (s.x + d.x) / 2;
      const minY = Math.min(s.y, d.y), maxY = Math.max(s.y, d.y);
      D.commits.forEach(o => {{
        if (o.id === pid || o.id === c.id || nudged.has(o.id)) return;
        const ny = pos[o.id].y;
        if (ny <= minY || ny >= maxY) return;
        const t = _solveT(s.y, d.y, ny);
        const bx = _bezX(t, s.x, emx, d.x);
        if (Math.abs(pos[o.id].x - bx) < OV_THRESH) {{
          pos[o.id].x += NUDGE_PX;
          nudged.add(o.id);
        }}
      }});
    }});
  }});

  const maxD = D.max_depth || 0;
  const W = LM + (maxD + 1) * COL + 40 + (nudged.size > 0 ? NUDGE_PX : 0);
  const H = TM + D.lane_count * LANE + 24;
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);

  /* ── Layer 1: branch lane rails (subtle dashed lines) ── */
  D.branches.forEach(b => {{
    const y = TM + b.lane * LANE;
    svg.appendChild(el("line", {{
      x1: LM - 4, y1: y, x2: W - 16, y2: y,
      stroke: clr(b.lane), "stroke-width": "1", "stroke-dasharray": "2,4", opacity: "0.18"
    }}));
  }});

  /* ── Layer 2: branch labels ── */
  D.branches.forEach(b => {{
    const y = TM + b.lane * LANE;
    const isCur = b.is_current;
    /* Rounded-rect badge */
    const label = b.name;
    const charW = isCur ? 7.4 : 6.8;
    const pw = label.length * charW + 14;
    const ph = 18;
    const bx = LM - 8 - pw;
    const bg = g();
    bg.appendChild(el("rect", {{
      x: bx, y: y - ph / 2, width: pw, height: ph, rx: "9", ry: "9",
      fill: isCur ? clr(b.lane) : "#f6f8fa",
      stroke: clr(b.lane), "stroke-width": isCur ? "0" : "1",
      opacity: isCur ? "1" : "0.7"
    }}));
    const txt = el("text", {{
      x: bx + pw / 2, y: y + 4, "text-anchor": "middle",
      fill: isCur ? "#ffffff" : clr(b.lane),
      "font-size": "11px", "font-weight": isCur ? "700" : "500",
      "font-family": "'SF Mono', Menlo, monospace"
    }});
    txt.textContent = label;
    bg.appendChild(txt);
    svg.appendChild(bg);
  }});

  /* ── Arrow head helper (two-line style) ── */
  function arrow(tx, ty, dx, dy, color, op) {{
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux = dx / len, uy = dy / len;
    const px = -uy, py = ux;
    const aL = 6, aW = 3;
    const bx = tx - ux * aL, by = ty - uy * aL;
    svg.appendChild(el("line", {{
      x1: tx, y1: ty, x2: bx + px * aW, y2: by + py * aW,
      stroke: color, "stroke-width": "1.5", opacity: op, "stroke-linecap": "round"
    }}));
    svg.appendChild(el("line", {{
      x1: tx, y1: ty, x2: bx - px * aW, y2: by - py * aW,
      stroke: color, "stroke-width": "1.5", opacity: op, "stroke-linecap": "round"
    }}));
  }}

  /* ── Layer 3: edges ── */
  const edgeG = g();
  svg.appendChild(edgeG);
  D.commits.forEach(c => {{
    [["parent_id", false], ["merge_parent_id", true]].forEach(([key, isMergeEdge]) => {{
      const pid = c[key];
      if (!pid || !pos[pid]) return;
      const s = pos[pid], d = pos[c.id];
      const pc = D.commits.find(x => x.id === pid);
      const lane = pc ? pc.lane : c.lane;
      const sc = clr(lane);
      const op = isMergeEdge ? "0.35" : "0.50";
      const sw = isMergeEdge ? "1.5" : "2";
      const dash = isMergeEdge ? "4,3" : "none";

      if (s.y === d.y) {{
        /* Same lane: straight */
        edgeG.appendChild(el("line", {{
          x1: s.x, y1: s.y, x2: d.x, y2: d.y,
          stroke: sc, "stroke-width": sw, opacity: op, "stroke-dasharray": dash
        }}));
        arrow(d.x - R - 1, d.y, 1, 0, sc, op);
      }} else {{
        /* Cross-lane: cubic bezier for smooth S-curve */
        const mx = (s.x + d.x) / 2;
        edgeG.appendChild(el("path", {{
          d: `M ${{s.x}} ${{s.y}} C ${{mx}} ${{s.y}}, ${{mx}} ${{d.y}}, ${{d.x}} ${{d.y}}`,
          stroke: sc, "stroke-width": sw, opacity: op, fill: "none",
          "stroke-dasharray": dash
        }}));
        arrow(d.x - R - 1, d.y, 1, 0, sc, op);
      }}
    }});
  }});

  /* ── Layer 4: commit nodes ── */
  D.commits.forEach(c => {{
    const cx = pos[c.id].x, cy = pos[c.id].y, cc = clr(c.lane);
    const ng = g();
    ng.style.cursor = "pointer";

    /* Current node: animated pulsing ring */
    if (c.is_current) {{
      const pulse = el("circle", {{
        cx, cy, r: R + 4, fill: "none", stroke: cc, "stroke-width": "2", opacity: "0.5"
      }});
      pulse.innerHTML =
        '<animate attributeName="r" values="' + (R + 4) + ';' + (R + 10) + ';' + (R + 4) + '" dur="2s" repeatCount="indefinite"/>' +
        '<animate attributeName="opacity" values="0.5;0.08;0.5" dur="2s" repeatCount="indefinite"/>';
      ng.appendChild(pulse);
    }}

    /* White outline to separate from edges */
    ng.appendChild(el("circle", {{ cx, cy, r: R + 1.5, fill: "#fafbfc" }}));

    if (c.is_merge) {{
      /* Merge node: double ring */
      ng.appendChild(el("circle", {{ cx, cy, r: R, fill: "none", stroke: cc, "stroke-width": "2" }}));
      ng.appendChild(el("circle", {{ cx, cy, r: R - 2.5, fill: cc }}));
    }} else if (c.is_head && !c.time) {{
      /* Uncommitted head: dashed outline */
      ng.appendChild(el("circle", {{
        cx, cy, r: R, fill: "#fafbfc", stroke: cc, "stroke-width": "1.5",
        "stroke-dasharray": "2,2"
      }}));
    }} else {{
      /* Normal commit: solid circle */
      ng.appendChild(el("circle", {{ cx, cy, r: R, fill: cc }}));
    }}

    /* Short hash label below node */
    const label = el("text", {{
      x: cx, y: cy + R + 12, "text-anchor": "middle", fill: "#8b949e",
      "font-size": "9px", "font-family": "'SF Mono', Menlo, monospace"
    }});
    label.textContent = c.short_id;
    ng.appendChild(label);

    /* Tooltip interaction */
    ng.addEventListener("mouseenter", () => {{
      let h = '<b>' + c.short_id + '</b> <span class="t-branch">' + c.branch + '</span>';
      if (c.is_merge) h += '<span class="t-tag t-merge">merge</span>';
      if (c.is_current) h += '<span class="t-tag t-cur">HEAD</span>';
      h += '\\n';
      if (c.message) h += '<span class="t-msg">' + c.message + '</span>\\n';
      else if (c.is_head && !c.time) h += '<span class="t-msg" style="opacity:0.6">(uncommitted)</span>\\n';
      if (c.author || c.time) h += '<span class="t-meta">' + (c.author || '') + (c.time ? ' \u00b7 ' + c.time : '') + '</span>';
      tip.innerHTML = h;
      tip.style.display = "block";
    }});
    ng.addEventListener("mousemove", (e) => {{
      const r = document.getElementById("wrap").getBoundingClientRect();
      tip.style.left = (e.clientX - r.left + 14) + "px";
      tip.style.top  = (e.clientY - r.top  - 8)  + "px";
    }});
    ng.addEventListener("mouseleave", () => {{ tip.style.display = "none"; }});

    svg.appendChild(ng);
  }});
}})();
</script>
</body>
</html>'''


def benchmark_parquet_vs_muller(ds: Any, query_conditions: List[Tuple[str, str, Any]],
                                parquet_path: Optional[str] = None,
                                num_runs: int = 3) -> Tuple[Optional[go.Figure], Optional[str]]:
    """Benchmark query performance: Parquet vs MULLER with multiple runs."""
    try:
        # Export to Parquet if not provided
        if parquet_path is None:
            parquet_path = str(Path(ds.path).parent / "benchmark_temp.parquet")
            ds.write_to_parquet(parquet_path)

        # Benchmark MULLER query (average over num_runs)
        muller_times = []
        for _ in range(num_runs):
            t0 = time.time()
            _ = ds.filter_vectorized(query_conditions)
            muller_times.append(time.time() - t0)
        muller_time = np.mean(muller_times)

        # Benchmark Parquet query via pandas
        import pyarrow.parquet as pq
        parquet_times = []
        for _ in range(num_runs):
            t0 = time.time()
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            for field, op, value in query_conditions:
                if op == "==":
                    df = df[df[field] == value]
                elif op == "!=":
                    df = df[df[field] != value]
                elif op == ">":
                    df = df[df[field] > value]
                elif op == "<":
                    df = df[df[field] < value]
                elif op == ">=":
                    df = df[df[field] >= value]
                elif op == "<=":
                    df = df[df[field] <= value]
            parquet_times.append(time.time() - t0)
        parquet_time = np.mean(parquet_times)

        # File sizes
        muller_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(ds.path)
            for f in files
        ) / (1024 * 1024)
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)

        # Create two-panel Plotly chart
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Query Latency", "Storage Size"))

        fig.add_trace(
            go.Bar(x=["MULLER", "Parquet"], y=[muller_time, parquet_time],
                   marker_color=["#636EFA", "#EF553B"], text=[f"{muller_time:.4f}s", f"{parquet_time:.4f}s"],
                   textposition="auto"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=["MULLER", "Parquet"], y=[muller_size, parquet_size],
                   marker_color=["#636EFA", "#EF553B"], text=[f"{muller_size:.2f}MB", f"{parquet_size:.2f}MB"],
                   textposition="auto"),
            row=1, col=2
        )

        fig.update_yaxes(title_text="Seconds", row=1, col=1)
        fig.update_yaxes(title_text="MB", row=1, col=2)
        fig.update_layout(title=f"MULLER vs Parquet (avg of {num_runs} runs)", height=400, showlegend=False)

        return fig, None

    except Exception as e:
        return None, f"Benchmark failed: {e}"


def load_dataset(path: str) -> Tuple[Optional[Any], Optional[str]]:
    """Load an existing MULLER dataset."""
    try:
        ds = muller.load(path)
        return ds, None
    except Exception as e:
        return None, f"Failed to load dataset: {e}"


def commit_dataset(ds: Any, message: str = "Commit from Streamlit UI") -> Tuple[Optional[str], Optional[str]]:
    """Commit changes to dataset. Returns (commit_id, error)."""
    try:
        cid = ds.commit(message=message)
        return cid, None
    except Exception as e:
        return None, f"Commit failed: {e}"


def get_dataset_info(ds: Any) -> Dict[str, Any]:
    """Return summary info about the dataset."""
    try:
        return {
            "path": ds.path,
            "branch": ds.branch,
            "num_samples": len(ds),
            "tensors": {name: {"htype": t.htype, "dtype": str(t.dtype)} for name, t in ds.tensors.items()},
            "commit_id": ds.commit_id,
            "has_uncommitted": ds.has_head_changes,
        }
    except Exception as e:
        return {"error": str(e)}
