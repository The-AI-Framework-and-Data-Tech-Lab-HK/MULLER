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
from typing import Optional, List, Tuple, Dict, Any, Union
import time
import os
import plotly.graph_objects as go


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


def add_samples(ds: Any, data: Dict[str, List], auto_commit: bool = True) -> Optional[str]:
    """Add samples to dataset using per-tensor extend."""
    try:
        with ds:
            for tensor_name, values in data.items():
                ds[tensor_name].extend(values)
        if auto_commit:
            ds.commit(message="Add samples via Streamlit UI")
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
    """Convert dataset (or view) to pandas DataFrame."""
    try:
        if end is not None:
            view = ds[start:end]
        elif start > 0:
            view = ds[start:]
        else:
            view = ds
        df = view.to_dataframe(tensor_list=tensor_list)
        return df, None
    except Exception as e:
        # Fallback: manual conversion for filtered views that may not support to_dataframe
        try:
            if tensor_list is None:
                tensor_list = list(ds.tensors.keys())
            data = {}
            for tname in tensor_list:
                tensor = ds[tname]
                vals = tensor.numpy(aslist=True)
                if end is not None:
                    vals = vals[start:end]
                elif start > 0:
                    vals = vals[start:]
                data[tname] = vals
            return pd.DataFrame(data), None
        except Exception as e2:
            return None, f"Failed to convert to DataFrame: {e2}"


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

    # BFS topological order from FIRST_COMMIT_ID
    root = commit_node_map.get(FIRST_COMMIT_ID)
    if not root:
        return {"commits": [], "branches": [], "lane_count": 0}

    topo_order = []
    visited = set()
    queue = deque([root])
    visited.add(root.commit_id)
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for child in node.children:
            if child.commit_id not in visited:
                visited.add(child.commit_id)
                queue.append(child)

    # Reverse so newest first
    topo_order.reverse()

    # Build commit records
    commits = []
    for row_idx, node in enumerate(topo_order):
        lane = branch_lanes.get(node.branch, 0)
        commits.append({
            "id": node.commit_id,
            "short_id": node.commit_id[:8],
            "branch": node.branch,
            "lane": lane,
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
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: transparent; }}
  .graph-container {{ position: relative; overflow: auto; max-height: {height}px; }}
  .tooltip {{
    position: absolute; display: none; background: #1e1e2e; color: #cdd6f4;
    padding: 10px 14px; border-radius: 8px; font-size: 12px; line-height: 1.5;
    pointer-events: none; z-index: 100; white-space: pre-line; max-width: 320px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3); border: 1px solid #45475a;
  }}
  .tooltip .hash {{ color: #89b4fa; font-family: monospace; }}
  .tooltip .branch-name {{ color: #a6e3a1; font-weight: 600; }}
  .tooltip .msg {{ color: #f5e0dc; }}
  .tooltip .meta {{ color: #9399b2; font-size: 11px; }}
</style>
</head>
<body>
<div class="graph-container" id="container">
  <svg id="graph" xmlns="http://www.w3.org/2000/svg"></svg>
  <div class="tooltip" id="tooltip"></div>
</div>
<script>
(function() {{
  const data = {json_data};
  const COL_W = 60, LANE_H = 50, R = 8;
  const LEFT_M = 80, TOP_M = 20;
  const COLORS = ["#4CAF50","#2196F3","#FF9800","#9C27B0","#F44336","#00BCD4","#795548","#607D8B","#E91E63","#CDDC39"];
  const N = data.commits ? data.commits.length : 0;

  if (N === 0) {{
    document.getElementById("container").innerHTML = '<p style="padding:16px;color:#888;">No commits yet.</p>';
    return;
  }}

  const svg = document.getElementById("graph");
  const tooltip = document.getElementById("tooltip");
  const W = LEFT_M + N * COL_W + 30;
  const H = TOP_M + data.lane_count * LANE_H + 30;
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);

  // Horizontal layout: x = time axis (oldest left, newest right), y = branch lane
  const pos = {{}};
  data.commits.forEach((c, i) => {{
    pos[c.id] = {{ x: LEFT_M + (N - 1 - i) * COL_W, y: TOP_M + c.lane * LANE_H }};
  }});

  function color(lane) {{ return COLORS[lane % COLORS.length]; }}

  function svgEl(tag, attrs) {{
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }}

  // Branch labels on the left
  data.branches.forEach(b => {{
    const y = TOP_M + b.lane * LANE_H;
    const label = svgEl("text", {{
      x: LEFT_M - 12, y: y + 4, "text-anchor": "end", fill: color(b.lane),
      "font-size": b.is_current ? "13px" : "11px",
      "font-weight": b.is_current ? "700" : "500",
      "font-family": "-apple-system, BlinkMacSystemFont, sans-serif"
    }});
    label.textContent = b.name + (b.is_current ? " *" : "");
    svg.appendChild(label);
  }});

  // Horizontal lane guide lines
  data.branches.forEach(b => {{
    const y = TOP_M + b.lane * LANE_H;
    svg.appendChild(svgEl("line", {{
      x1: LEFT_M - 8, y1: y, x2: W - 10, y2: y,
      stroke: color(b.lane), "stroke-width": 1, opacity: 0.12
    }}));
  }});

  // Connection lines
  data.commits.forEach(c => {{
    if (c.parent_id && pos[c.parent_id]) {{
      const from = pos[c.id];
      const to = pos[c.parent_id];
      if (from.y === to.y) {{
        svg.appendChild(svgEl("line", {{
          x1: from.x - R, y1: from.y, x2: to.x + R, y2: to.y,
          stroke: color(c.lane), "stroke-width": 2, opacity: 0.7
        }}));
      }} else {{
        const midX = (from.x + to.x) / 2;
        svg.appendChild(svgEl("path", {{
          d: `M ${{from.x - R}} ${{from.y}} C ${{midX}} ${{from.y}}, ${{midX}} ${{to.y}}, ${{to.x + R}} ${{to.y}}`,
          stroke: color(c.lane), "stroke-width": 2, fill: "none", opacity: 0.7
        }}));
      }}
    }}
    if (c.merge_parent_id && pos[c.merge_parent_id]) {{
      const from = pos[c.merge_parent_id];
      const to = pos[c.id];
      const midX = (from.x + to.x) / 2;
      svg.appendChild(svgEl("path", {{
        d: `M ${{from.x + R}} ${{from.y}} C ${{midX}} ${{from.y}}, ${{midX}} ${{to.y}}, ${{to.x - R}} ${{to.y}}`,
        stroke: color(data.commits.find(x => x.id === c.merge_parent_id)?.lane || c.lane),
        "stroke-width": 2, fill: "none", "stroke-dasharray": "6,3", opacity: 0.6
      }}));
    }}
  }});

  // Commit nodes
  data.commits.forEach(c => {{
    const cx = pos[c.id].x;
    const cy = pos[c.id].y;
    const col = color(c.lane);
    const g = svgEl("g", {{}});

    if (c.is_current) {{
      const ring = svgEl("circle", {{
        cx: cx, cy: cy, r: 14, fill: "none", stroke: col,
        "stroke-width": 2, opacity: 0.4
      }});
      ring.innerHTML = `<animate attributeName="r" values="12;17;12" dur="2s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="0.5;0.15;0.5" dur="2s" repeatCount="indefinite"/>`;
      g.appendChild(ring);
    }}

    if (c.is_merge) {{
      g.appendChild(svgEl("circle", {{
        cx: cx, cy: cy, r: R, fill: col, stroke: "#fff", "stroke-width": 2
      }}));
    }} else if (c.is_checkout) {{
      g.appendChild(svgEl("circle", {{
        cx: cx, cy: cy, r: R, fill: "#fff", stroke: col, "stroke-width": 2.5
      }}));
    }} else {{
      g.appendChild(svgEl("circle", {{
        cx: cx, cy: cy, r: R, fill: col, stroke: col, "stroke-width": 1
      }}));
    }}

    // Tooltip
    g.style.cursor = "default";
    g.addEventListener("mouseenter", (e) => {{
      let html = `<span class="hash">${{c.short_id}}</span> <span class="branch-name">${{c.branch}}</span>`;
      if (c.is_merge) html += ' <span style="color:#f9e2af">[merge]</span>';
      if (c.is_checkout) html += ' <span style="color:#89dceb">[checkout]</span>';
      if (c.is_current) html += ' <span style="color:#a6e3a1">[current]</span>';
      html += `\\n`;
      if (c.message) html += `<span class="msg">${{c.message}}</span>\\n`;
      else if (c.is_head && !c.time) html += `<span class="msg">(Uncommitted working copy)</span>\\n`;
      if (c.author || c.time) html += `<span class="meta">${{c.author}}${{c.time ? " — " + c.time : ""}}</span>`;
      tooltip.innerHTML = html;
      tooltip.style.display = "block";
      const rect = document.getElementById("container").getBoundingClientRect();
      tooltip.style.left = (e.clientX - rect.left + 15) + "px";
      tooltip.style.top = (e.clientY - rect.top - 10) + "px";
    }});
    g.addEventListener("mousemove", (e) => {{
      const rect = document.getElementById("container").getBoundingClientRect();
      tooltip.style.left = (e.clientX - rect.left + 15) + "px";
      tooltip.style.top = (e.clientY - rect.top - 10) + "px";
    }});
    g.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});

    svg.appendChild(g);
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
