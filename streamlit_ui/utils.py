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
