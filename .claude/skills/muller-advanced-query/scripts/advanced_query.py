#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""
MULLER Advanced Query - Indexing, vector search, and aggregation.

Operations: create-index, create-vector-index, load-vector-index, vector-search, aggregate, filter-advanced
"""

import argparse
import json
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

try:
    import muller
except ImportError:
    print(json.dumps({
        "success": False,
        "error": "ImportError",
        "message": "muller package not found. Ensure it's installed.",
        "suggestion": "Run: pip install -e . from project root"
    }))
    sys.exit(1)


def create_index(args):
    """Create inverted index for text search."""
    try:
        ds = muller.load(args.path)

        columns_arg = args.columns or args.tensors
        columns = columns_arg.split(",") if columns_arg else None
        ds.create_index(columns)

        return {
            "success": True,
            "operation": "create_index",
            "result": {
                "path": args.path,
                "columns": columns,
                "tensors": columns
            },
            "message": f"Created inverted index for {len(columns) if columns else 'all'} columns"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "create_index",
            "error": type(e).__name__,
            "message": str(e),
            "suggestion": "Ensure dataset is committed before creating index"
        }


def create_vector_index(args):
    """Create vector index."""
    try:
        ds = muller.load(args.path)
        column = args.column or args.tensor
        if not column:
            raise ValueError("Column name is required")

        kwargs = {
            "index_name": args.index_name,
            "index_type": args.index_type,
            "metric": args.metric
        }

        # Add optional parameters
        if args.ef_construction:
            kwargs["ef_construction"] = args.ef_construction
        if args.m:
            kwargs["m"] = args.m
        if args.nlist:
            kwargs["nlist"] = args.nlist

        ds.create_vector_index(column, **kwargs)

        return {
            "success": True,
            "operation": "create_vector_index",
            "result": {
                "path": args.path,
                "column": column,
                "tensor": column,
                "index_name": args.index_name,
                "index_type": args.index_type
            },
            "message": f"Created {args.index_type} vector index: {args.index_name}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "create_vector_index",
            "error": type(e).__name__,
            "message": str(e),
            "suggestion": "Ensure dataset is committed before creating index"
        }


def load_vector_index(args):
    """Load vector index into memory."""
    try:
        ds = muller.load(args.path)
        column = args.column or args.tensor
        if not column:
            raise ValueError("Column name is required")
        ds.load_vector_index(column, index_name=args.index_name)

        return {
            "success": True,
            "operation": "load_vector_index",
            "result": {
                "path": args.path,
                "column": column,
                "tensor": column,
                "index_name": args.index_name
            },
            "message": f"Loaded vector index: {args.index_name}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "load_vector_index",
            "error": type(e).__name__,
            "message": str(e)
        }


def vector_search(args):
    """Perform vector similarity search."""
    try:
        ds = muller.load(args.path, read_only=True)
        column = args.column or args.tensor
        if not column:
            raise ValueError("Column name is required")

        # Load query vectors
        if args.query_file:
            query_vector = np.load(args.query_file)
        else:
            return {
                "success": False,
                "operation": "vector_search",
                "error": "ValueError",
                "message": "Query file is required"
            }

        # Perform search
        kwargs = {"topk": args.topk}
        if args.ef_search:
            kwargs["ef_search"] = args.ef_search

        distances, indices = ds.vector_search(
            query_vector=query_vector,
            column_name=column,
            index_name=args.index_name,
            **kwargs
        )

        return {
            "success": True,
            "operation": "vector_search",
            "result": {
                "path": args.path,
                "column": column,
                "tensor": column,
                "index_name": args.index_name,
                "num_queries": len(query_vector),
                "topk": args.topk,
                "indices": indices.tolist() if hasattr(indices, 'tolist') else indices,
                "distances": distances.tolist() if hasattr(distances, 'tolist') else distances
            },
            "message": f"Found top-{args.topk} results for {len(query_vector)} queries"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "vector_search",
            "error": type(e).__name__,
            "message": str(e),
            "suggestion": "Ensure vector index is loaded first"
        }


def aggregate_query(args):
    """Run aggregation query."""
    try:
        ds = muller.load(args.path, read_only=True)

        group_by = args.group_by.split(",") if args.group_by else []
        selected = args.select.split(",") if args.select else []
        order_by = args.order_by.split(",") if args.order_by else []

        # Parse aggregate columns. The legacy --aggregate-tensors flag is still accepted.
        aggregate_columns_arg = args.aggregate_columns or args.aggregate_tensors
        aggregate_columns = []
        if aggregate_columns_arg:
            if aggregate_columns_arg == "*":
                aggregate_columns = ["*"]
            else:
                aggregate_columns = aggregate_columns_arg.split(",")

        result = ds.aggregate_vectorized(
            group_by_columns=group_by,
            selected_columns=selected,
            order_by_columns=order_by,
            aggregate_columns=aggregate_columns
        )

        return {
            "success": True,
            "operation": "aggregate",
            "result": {
                "path": args.path,
                "group_by": group_by,
                "selected": selected,
                "num_results": result.num_samples if hasattr(result, 'num_samples') else 0
            },
            "message": f"Aggregation completed"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "aggregate",
            "error": type(e).__name__,
            "message": str(e)
        }


def filter_advanced(args):
    """Advanced filtering with multiple conditions."""
    try:
        ds = muller.load(args.path, read_only=True)

        # Parse conditions
        conditions = []
        if args.conditions:
            for cond in args.conditions.split(";"):
                parts = cond.split(",")
                if len(parts) >= 3:
                    column, op, value = parts[0], parts[1], ",".join(parts[2:])
                    # Try to convert value to number
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        value = value.strip('"\'')
                    conditions.append((column, op, value))

        # Parse connectors
        connectors = args.connectors.split(",") if args.connectors else []

        result = ds.filter_vectorized(
            conditions,
            connectors,
            offset=args.offset,
            limit=args.limit
        )

        # Get sample results
        samples = []
        limit = min(args.limit if args.limit else 10, result.num_samples)
        for i, sample in enumerate(result):
            if i >= limit:
                break
            samples.append({k: str(v) for k, v in sample.items()})

        return {
            "success": True,
            "operation": "filter_advanced",
            "result": {
                "path": args.path,
                "total_matches": result.num_samples,
                "returned": len(samples),
                "samples": samples
            },
            "message": f"Found {result.num_samples} matches"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "filter_advanced",
            "error": type(e).__name__,
            "message": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="MULLER Advanced Query")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create index command
    create_index_parser = subparsers.add_parser("create-index", help="Create inverted index")
    create_index_parser.add_argument("--path", required=True, help="Dataset path")
    create_index_parser.add_argument("--columns", help="Comma-separated column names")
    create_index_parser.add_argument("--tensors", help=argparse.SUPPRESS)

    # Create vector index command
    create_vector_index_parser = subparsers.add_parser("create-vector-index", help="Create vector index")
    create_vector_index_parser.add_argument("--path", required=True, help="Dataset path")
    create_vector_index_parser.add_argument("--column", help="Column name")
    create_vector_index_parser.add_argument("--tensor", help=argparse.SUPPRESS)
    create_vector_index_parser.add_argument("--index-name", required=True, help="Index name")
    create_vector_index_parser.add_argument("--index-type", required=True, help="Index type: FLAT/HNSWFLAT/DISKANN")
    create_vector_index_parser.add_argument("--metric", default="l2", help="Distance metric: l2/cosine")
    create_vector_index_parser.add_argument("--ef-construction", type=int, help="HNSW ef_construction")
    create_vector_index_parser.add_argument("--m", type=int, help="HNSW m parameter")
    create_vector_index_parser.add_argument("--nlist", type=int, help="Number of clusters")

    # Load vector index command
    load_vector_index_parser = subparsers.add_parser("load-vector-index", help="Load vector index")
    load_vector_index_parser.add_argument("--path", required=True, help="Dataset path")
    load_vector_index_parser.add_argument("--column", help="Column name")
    load_vector_index_parser.add_argument("--tensor", help=argparse.SUPPRESS)
    load_vector_index_parser.add_argument("--index-name", required=True, help="Index name")

    # Vector search command
    vector_search_parser = subparsers.add_parser("vector-search", help="Vector similarity search")
    vector_search_parser.add_argument("--path", required=True, help="Dataset path")
    vector_search_parser.add_argument("--column", help="Column name")
    vector_search_parser.add_argument("--tensor", help=argparse.SUPPRESS)
    vector_search_parser.add_argument("--index-name", required=True, help="Index name")
    vector_search_parser.add_argument("--query-file", required=True, help="Query vectors file (.npy)")
    vector_search_parser.add_argument("--topk", type=int, default=10, help="Top K results")
    vector_search_parser.add_argument("--ef-search", type=int, help="HNSW ef_search")

    # Aggregate command
    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregation query")
    aggregate_parser.add_argument("--path", required=True, help="Dataset path")
    aggregate_parser.add_argument("--group-by", help="Comma-separated group by columns")
    aggregate_parser.add_argument("--select", help="Comma-separated selected columns")
    aggregate_parser.add_argument("--order-by", help="Comma-separated order by columns")
    aggregate_parser.add_argument("--aggregate-columns", help="Aggregate columns (* or column:func)")
    aggregate_parser.add_argument("--aggregate-tensors", help=argparse.SUPPRESS)

    # Filter advanced command
    filter_parser = subparsers.add_parser("filter-advanced", help="Advanced filtering")
    filter_parser.add_argument("--path", required=True, help="Dataset path")
    filter_parser.add_argument("--conditions", help="Conditions: column,op,value;...")
    filter_parser.add_argument("--connectors", help="Connectors: AND,OR,NOT")
    filter_parser.add_argument("--offset", type=int, default=0, help="Offset")
    filter_parser.add_argument("--limit", type=int, help="Limit")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    result = None
    if args.command == "create-index":
        result = create_index(args)
    elif args.command == "create-vector-index":
        result = create_vector_index(args)
    elif args.command == "load-vector-index":
        result = load_vector_index(args)
    elif args.command == "vector-search":
        result = vector_search(args)
    elif args.command == "aggregate":
        result = aggregate_query(args)
    elif args.command == "filter-advanced":
        result = filter_advanced(args)

    # Output JSON
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
