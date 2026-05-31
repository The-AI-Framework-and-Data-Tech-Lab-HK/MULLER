#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""
MULLER Data Operations - Handle CRUD operations on dataset samples.

Operations: append, extend, update, delete, query, import
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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


def append_sample(args):
    """Append a single sample."""
    try:
        ds = muller.load(args.path)
        data = json.loads(args.data)

        with ds:
            ds.append(data)

        return {
            "success": True,
            "operation": "append",
            "result": {"path": args.path, "num_samples": ds.num_samples},
            "message": "Sample appended"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "append",
            "error": type(e).__name__,
            "message": str(e)
        }


def extend_samples(args):
    """Extend with multiple samples."""
    try:
        ds = muller.load(args.path)

        if args.data_file:
            with open(args.data_file) as f:
                data = json.load(f)
        else:
            data = json.loads(args.data)

        with ds:
            ds.extend(data)

        return {
            "success": True,
            "operation": "extend",
            "result": {"path": args.path, "num_samples": ds.num_samples},
            "message": f"Extended dataset to {ds.num_samples} samples"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "extend",
            "error": type(e).__name__,
            "message": str(e)
        }


def update_sample(args):
    """Update existing sample."""
    try:
        ds = muller.load(args.path)
        data = json.loads(args.data)

        with ds:
            ds[args.index].update(data)

        return {
            "success": True,
            "operation": "update",
            "result": {"path": args.path, "index": args.index},
            "message": f"Sample {args.index} updated"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "update",
            "error": type(e).__name__,
            "message": str(e)
        }


def delete_samples(args):
    """Delete samples."""
    try:
        ds = muller.load(args.path)

        if args.indices:
            indices = [int(i) for i in args.indices.split(",")]
        else:
            indices = args.index

        with ds:
            ds.pop(indices)

        return {
            "success": True,
            "operation": "delete",
            "result": {"path": args.path, "num_samples": ds.num_samples},
            "message": f"Samples deleted, {ds.num_samples} remaining"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "delete",
            "error": type(e).__name__,
            "message": str(e)
        }


def query_samples(args):
    """Query and filter samples."""
    try:
        ds = muller.load(args.path, read_only=True)

        if args.filter:
            # Parse filter: "tensor op value"
            parts = args.filter.split()
            if len(parts) >= 3:
                tensor = parts[0]
                op = parts[1]
                value = " ".join(parts[2:])

                # Try to convert value to number
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    value = value.strip('"\'')

                result = ds.filter_vectorized([(tensor, op, value)])
            else:
                result = ds
        else:
            result = ds

        # Apply limit
        limit = args.limit if args.limit else result.num_samples
        samples = []
        for i, sample in enumerate(result):
            if i >= limit:
                break
            samples.append({k: str(v) for k, v in sample.items()})

        return {
            "success": True,
            "operation": "query",
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
            "operation": "query",
            "error": type(e).__name__,
            "message": str(e)
        }


def import_data(args):
    """Import data from file."""
    try:
        if args.schema_file:
            with open(args.schema_file) as f:
                schema = json.load(f)
        else:
            schema = None

        ds = muller.from_file(args.source, args.path, schema=schema)

        return {
            "success": True,
            "operation": "import",
            "result": {
                "path": args.path,
                "source": args.source,
                "num_samples": ds.num_samples
            },
            "message": f"Imported {ds.num_samples} samples"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "import",
            "error": type(e).__name__,
            "message": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="MULLER Data Operations")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Append command
    append_parser = subparsers.add_parser("append", help="Append sample")
    append_parser.add_argument("--path", required=True, help="Dataset path")
    append_parser.add_argument("--data", required=True, help="JSON data")

    # Extend command
    extend_parser = subparsers.add_parser("extend", help="Extend samples")
    extend_parser.add_argument("--path", required=True, help="Dataset path")
    extend_parser.add_argument("--data", help="JSON data")
    extend_parser.add_argument("--data-file", help="JSON file path")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update sample")
    update_parser.add_argument("--path", required=True, help="Dataset path")
    update_parser.add_argument("--index", type=int, required=True, help="Sample index")
    update_parser.add_argument("--data", required=True, help="JSON data")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete samples")
    delete_parser.add_argument("--path", required=True, help="Dataset path")
    delete_parser.add_argument("--index", type=int, help="Single index")
    delete_parser.add_argument("--indices", help="Comma-separated indices")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query samples")
    query_parser.add_argument("--path", required=True, help="Dataset path")
    query_parser.add_argument("--filter", help="Filter: 'tensor op value'")
    query_parser.add_argument("--limit", type=int, help="Max results")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import from file")
    import_parser.add_argument("--path", required=True, help="Dataset path")
    import_parser.add_argument("--source", required=True, help="Source file")
    import_parser.add_argument("--schema-file", help="Schema JSON file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    result = None
    if args.command == "append":
        result = append_sample(args)
    elif args.command == "extend":
        result = extend_samples(args)
    elif args.command == "update":
        result = update_sample(args)
    elif args.command == "delete":
        result = delete_samples(args)
    elif args.command == "query":
        result = query_samples(args)
    elif args.command == "import":
        result = import_data(args)

    # Output JSON
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
