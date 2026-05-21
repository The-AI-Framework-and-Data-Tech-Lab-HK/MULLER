#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""
MULLER Dataset Manager - Manage dataset lifecycle and structure.

Operations: create, load, delete, info, stats, create-column, delete-column, rename-column
"""

import argparse
import json
import sys
import os

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


def create_dataset(args):
    """Create a new dataset."""
    try:
        ds = muller.dataset(args.path, overwrite=args.overwrite)

        # Create columns if specified. The legacy --tensors flag is still accepted.
        columns_created = []
        column_specs = args.columns or args.tensors
        if column_specs:
            with ds:
                for column_spec in column_specs.split(","):
                    parts = column_spec.split(":")
                    name = parts[0]
                    htype = parts[1] if len(parts) > 1 else "generic"
                    param3 = parts[2] if len(parts) > 2 else None
                    
                    # Determine if param3 is compression or dtype
                    compression = None
                    dtype = None
                    if param3:
                        # For image/video/audio, param3 is compression
                        if htype in ["image", "video", "audio"]:
                            compression = param3
                        # For others, param3 is dtype
                        else:
                            dtype = param3

                    ds.create_column(name, htype=htype, sample_compression=compression, dtype=dtype)
                    columns_created.append(name)

        return {
            "success": True,
            "operation": "create_dataset",
            "result": {
                "path": args.path,
                "num_columns": len(columns_created),
                "columns": columns_created,
                "num_tensors": len(columns_created),
                "tensors": columns_created
            },
            "message": f"Dataset created at {args.path}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "create_dataset",
            "error": type(e).__name__,
            "message": str(e),
            "suggestion": "Check path and permissions"
        }


def get_info(args):
    """Get dataset information."""
    try:
        ds = muller.load(args.path, read_only=True)

        return {
            "success": True,
            "operation": "get_info",
            "result": {
                "path": args.path,
                "num_samples": ds.num_samples,
                "columns": list(ds.columns.keys()),
                "num_columns": len(ds.columns),
                "tensors": list(ds.tensors.keys()),
                "num_tensors": len(ds.tensors)
            },
            "message": f"Dataset info retrieved"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "get_info",
            "error": type(e).__name__,
            "message": str(e),
            "suggestion": "Ensure dataset exists at path"
        }


def get_stats(args):
    """Get dataset statistics."""
    try:
        ds = muller.load(args.path, read_only=True)
        stats = ds.statistics()

        return {
            "success": True,
            "operation": "get_stats",
            "result": stats,
            "message": "Statistics retrieved"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "get_stats",
            "error": type(e).__name__,
            "message": str(e)
        }


def delete_dataset(args):
    """Delete a dataset."""
    try:
        muller.delete(args.path, large_ok=args.large_ok)

        return {
            "success": True,
            "operation": "delete_dataset",
            "result": {"path": args.path},
            "message": f"Dataset deleted at {args.path}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "delete_dataset",
            "error": type(e).__name__,
            "message": str(e)
        }


def create_column(args):
    """Create a new column."""
    try:
        ds = muller.load(args.path)

        with ds:
            ds.create_column(
                args.name,
                htype=args.htype,
                dtype=args.dtype,
                sample_compression=args.compression
            )

        return {
            "success": True,
            "operation": "create_column",
            "result": {
                "path": args.path,
                "column": args.name,
                "tensor": args.name,
                "htype": args.htype
            },
            "message": f"Column '{args.name}' created"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "create_column",
            "error": type(e).__name__,
            "message": str(e),
            "suggestion": "Check if column already exists"
        }


def delete_column(args):
    """Delete a column."""
    try:
        ds = muller.load(args.path)

        with ds:
            ds.delete_column(args.name, large_ok=args.large_ok)

        return {
            "success": True,
            "operation": "delete_column",
            "result": {"path": args.path, "column": args.name, "tensor": args.name},
            "message": f"Column '{args.name}' deleted"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "delete_column",
            "error": type(e).__name__,
            "message": str(e)
        }


def rename_column(args):
    """Rename a column."""
    try:
        ds = muller.load(args.path)

        with ds:
            ds.rename_column(args.old_name, args.new_name)

        return {
            "success": True,
            "operation": "rename_column",
            "result": {
                "path": args.path,
                "old_name": args.old_name,
                "new_name": args.new_name
            },
            "message": f"Column renamed: {args.old_name} -> {args.new_name}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "rename_column",
            "error": type(e).__name__,
            "message": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="MULLER Dataset Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create dataset")
    create_parser.add_argument("--path", required=True, help="Dataset path")
    create_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing")
    create_parser.add_argument("--columns", help="Columns: name:htype:compression_or_dtype,...")
    create_parser.add_argument("--tensors", help=argparse.SUPPRESS)

    # Info command
    info_parser = subparsers.add_parser("info", help="Get dataset info")
    info_parser.add_argument("--path", required=True, help="Dataset path")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics")
    stats_parser.add_argument("--path", required=True, help="Dataset path")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete dataset")
    delete_parser.add_argument("--path", required=True, help="Dataset path")
    delete_parser.add_argument("--large-ok", action="store_true", help="Allow large delete")

    # Column commands. Legacy tensor command names remain accepted.
    for command_name, help_text in (
        ("create-column", "Create column"),
        ("create-tensor", argparse.SUPPRESS),
    ):
        create_column_parser = subparsers.add_parser(command_name, help=help_text)
        create_column_parser.add_argument("--path", required=True, help="Dataset path")
        create_column_parser.add_argument("--name", required=True, help="Column name")
        create_column_parser.add_argument("--htype", default="generic", help="Column htype")
        create_column_parser.add_argument("--dtype", help="Data type")
        create_column_parser.add_argument("--compression", help="Sample compression")

    for command_name, help_text in (
        ("delete-column", "Delete column"),
        ("delete-tensor", argparse.SUPPRESS),
    ):
        delete_column_parser = subparsers.add_parser(command_name, help=help_text)
        delete_column_parser.add_argument("--path", required=True, help="Dataset path")
        delete_column_parser.add_argument("--name", required=True, help="Column name")
        delete_column_parser.add_argument("--large-ok", action="store_true", help="Allow large delete")

    for command_name, help_text in (
        ("rename-column", "Rename column"),
        ("rename-tensor", argparse.SUPPRESS),
    ):
        rename_column_parser = subparsers.add_parser(command_name, help=help_text)
        rename_column_parser.add_argument("--path", required=True, help="Dataset path")
        rename_column_parser.add_argument("--old-name", required=True, help="Old column name")
        rename_column_parser.add_argument("--new-name", required=True, help="New column name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    result = None
    if args.command == "create":
        result = create_dataset(args)
    elif args.command == "info":
        result = get_info(args)
    elif args.command == "stats":
        result = get_stats(args)
    elif args.command == "delete":
        result = delete_dataset(args)
    elif args.command in ("create-column", "create-tensor"):
        result = create_column(args)
    elif args.command in ("delete-column", "delete-tensor"):
        result = delete_column(args)
    elif args.command in ("rename-column", "rename-tensor"):
        result = rename_column(args)

    # Output JSON
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
