#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""
MULLER Dataset Manager - Manage dataset lifecycle and structure.

Operations: create, load, delete, info, stats, create-tensor, delete-tensor, rename-tensor
"""

import argparse
import json
import sys
import os

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


def create_dataset(args):
    """Create a new dataset."""
    try:
        ds = muller.dataset(args.path, overwrite=args.overwrite)

        # Create tensors if specified
        tensors_created = []
        if args.tensors:
            with ds:
                for tensor_spec in args.tensors.split(","):
                    parts = tensor_spec.split(":")
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

                    ds.create_tensor(name, htype=htype, sample_compression=compression, dtype=dtype)
                    tensors_created.append(name)

        return {
            "success": True,
            "operation": "create_dataset",
            "result": {
                "path": args.path,
                "num_tensors": len(tensors_created),
                "tensors": tensors_created
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


def create_tensor(args):
    """Create a new tensor."""
    try:
        ds = muller.load(args.path)

        with ds:
            ds.create_tensor(
                args.name,
                htype=args.htype,
                dtype=args.dtype,
                sample_compression=args.compression
            )

        return {
            "success": True,
            "operation": "create_tensor",
            "result": {
                "path": args.path,
                "tensor": args.name,
                "htype": args.htype
            },
            "message": f"Tensor '{args.name}' created"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "create_tensor",
            "error": type(e).__name__,
            "message": str(e),
            "suggestion": "Check if tensor already exists"
        }


def delete_tensor(args):
    """Delete a tensor."""
    try:
        ds = muller.load(args.path)

        with ds:
            ds.delete_tensor(args.name, large_ok=args.large_ok)

        return {
            "success": True,
            "operation": "delete_tensor",
            "result": {"path": args.path, "tensor": args.name},
            "message": f"Tensor '{args.name}' deleted"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "delete_tensor",
            "error": type(e).__name__,
            "message": str(e)
        }


def rename_tensor(args):
    """Rename a tensor."""
    try:
        ds = muller.load(args.path)

        with ds:
            ds.rename_tensor(args.old_name, args.new_name)

        return {
            "success": True,
            "operation": "rename_tensor",
            "result": {
                "path": args.path,
                "old_name": args.old_name,
                "new_name": args.new_name
            },
            "message": f"Tensor renamed: {args.old_name} -> {args.new_name}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "rename_tensor",
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
    create_parser.add_argument("--tensors", help="Tensors: name:htype:compression_or_dtype,...")

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

    # Create tensor command
    create_tensor_parser = subparsers.add_parser("create-tensor", help="Create tensor")
    create_tensor_parser.add_argument("--path", required=True, help="Dataset path")
    create_tensor_parser.add_argument("--name", required=True, help="Tensor name")
    create_tensor_parser.add_argument("--htype", default="generic", help="Tensor htype")
    create_tensor_parser.add_argument("--dtype", help="Data type")
    create_tensor_parser.add_argument("--compression", help="Sample compression")

    # Delete tensor command
    delete_tensor_parser = subparsers.add_parser("delete-tensor", help="Delete tensor")
    delete_tensor_parser.add_argument("--path", required=True, help="Dataset path")
    delete_tensor_parser.add_argument("--name", required=True, help="Tensor name")
    delete_tensor_parser.add_argument("--large-ok", action="store_true", help="Allow large delete")

    # Rename tensor command
    rename_tensor_parser = subparsers.add_parser("rename-tensor", help="Rename tensor")
    rename_tensor_parser.add_argument("--path", required=True, help="Dataset path")
    rename_tensor_parser.add_argument("--old-name", required=True, help="Old tensor name")
    rename_tensor_parser.add_argument("--new-name", required=True, help="New tensor name")

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
    elif args.command == "create-tensor":
        result = create_tensor(args)
    elif args.command == "delete-tensor":
        result = delete_tensor(args)
    elif args.command == "rename-tensor":
        result = rename_tensor(args)

    # Output JSON
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
