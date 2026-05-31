#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""
MULLER Export - Export datasets to various formats.

Operations: to-arrow, to-parquet, to-json, to-numpy, to-mindrecord, get-info
"""

import argparse
import json
import sys
import os
import numpy as np

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


def export_to_arrow(args):
    """Export to Apache Arrow format."""
    try:
        ds = muller.load(args.path, read_only=True)

        # Export to Arrow
        arrow_ds = ds.to_arrow()

        if args.output:
            # Save to file if output specified
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Write to Parquet file (Arrow format)
            pq.write_table(arrow_ds, args.output)

            return {
                "success": True,
                "operation": "to_arrow",
                "result": {
                    "path": args.path,
                    "output": args.output,
                    "num_rows": len(arrow_ds)
                },
                "message": f"Exported to Arrow format: {args.output}"
            }
        else:
            return {
                "success": True,
                "operation": "to_arrow",
                "result": {
                    "path": args.path,
                    "num_rows": len(arrow_ds)
                },
                "message": "Converted to Arrow format (in-memory)"
            }
    except Exception as e:
        return {
            "success": False,
            "operation": "to_arrow",
            "error": type(e).__name__,
            "message": str(e),
            "suggestion": "Ensure pyarrow is installed: pip install pyarrow"
        }


def export_to_parquet(args):
    """Export to Parquet files."""
    try:
        ds = muller.load(args.path, read_only=True)

        # Export to Parquet
        ds.write_to_parquet(args.output)

        return {
            "success": True,
            "operation": "to_parquet",
            "result": {
                "path": args.path,
                "output": args.output,
                "num_samples": ds.num_samples
            },
            "message": f"Exported to Parquet: {args.output}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "to_parquet",
            "error": type(e).__name__,
            "message": str(e)
        }


def export_to_json(args):
    """Export to JSON format."""
    try:
        ds = muller.load(args.path, read_only=True)

        # Export to JSON
        json_data = ds.to_json(
            offset=args.offset,
            limit=args.limit if args.limit else ds.num_samples
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(json_data, f, indent=2 if args.pretty else None)

            return {
                "success": True,
                "operation": "to_json",
                "result": {
                    "path": args.path,
                    "output": args.output,
                    "num_samples": len(json_data) if isinstance(json_data, list) else 1
                },
                "message": f"Exported to JSON: {args.output}"
            }
        else:
            return {
                "success": True,
                "operation": "to_json",
                "result": {
                    "path": args.path,
                    "data": json_data
                },
                "message": "Converted to JSON (in-memory)"
            }
    except Exception as e:
        return {
            "success": False,
            "operation": "to_json",
            "error": type(e).__name__,
            "message": str(e)
        }


def export_to_numpy(args):
    """Convert tensor to NumPy array."""
    try:
        ds = muller.load(args.path, read_only=True)

        if not args.tensor:
            return {
                "success": False,
                "operation": "to_numpy",
                "error": "ValueError",
                "message": "Tensor name is required for NumPy export"
            }

        # Get tensor data
        tensor_data = ds[args.tensor].numpy()

        if args.output:
            np.save(args.output, tensor_data)

            return {
                "success": True,
                "operation": "to_numpy",
                "result": {
                    "path": args.path,
                    "tensor": args.tensor,
                    "output": args.output,
                    "shape": tensor_data.shape,
                    "dtype": str(tensor_data.dtype)
                },
                "message": f"Exported tensor to NumPy: {args.output}"
            }
        else:
            return {
                "success": True,
                "operation": "to_numpy",
                "result": {
                    "path": args.path,
                    "tensor": args.tensor,
                    "shape": tensor_data.shape,
                    "dtype": str(tensor_data.dtype)
                },
                "message": "Converted tensor to NumPy (in-memory)"
            }
    except Exception as e:
        return {
            "success": False,
            "operation": "to_numpy",
            "error": type(e).__name__,
            "message": str(e)
        }


def export_to_mindrecord(args):
    """Export to MindRecord format."""
    try:
        ds = muller.load(args.path, read_only=True)

        # Export to MindRecord
        ds.to_mindrecord(args.output)

        return {
            "success": True,
            "operation": "to_mindrecord",
            "result": {
                "path": args.path,
                "output": args.output,
                "num_samples": ds.num_samples
            },
            "message": f"Exported to MindRecord: {args.output}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "to_mindrecord",
            "error": type(e).__name__,
            "message": str(e),
            "suggestion": "Ensure MindSpore is installed for MindRecord export"
        }


def get_export_info(args):
    """Get export information."""
    try:
        ds = muller.load(args.path, read_only=True)

        return {
            "success": True,
            "operation": "get_info",
            "result": {
                "path": args.path,
                "num_samples": ds.num_samples,
                "tensors": list(ds.tensors.keys()),
                "supported_formats": [
                    "arrow",
                    "parquet",
                    "json",
                    "numpy",
                    "mindrecord"
                ]
            },
            "message": "Export information retrieved"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "get_info",
            "error": type(e).__name__,
            "message": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="MULLER Export")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # To Arrow command
    to_arrow_parser = subparsers.add_parser("to-arrow", help="Export to Arrow")
    to_arrow_parser.add_argument("--path", required=True, help="Dataset path")
    to_arrow_parser.add_argument("--output", help="Output file path")

    # To Parquet command
    to_parquet_parser = subparsers.add_parser("to-parquet", help="Export to Parquet")
    to_parquet_parser.add_argument("--path", required=True, help="Dataset path")
    to_parquet_parser.add_argument("--output", required=True, help="Output directory")

    # To JSON command
    to_json_parser = subparsers.add_parser("to-json", help="Export to JSON")
    to_json_parser.add_argument("--path", required=True, help="Dataset path")
    to_json_parser.add_argument("--output", help="Output file path")
    to_json_parser.add_argument("--offset", type=int, default=0, help="Offset")
    to_json_parser.add_argument("--limit", type=int, help="Limit")
    to_json_parser.add_argument("--pretty", action="store_true", help="Pretty print")

    # To NumPy command
    to_numpy_parser = subparsers.add_parser("to-numpy", help="Export tensor to NumPy")
    to_numpy_parser.add_argument("--path", required=True, help="Dataset path")
    to_numpy_parser.add_argument("--tensor", required=True, help="Tensor name")
    to_numpy_parser.add_argument("--output", help="Output .npy file")

    # To MindRecord command
    to_mindrecord_parser = subparsers.add_parser("to-mindrecord", help="Export to MindRecord")
    to_mindrecord_parser.add_argument("--path", required=True, help="Dataset path")
    to_mindrecord_parser.add_argument("--output", required=True, help="Output file path")

    # Get info command
    get_info_parser = subparsers.add_parser("get-info", help="Get export info")
    get_info_parser.add_argument("--path", required=True, help="Dataset path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    result = None
    if args.command == "to-arrow":
        result = export_to_arrow(args)
    elif args.command == "to-parquet":
        result = export_to_parquet(args)
    elif args.command == "to-json":
        result = export_to_json(args)
    elif args.command == "to-numpy":
        result = export_to_numpy(args)
    elif args.command == "to-mindrecord":
        result = export_to_mindrecord(args)
    elif args.command == "get-info":
        result = get_export_info(args)

    # Output JSON
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
