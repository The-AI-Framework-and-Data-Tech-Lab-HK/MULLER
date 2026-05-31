#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""
MULLER Version Control - Git-like version control for datasets.

Operations: commit, checkout, branch, merge, log, diff, commits
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


def commit_changes(args):
    """Commit current changes."""
    try:
        ds = muller.load(args.path)
        commit_id = ds.commit(message=args.message, allow_empty=args.allow_empty)

        return {
            "success": True,
            "operation": "commit",
            "result": {
                "path": args.path,
                "commit_id": commit_id,
                "message": args.message
            },
            "message": f"Committed changes: {commit_id[:8]}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "commit",
            "error": type(e).__name__,
            "message": str(e)
        }


def checkout_branch(args):
    """Checkout or create branch."""
    try:
        ds = muller.load(args.path)
        ds.checkout(args.branch, create=args.create)

        return {
            "success": True,
            "operation": "checkout",
            "result": {
                "path": args.path,
                "branch": args.branch,
                "created": args.create
            },
            "message": f"Checked out branch: {args.branch}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "checkout",
            "error": type(e).__name__,
            "message": str(e),
            "suggestion": "Use --create to create new branch"
        }


def list_branches(args):
    """List all branches."""
    try:
        ds = muller.load(args.path, read_only=True)
        branches = ds.branches
        current = ds.branch

        return {
            "success": True,
            "operation": "branch",
            "result": {
                "path": args.path,
                "current_branch": current,
                "branches": branches,
                "count": len(branches)
            },
            "message": f"Found {len(branches)} branches"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "branch",
            "error": type(e).__name__,
            "message": str(e)
        }


def merge_branch(args):
    """Merge branch with conflict resolution."""
    try:
        ds = muller.load(args.path)

        # Check for conflicts first
        if args.check_conflicts:
            conflict_cols, conflict_records = ds.detect_merge_conflict(args.branch, show_value=True)
            return {
                "success": True,
                "operation": "merge_check",
                "result": {
                    "path": args.path,
                    "branch": args.branch,
                    "has_conflicts": len(conflict_cols) > 0,
                    "conflict_columns": conflict_cols,
                    "conflict_records": conflict_records
                },
                "message": f"Conflicts detected: {len(conflict_cols)} columns"
            }

        # Perform merge
        ds.merge(
            args.branch,
            conflict_resolution=args.conflict_resolution,
            append_resolution=args.append_resolution,
            pop_resolution=args.pop_resolution,
            update_resolution=args.update_resolution
        )

        return {
            "success": True,
            "operation": "merge",
            "result": {
                "path": args.path,
                "branch": args.branch,
                "current_branch": ds.branch
            },
            "message": f"Merged {args.branch} into {ds.branch}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "merge",
            "error": type(e).__name__,
            "message": str(e),
            "suggestion": "Check for conflicts first with --check-conflicts"
        }


def view_log(args):
    """View commit history."""
    try:
        ds = muller.load(args.path, read_only=True)
        log = ds.log(ordered_by_date=args.ordered_by_date)

        return {
            "success": True,
            "operation": "log",
            "result": {
                "path": args.path,
                "branch": ds.branch,
                "log": log
            },
            "message": f"Retrieved commit history"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "log",
            "error": type(e).__name__,
            "message": str(e)
        }


def compare_diff(args):
    """Compare commits or branches."""
    try:
        ds = muller.load(args.path, read_only=True)

        if args.as_dataframe:
            diff = ds.direct_diff(args.id1, args.id2, as_dataframe=True, force=args.force)
        else:
            diff = ds.diff(args.id1, args.id2, as_dataframe=False)

        return {
            "success": True,
            "operation": "diff",
            "result": {
                "path": args.path,
                "id1": args.id1,
                "id2": args.id2,
                "diff": str(diff)
            },
            "message": f"Compared {args.id1} and {args.id2}"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "diff",
            "error": type(e).__name__,
            "message": str(e)
        }


def list_commits(args):
    """List all commits."""
    try:
        ds = muller.load(args.path, read_only=True)
        commits = ds.commits(ordered_by_date=args.ordered_by_date)

        return {
            "success": True,
            "operation": "commits",
            "result": {
                "path": args.path,
                "commits": commits,
                "count": len(commits)
            },
            "message": f"Found {len(commits)} commits"
        }
    except Exception as e:
        return {
            "success": False,
            "operation": "commits",
            "error": type(e).__name__,
            "message": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="MULLER Version Control")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Commit command
    commit_parser = subparsers.add_parser("commit", help="Commit changes")
    commit_parser.add_argument("--path", required=True, help="Dataset path")
    commit_parser.add_argument("--message", required=True, help="Commit message")
    commit_parser.add_argument("--allow-empty", action="store_true", help="Allow empty commit")

    # Checkout command
    checkout_parser = subparsers.add_parser("checkout", help="Checkout branch")
    checkout_parser.add_argument("--path", required=True, help="Dataset path")
    checkout_parser.add_argument("--branch", required=True, help="Branch name")
    checkout_parser.add_argument("--create", action="store_true", help="Create new branch")

    # Branch command
    branch_parser = subparsers.add_parser("branch", help="List branches")
    branch_parser.add_argument("--path", required=True, help="Dataset path")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge branch")
    merge_parser.add_argument("--path", required=True, help="Dataset path")
    merge_parser.add_argument("--branch", required=True, help="Branch to merge")
    merge_parser.add_argument("--check-conflicts", action="store_true", help="Check conflicts only")
    merge_parser.add_argument("--conflict-resolution", default="manual", help="Conflict resolution strategy")
    merge_parser.add_argument("--append-resolution", default="both", help="Append resolution: ours/theirs/both")
    merge_parser.add_argument("--pop-resolution", default="ours", help="Pop resolution: ours/theirs")
    merge_parser.add_argument("--update-resolution", default="ours", help="Update resolution: ours/theirs")

    # Log command
    log_parser = subparsers.add_parser("log", help="View commit history")
    log_parser.add_argument("--path", required=True, help="Dataset path")
    log_parser.add_argument("--ordered-by-date", action="store_true", help="Order by date")

    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Compare commits/branches")
    diff_parser.add_argument("--path", required=True, help="Dataset path")
    diff_parser.add_argument("--id1", required=True, help="First commit/branch")
    diff_parser.add_argument("--id2", required=True, help="Second commit/branch")
    diff_parser.add_argument("--as-dataframe", action="store_true", help="Return as dataframe")
    diff_parser.add_argument("--force", action="store_true", help="Force diff")

    # Commits command
    commits_parser = subparsers.add_parser("commits", help="List all commits")
    commits_parser.add_argument("--path", required=True, help="Dataset path")
    commits_parser.add_argument("--ordered-by-date", action="store_true", help="Order by date")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    result = None
    if args.command == "commit":
        result = commit_changes(args)
    elif args.command == "checkout":
        result = checkout_branch(args)
    elif args.command == "branch":
        result = list_branches(args)
    elif args.command == "merge":
        result = merge_branch(args)
    elif args.command == "log":
        result = view_log(args)
    elif args.command == "diff":
        result = compare_diff(args)
    elif args.command == "commits":
        result = list_commits(args)

    # Output JSON
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
