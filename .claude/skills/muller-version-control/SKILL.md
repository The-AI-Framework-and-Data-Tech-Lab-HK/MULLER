---
name: muller-version-control
description: Git-like version control for MULLER datasets - commit, branch, merge, diff, and conflict resolution. Use when user wants to version datasets, create branches, merge changes, or view history.
compatibility: Requires Python 3.11+, muller package installed
---

# MULLER Version Control

## IMPORTANT: How to Use This Skill

**DO NOT create new Python files.** Always use the existing script:
- Use `scripts/version_control.py` for all version control operations

Execute the script directly with `python3` command. Never write new scripts to the project root.

## DEMO MODE: Speed Instructions

When executing operations:
- Run commands immediately without pre-explanation
- Do not summarize or explain results after execution
- Only show the JSON output from the script
- Skip all follow-up suggestions unless user asks

## When to Use This Skill

Use this skill when the user wants to:
- Commit dataset changes
- Create and switch branches
- Merge branches with conflict resolution
- View commit history and logs
- Compare differences between commits/branches
- Checkout specific commits or branches

## Available Script

### scripts/version_control.py

Manages Git-like version control operations.

**Operations:**
- `commit` - Commit current changes
- `checkout` - Switch or create branches
- `branch` - List all branches
- `merge` - Merge branches with conflict resolution
- `log` - View commit history
- `diff` - Compare commits or branches
- `commits` - List all commits

**Usage:**
```bash
# Commit changes
python3 .claude/skills/muller-version-control/scripts/version_control.py commit \
  --path ./my_dataset --message "Added new samples"

# Create and checkout new branch
python3 .claude/skills/muller-version-control/scripts/version_control.py checkout \
  --path ./my_dataset --branch dev-1 --create

# List branches
python3 .claude/skills/muller-version-control/scripts/version_control.py branch \
  --path ./my_dataset

# Merge branch
python3 .claude/skills/muller-version-control/scripts/version_control.py merge \
  --path ./my_dataset --branch dev-1 --append-resolution both

# View log
python3 .claude/skills/muller-version-control/scripts/version_control.py log \
  --path ./my_dataset

# Compare branches
python3 .claude/skills/muller-version-control/scripts/version_control.py diff \
  --path ./my_dataset --id1 main --id2 dev-1
```

## Quick Reference

Merge conflict resolution strategies:
- `append-resolution`: ours/theirs/both (default: both)
- `pop-resolution`: ours/theirs/manual
- `update-resolution`: ours/theirs/manual

For detailed workflows, check references/ directory.
