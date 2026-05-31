---
name: muller-export
description: Export and integrate MULLER datasets with other formats - Arrow, Parquet, JSON, NumPy, MindRecord. Use when user wants to export data, convert formats, or integrate with other frameworks.
compatibility: Requires Python 3.11+, muller package installed
---

# MULLER Export & Integration

## IMPORTANT: How to Use This Skill

**DO NOT create new Python files.** Always use the existing script:
- Use `scripts/export.py` for all export and integration operations

Execute the script directly with `python3` command. Never write new scripts to the project root.

## DEMO MODE: Speed Instructions

When executing operations:
- Run commands immediately without pre-explanation
- Do not summarize or explain results after execution
- Only show the JSON output from the script
- Skip all follow-up suggestions unless user asks

## When to Use This Skill

Use this skill when the user wants to:
- Export datasets to Arrow format
- Export datasets to Parquet files
- Export datasets to JSON format
- Convert datasets to NumPy arrays
- Export to MindRecord format (for MindSpore)
- Integrate with PyTorch, TensorFlow, or other frameworks

## Available Script

### scripts/export.py

Handles export and format conversion operations.

**Operations:**
- `to-arrow` - Export to Apache Arrow format
- `to-parquet` - Export to Parquet files
- `to-json` - Export to JSON format
- `to-numpy` - Convert tensors to NumPy arrays
- `to-mindrecord` - Export to MindRecord format
- `get-info` - Get export information

**Usage:**
```bash
# Export to Arrow
python3 .claude/skills/muller-export/scripts/export.py to-arrow \
  --path ./my_dataset --output ./output.arrow

# Export to Parquet
python3 .claude/skills/muller-export/scripts/export.py to-parquet \
  --path ./my_dataset --output ./output_dir

# Export to JSON
python3 .claude/skills/muller-export/scripts/export.py to-json \
  --path ./my_dataset --output ./output.json

# Convert tensor to NumPy
python3 .claude/skills/muller-export/scripts/export.py to-numpy \
  --path ./my_dataset --tensor embeddings --output ./embeddings.npy

# Export to MindRecord
python3 .claude/skills/muller-export/scripts/export.py to-mindrecord \
  --path ./my_dataset --output ./output.mindrecord
```

## Quick Reference

Export formats: Arrow (columnar exchange), Parquet (storage/analytics), JSON (web APIs), NumPy (ML frameworks), MindRecord (MindSpore).

For detailed workflows, check references/ directory.
