# Dataset Version Control

This page documents version control methods for managing dataset history, branches, and commits.

## Table of Contents

- [ds.commit()](#dscommit)
- [ds.checkout()](#dscheckout)
- [ds.commits()](#dscommits)
- [ds.merge()](#dsmerge)
- [ds.detect_merge_conflict()](#dsdetect_merge_conflict)
- [ds.diff()](#dsdiff)
- [ds.direct_diff()](#dsdirect_diff)
- [ds.diff_to_prev()](#dsdiff_to_prev)
- [ds.reset()](#dsreset)
- [ds.log()](#dslog)
- [ds.delete_branch()](#dsdelete_branch)
- [ds.branch](#dsbranch)
- [ds.branches](#dsbranches)
- [ds.commit_id](#dscommit_id)

---

### ds.commit()

#### Overview

Create a commit to save the current state of the dataset. This creates a snapshot of all changes made since the last commit.

#### Parameters

- **message** (`str`, optional): Commit message describing the changes. If not provided, an automatic message will be generated. Defaults to `None`.
- **allow_empty** (`bool`, optional): If `True`, allows creating a commit even when there are no changes. Defaults to `False`.

#### Returns

- **str**: The commit ID of the newly created commit.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Make changes and commit
with ds:
    ds.append({"images": image_data, "labels": 1})
    commit_id = ds.commit("Added new training sample")
    print(f"Created commit: {commit_id}")

# Commit with automatic message
with ds:
    ds.extend({"images": images, "labels": labels})
    commit_id = ds.commit()

# Allow empty commit
commit_id = ds.commit("Empty checkpoint", allow_empty=True)

# Commit after multiple operations
with ds:
    ds.create_tensor("new_feature")
    ds.extend({"new_feature": feature_data})
    ds.delete_tensor("old_feature")
    ds.commit("Refactored dataset schema")
```

---

### ds.checkout()

#### Overview

Checkout a specific commit or branch. This changes the dataset state to match the specified version.

#### Parameters

- **address** (`str`): The commit ID or branch name to checkout.
- **create** (`bool`, optional): If `True`, creates a new branch at the specified commit. Defaults to `False`.
- **reset** (`bool`, optional): If checkout fails because the target branch HEAD is corrupted, reset uncommitted HEAD changes and retry. Defaults to `False`.

#### Returns

- **Optional[str]**: The committed commit ID after checkout, or `None` if there is no committed snapshot yet.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Checkout a specific commit
ds.checkout("abc123def456")

# Checkout a branch
ds.checkout("main")
ds.checkout("feature-branch")

# Create and checkout a new branch
ds.checkout("new-feature", create=True)

# Checkout previous commit
commits = ds.commits()
previous_commit = commits[1]["commit"]
ds.checkout(previous_commit)

# Checkout and make changes
ds.checkout("experiment", create=True)
with ds:
    ds.append({"data": experimental_data})
    ds.commit("Experimental changes")
```

---

### ds.commits()

#### Overview

Get a list of all commits in the dataset history.

#### Parameters

- **ordered_by_date** (`bool`, optional): If `True`, orders commits by date instead of by commit graph. Defaults to `False`.

#### Returns

- **List[Dict]**: List of commit dictionaries containing commit information.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Get all commits
commits = ds.commits()
for commit in commits:
    print(f"{commit['commit']}: {commit['message']}")

# Get commits ordered by date
commits = ds.commits(ordered_by_date=True)

# Access commit details
latest_commit = commits[0]
print(f"Author: {latest_commit.get('author')}")
print(f"Time: {latest_commit.get('time')}")
print(f"Message: {latest_commit.get('message')}")

# Find specific commit
target_message = "Added validation data"
for commit in commits:
    if target_message in commit.get("message", ""):
        print(f"Found commit: {commit['commit']}")
        break

# Get commit count
print(f"Total commits: {len(commits)}")
```

---

### ds.merge()

#### Overview

Merge changes from another branch into the current branch. This combines the history and changes from two branches.

#### Parameters

- **target_id** (`str`): Commit ID or branch name to merge into the current branch.
- **append_resolution** (`str`, optional): Strategy for append conflicts. Must be `None`, `"ours"`, `"theirs"`, or `"both"`. Defaults to `None`, which raises if append conflicts exist.
- **update_resolution** (`str`, optional): Strategy for update conflicts. Must be `None`, `"ours"`, or `"theirs"`. Defaults to `None`, which raises if update conflicts exist.
- **pop_resolution** (`str`, optional): Strategy for pop conflicts. Must be `None`, `"ours"`, `"theirs"`, or `"both"`. Defaults to `None`, which raises if pop conflicts exist.
- **delete_removed_tensors** (`bool`, optional): If `True`, tensors removed by the merge are deleted from the dataset. Defaults to `False`.
- **force** (`bool`, optional): Force merge through certain rename-related conflicts. Defaults to `False`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Merge a feature branch into main
ds.checkout("main")
ds.merge("feature-branch")

# Merge with conflict resolution
ds.checkout("main")
ds.merge("experiment", append_resolution="ours", update_resolution="theirs")

# Merge workflow
ds.checkout("feature", create=True)
with ds:
    ds.append({"data": new_data})
    ds.commit("Added feature data")

ds.checkout("main")
ds.merge("feature")
ds.commit("Merged feature branch")
```

---

### ds.detect_merge_conflict()

#### Overview

Detect merge conflicts between the current branch HEAD and a target commit or branch before calling `ds.merge()`.

#### Parameters

- **target_id** (`str`): Commit ID or branch name to compare against the current branch.
- **show_value** (`bool`, optional): If `True`, include conflicting values in the returned records. Defaults to `False`.

#### Returns

- **Tuple[set, dict]**: Conflict tensor names and per-tensor conflict records. Hidden `_uuid` conflict records are omitted from the returned visible records.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")
ds.checkout("main")

conflict_tensors, conflict_records = ds.detect_merge_conflict("feature-branch", show_value=True)
if conflict_tensors:
    print(conflict_records)
```

---

### ds.diff()

#### Overview

Show differences between the current state and a previous version, or between two commit IDs or branch names.

#### Parameters

- **id_1** (`str`, optional): First commit ID or branch name. If omitted with `id_2`, compares the current state with the previous commit. Defaults to `None`.
- **id_2** (`str`, optional): Second commit ID or branch name. Providing `id_2` without `id_1` raises `ValueError`. Defaults to `None`.
- **as_dict** (`bool`, optional): If `True`, return structured diff records. If `False`, print a readable diff and return `{}`. Defaults to `False`.
- **show_value** (`bool`, optional): If `True`, include appended, updated, and deleted values in the diff output. Defaults to `False`.
- **offset** (`int`, optional): Number of value records to skip when `show_value=True`. Defaults to `0`.
- **limit** (`int`, optional): Maximum number of value records to return when `show_value=True`. Defaults to `None`.
- **asrow** (`bool`, optional): If `True`, return shown values row-wise when tensor changes are aligned. Defaults to `False`.

#### Returns

- **Optional[Dict]**: Structured diff when `as_dict=True`; otherwise prints the diff and returns `{}`.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Show uncommitted changes
ds.diff()

# Compare with specific commit
diff = ds.diff("abc123def456", as_dict=True)

# Compare two commits
diff = ds.diff("commit1", "commit2", as_dict=True)

# Include values for a limited window
diff = ds.diff("main", "experiment", as_dict=True, show_value=True, offset=0, limit=100)
```

---

### ds.direct_diff()

#### Overview

Compute the direct difference of `id_2` compared with `id_1`. Unlike `ds.diff()`, this returns direct added/removed columns and modified row records between the two resolved versions.

#### Parameters

- **id_1** (`str`, optional): First commit ID or branch name.
- **id_2** (`str`, optional): Second commit ID or branch name.
- **as_dataframe** (`bool`, optional): If `True`, return pandas DataFrames for added columns, removed columns, added rows, removed rows, and edited rows. Defaults to `False`.
- **force** (`bool`, optional): If `as_dataframe=True`, allow DataFrame export beyond the safety limit. Defaults to `False`.

#### Returns

- **Dict**: Direct diff result. With `as_dataframe=False`, contains `added_columns`, `removed_columns`, and `modified_records`.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Direct diff from main to an experiment branch
result = ds.direct_diff("main", "experiment")
print(result["added_columns"])

# Return DataFrames for inspection
frames = ds.direct_diff("commit1", "commit2", as_dataframe=True, force=True)
```

---

### ds.diff_to_prev()

#### Overview

Show differences between a given commit, or the current commit when omitted, and its previous commit.

#### Parameters

- **commit_id** (`str`, optional): Commit ID to compare with its previous commit. If omitted, uses the current commit node. Defaults to `None`.
- **as_dict** (`bool`, optional): If `True`, return structured diff records. If `False`, print a readable diff and return `{}`. Defaults to `False`.
- **show_value** (`bool`, optional): If `True`, include appended, updated, and deleted values. Defaults to `False`.
- **offset** (`int`, optional): Number of value records to skip when `show_value=True`. Defaults to `0`.
- **limit** (`int`, optional): Maximum number of value records to return when `show_value=True`. Defaults to `None`.
- **asrow** (`bool`, optional): If `True`, return shown values row-wise when tensor changes are aligned. Defaults to `False`.

#### Returns

- **Optional[Dict]**: Structured diff when `as_dict=True`; otherwise prints the diff and returns `{}`.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Diff current commit node to its previous commit
ds.diff_to_prev()

# Return structured diff for a specific commit
diff = ds.diff_to_prev("abc123def456", as_dict=True, show_value=True)
```

---

### ds.reset()

#### Overview

Reset uncommitted changes on the current branch. This does not reset to an arbitrary historical commit.

#### Parameters

- **force** (`bool`, optional): If `True`, run reset even when no uncommitted changes are detected. Defaults to `False`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Discard uncommitted changes on the current branch
ds.reset()

# Force reset even if no HEAD changes are detected
ds.reset(force=True)
```

#### Warning

Reset permanently discards uncommitted data from the underlying storage. Use with caution.

---

### ds.log()

#### Overview

Display the commit history in a readable format, similar to `git log`.

#### Parameters

- **ordered_by_date** (`bool`, optional): If `True`, display commits sorted by date from newest to oldest. If `False`, display them in graph traversal order. Defaults to `False`.

#### Returns

- **None** (prints to console)

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Show full commit log
ds.log()

# Show commits ordered by date
ds.log(ordered_by_date=True)
```

### Example Output

```
---------------
MULLER_F Version Log
---------------

Current Branch: main
Commit : abc123def456 (main)
Author : John Doe
Time   : 2026-03-04 10:30:00
Message: Added validation dataset

Commit : 789ghi012jkl (main)
Author : Jane Smith
Time   : 2026-03-03 15:20:00
Message: Initial dataset creation
```

---

### ds.delete_branch()

#### Overview

Delete a branch from the dataset. The branch must not be the currently checked out branch.

#### Parameters

- **name** (`str`): Name of the branch to delete.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Delete a merged branch
ds.delete_branch("old-feature")

# Delete multiple branches
for branch in ["temp1", "temp2", "test-branch"]:
    ds.delete_branch(branch)

# Safe branch cleanup
branches = ds.branches
current_branch = ds.branch
for branch in branches:
    if branch != current_branch and branch.startswith("temp-"):
        ds.delete_branch(branch)
```

#### Warning

Deleting a branch removes its branch reference from the dataset version metadata.

---

### ds.branch

#### Overview

Property that returns the current branch name.

#### Type

- **str**: Current branch name.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

print(f"Current branch: {ds.branch}")
ds.checkout("experiment", create=True)
print(ds.branch)
```

---

### ds.branches

#### Overview

Property that returns branch metadata stored in the dataset version state.

#### Type

- **dict**: Mapping of branch name to metadata such as `based_on` and `create_time`, or `"Not Supported"` if branch metadata is unavailable.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Get all branches
branches = ds.branches
print(f"Available branches: {branches}")

# Check if branch exists
if "feature-branch" in ds.branches:
    ds.checkout("feature-branch")

# List all branch names
for branch in ds.branches.keys():
    print(f"- {branch}")

# Create branch if it doesn't exist
branch_name = "new-feature"
if branch_name not in ds.branches:
    ds.checkout(branch_name, create=True)
```

---

### ds.commit_id

#### Overview

Property that returns the latest committed commit ID visible from the current dataset state. At a branch HEAD with uncommitted changes, this is the parent committed commit ID. If there are no committed snapshots, it returns `None`.

#### Type

- **str**: The current commit ID, or `None` if no commits exist.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Get current commit ID
current_commit = ds.commit_id
print(f"Current commit: {current_commit}")

# Check if dataset has commits
if ds.commit_id is None:
    print("No commits yet")
else:
    print(f"On commit: {ds.commit_id}")

# Save commit ID before making changes
original_commit = ds.commit_id
with ds:
    ds.append({"data": new_data})
    ds.commit("Added data")

# Compare commits
print(f"Changed from {original_commit} to {ds.commit_id}")

# Use in version tracking
version_info = {
    "commit": ds.commit_id,
    "branch": ds.branch,
    "timestamp": datetime.now()
}
```

---

## Version Control Workflow Examples

### Basic Workflow

```python
import muller

# Load dataset
ds = muller.load("./my_dataset")

# Make changes
with ds:
    ds.append({"images": image, "labels": label})

# Commit changes
ds.commit("Added new sample")

# View history
ds.log()
```

### Branching Workflow

```python
import muller

ds = muller.load("./my_dataset")

# Create feature branch
ds.checkout("feature-augmentation", create=True)

# Make changes on feature branch
with ds:
    ds.create_tensor("augmented_images")
    ds.extend({"augmented_images": augmented_data})
    ds.commit("Added augmented images")

# Switch back to main and merge
ds.checkout("main")
ds.merge("feature-augmentation")
ds.commit("Merged augmentation feature")

# Clean up
ds.delete_branch("feature-augmentation")
```

### Experimentation Workflow

```python
import muller

ds = muller.load("./my_dataset")

# Save current state
original_commit = ds.commit_id

# Create experiment branch
ds.checkout("experiment-1", create=True)

# Try experimental changes
with ds:
    ds.append({"data": experimental_data})
    ds.commit("Experimental changes")

# If experiment fails, go back
ds.checkout("main")
ds.delete_branch("experiment-1")

# If experiment succeeds, merge
ds.checkout("main")
ds.merge("experiment-1")
```

### Collaborative Workflow

```python
import muller

# User A: Create feature
ds = muller.load("./shared_dataset")
ds.checkout("user-a-feature", create=True)
with ds:
    ds.append({"data": data_a})
    ds.commit("User A: Added feature data")

# User B: Create different feature
ds = muller.load("./shared_dataset")
ds.checkout("user-b-feature", create=True)
with ds:
    ds.append({"data": data_b})
    ds.commit("User B: Added feature data")

# Merge both features
ds.checkout("main")
ds.merge("user-a-feature")
ds.merge("user-b-feature")
ds.commit("Merged features from User A and B")
```
