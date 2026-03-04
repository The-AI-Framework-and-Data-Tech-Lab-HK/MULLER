# Dataset Version Control

This page documents version control methods for managing dataset history, branches, and commits.

## Table of Contents

- [commit()](#commit)
- [checkout()](#checkout)
- [commits()](#commits)
- [merge()](#merge)
- [diff()](#diff)
- [reset()](#reset)
- [log()](#log)
- [delete_branch()](#delete_branch)
- [branches](#branches)
- [commit_id](#commit_id)

---

## commit()

### Overview

Create a commit to save the current state of the dataset. This creates a snapshot of all changes made since the last commit.

### Parameters

- **message** (`str`, optional): Commit message describing the changes. If not provided, an automatic message will be generated. Defaults to `None`.
- **allow_empty** (`bool`, optional): If `True`, allows creating a commit even when there are no changes. Defaults to `False`.

### Returns

- **str**: The commit ID of the newly created commit.

### Examples

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

## checkout()

### Overview

Checkout a specific commit, branch, or tag. This changes the dataset state to match the specified version.

### Parameters

- **address** (`str`): The commit ID, branch name, or tag to checkout.
- **create** (`bool`, optional): If `True`, creates a new branch at the specified commit. Defaults to `False`.

### Returns

- **None**

### Examples

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
previous_commit = commits[1]["commit_id"]
ds.checkout(previous_commit)

# Checkout and make changes
ds.checkout("experiment", create=True)
with ds:
    ds.append({"data": experimental_data})
    ds.commit("Experimental changes")
```

---

## commits()

### Overview

Get a list of all commits in the dataset history.

### Parameters

- **ordered_by_date** (`bool`, optional): If `True`, orders commits by date instead of by commit graph. Defaults to `False`.

### Returns

- **List[Dict]**: List of commit dictionaries containing commit information.

### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Get all commits
commits = ds.commits()
for commit in commits:
    print(f"{commit['commit_id']}: {commit['message']}")

# Get commits ordered by date
commits = ds.commits(ordered_by_date=True)

# Access commit details
latest_commit = commits[0]
print(f"Author: {latest_commit.get('author')}")
print(f"Date: {latest_commit.get('date')}")
print(f"Message: {latest_commit.get('message')}")

# Find specific commit
target_message = "Added validation data"
for commit in commits:
    if target_message in commit.get("message", ""):
        print(f"Found commit: {commit['commit_id']}")
        break

# Get commit count
print(f"Total commits: {len(commits)}")
```

---

## merge()

### Overview

Merge changes from another branch into the current branch. This combines the history and changes from two branches.

### Parameters

- **branch** (`str`): Name of the branch to merge into the current branch.
- **conflict_resolution** (`str`, optional): Strategy for resolving conflicts ("ours", "theirs", "manual"). Defaults to `"manual"`.

### Returns

- **str**: The commit ID of the merge commit.

### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Merge a feature branch into main
ds.checkout("main")
merge_commit = ds.merge("feature-branch")
print(f"Merge commit: {merge_commit}")

# Merge with conflict resolution
ds.checkout("main")
merge_commit = ds.merge("experiment", conflict_resolution="ours")

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

## diff()

### Overview

Show the differences between the current state and a specific commit, or between two commits.

### Parameters

- **commit_id** (`str`, optional): Commit ID to compare against. If not provided, compares against the last commit. Defaults to `None`.
- **other_commit_id** (`str`, optional): Second commit ID for comparing two commits. Defaults to `None`.

### Returns

- **Dict**: Dictionary containing the differences.

### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Show uncommitted changes
diff = ds.diff()
print(diff)

# Compare with specific commit
diff = ds.diff("abc123def456")

# Compare two commits
diff = ds.diff("commit1", "commit2")

# Analyze differences
diff = ds.diff()
if "tensors_added" in diff:
    print(f"Added tensors: {diff['tensors_added']}")
if "tensors_deleted" in diff:
    print(f"Deleted tensors: {diff['tensors_deleted']}")
if "samples_added" in diff:
    print(f"Added samples: {diff['samples_added']}")
```

---

## reset()

### Overview

Reset the dataset to a specific commit, discarding all changes after that commit.

### Parameters

- **commit_id** (`str`): The commit ID to reset to.
- **hard** (`bool`, optional): If `True`, discards all uncommitted changes. If `False`, keeps uncommitted changes. Defaults to `False`.

### Returns

- **None**

### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Soft reset (keep uncommitted changes)
ds.reset("abc123def456")

# Hard reset (discard all changes)
ds.reset("abc123def456", hard=True)

# Reset to previous commit
commits = ds.commits()
previous_commit = commits[1]["commit_id"]
ds.reset(previous_commit, hard=True)

# Reset and create new branch
ds.reset("abc123def456", hard=True)
ds.checkout("recovery", create=True)
```

### Warning

Hard reset permanently discards all changes after the specified commit. Use with caution.

---

## log()

### Overview

Display the commit history in a readable format, similar to `git log`.

### Parameters

- **max_count** (`int`, optional): Maximum number of commits to display. Defaults to `None` (all commits).
- **oneline** (`bool`, optional): If `True`, displays each commit on a single line. Defaults to `False`.

### Returns

- **None** (prints to console)

### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Show full commit log
ds.log()

# Show last 10 commits
ds.log(max_count=10)

# Show one-line format
ds.log(oneline=True)

# Show recent commits in one-line format
ds.log(max_count=5, oneline=True)
```

### Example Output

```
commit abc123def456
Author: John Doe
Date: 2026-03-04 10:30:00

    Added validation dataset

commit 789ghi012jkl
Author: Jane Smith
Date: 2026-03-03 15:20:00

    Initial dataset creation
```

---

## delete_branch()

### Overview

Delete a branch from the dataset. The branch must not be the currently checked out branch.

### Parameters

- **branch_name** (`str`): Name of the branch to delete.
- **force** (`bool`, optional): If `True`, deletes the branch even if it has unmerged changes. Defaults to `False`.

### Returns

- **None**

### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Delete a merged branch
ds.delete_branch("old-feature")

# Force delete an unmerged branch
ds.delete_branch("experimental", force=True)

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

### Warning

Deleting a branch with unmerged changes will permanently lose those changes unless `force=True` is used intentionally.

---

## branches

### Overview

Property that returns a list of all branches in the dataset.

### Type

- **List[str]**: List of branch names.

### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Get all branches
branches = ds.branches
print(f"Available branches: {branches}")

# Check if branch exists
if "feature-branch" in ds.branches:
    ds.checkout("feature-branch")

# List all branches
for branch in ds.branches:
    print(f"- {branch}")

# Create branch if it doesn't exist
branch_name = "new-feature"
if branch_name not in ds.branches:
    ds.checkout(branch_name, create=True)
```

---

## commit_id

### Overview

Property that returns the current commit ID of the dataset.

### Type

- **str**: The current commit ID, or `None` if no commits exist.

### Examples

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
ds.log(max_count=5)
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
ds.delete_branch("experiment-1", force=True)

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
