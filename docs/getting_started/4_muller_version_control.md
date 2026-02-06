## Version Control

MULLER provides Git-like commands to manage dataset changes. It works with datasets of any size and records key information about dataset evolution. For the full API reference, see [Dataset Version Control]().

Below are several key commands.

### 1. Commit

After performing write operations (add/delete/update), call `ds.commit()` to create a new version (commit ID).

Example: load a dataset, create a tensor column, and commit.

```python
>>> ds = muller.dataset("/data/muller_exp/", overwrite=True)
>>> ds.create_tensor(name="labels", htype="generic", dtype="int")
>>> first_commit_id = ds.commit(message="first commit.")
>>> first_commit_id
'firstdbf9474d461a19e9333c2fd19b46115348f'
```

You can then continue to modify the dataset and commit again.

```python
>>> ds.labels.extend([1, 2, 3, 4, 5])
>>> second_commit_id = ds.commit(message="add labels.")
>>> second_commit_id
'637adeb5232f0152b866c9a2a49e8da19f00c1da'
```

- For details, see [`dataset.commit()`]().

### 2. Checkout

Each newly created dataset starts on the `main` branch. Use `ds.checkout()` to create a branch or switch branches.

Create a new branch with `create=True`:

```python
>>> ds.checkout("dev", create=True)
>>> ds.create_tensor("categories", htype="text")
>>> ds.categories.extend(["agent", "emotion", "generation", "writing", "emotion"])
>>> ds.commit("created categories tensor column and add values.")
>>> ds.summary()  # Now dev has two tensor columns (categories and labels).
tensor     htype    shape    dtype  compression
 -------    -------  -------  -------  -------
categories   text    (5, 1)     str     None
  labels    generic  (5, 1)    int64    None
```

Switch to an existing branch:

```python
>>> ds.checkout("main")  # Switch back to main.
>>> ds.branch            # Check the current branch name.
'main'
>>> ds.summary()         # This branch has only one tensor column (labels).
tensor    htype    shape    dtype  compression
-------  -------  -------  -------  -------
labels   generic  (5, 1)    int64    None
```

- For details, see [`dataset.checkout()`]().
- For details, see [`dataset.branch`]().

### 3. Load a Specific Branch/Version and View History

Use `muller.load()` to load a specific branch or commit.

Load the latest commit on `main` (default):

```python
ds = muller.load(path="/data/muller_exp/")
```

Load a specific branch using `@{branch_name}`:

```python
ds = muller.load(path="/data/muller_exp@dev")
```

Load a specific commit using `@{commit_id}`:

```python
ds = muller.load(path="/data/muller_exp@3e49cded62b6b335c74ff07e97f8451a37aca7b2")
```

`ds.log()` prints the log to the console and returns commit records starting from the current version.

```python
>>> ds = muller.load(path="/data/muller_exp@dev")
>>> ds.log()
---------------
MULLER Version Log
---------------

Current Branch: dev
Commit : 476be1766915dfe652f39e39f5370f9c04528df9 (dev)
Author : public
Time   : 2025-02-28 08:07:48
Message: created categories tensor column and add values.

Commit : 637adeb5232f0152b866c9a2a49e8da19f00c1da (main)
Author : public
Time   : 2025-02-28 08:03:16
Message: add labels.

Commit : firstdbf9474d461a19e9333c2fd19b46115348f (main)
Author : public
Time   : 2025-02-28 03:32:53
Message: first commit.
```

- For details, see [`dataset.log()`]().
- If needed, we may extend `log()` to also return the history of commits merged into the current branch.
- Other related APIs:
  - Get commit details: [`dataset.get_commit_details(commit_id)`]()
  - Get current commit ID: [`dataset.commit_id`]()
  - Get the next (pending) commit ID: [`dataset.pending_commit_id`]()
  - Check whether the current branch has uncommitted changes: [`dataset.has_head_changes`]()
  - List all commits (without printing): [`dataset.commits`]()
  - List all branches: [`dataset.branches`]()
  - Get commits between a version and a branch: [`dataset_commits_between()`]()

### 4. Direct Diff (available in v0.6.10+)

Understanding changes between versions is critical. MULLER provides `ds.direct_diff(id_1, id_2)` to compute the *direct* per-tensor, per-row differences between `id_1` and `id_2` (in the direction “from `id_1` to `id_2`”). It can optionally return a pandas DataFrame for inspection.

Parameters (note that the order matters):

- **id_1 (str)**: the first version ID or branch name.
- **id_2 (str)**: the second version ID or branch name.
- **as_dataframe (bool, optional)**: if `True`, return DataFrame(s).
- **force (bool, optional)**: when the resulting DataFrame is large (over 100,000 rows by default), set `force=True` to confirm execution.

Example:

```python
import muller

ds = muller.dataset(path="temp_test", overwrite=True)

ds.create_tensor(name="labels", htype="generic", dtype="int")
ds.create_tensor(name="categories", htype="text")
ds.create_tensor(name="test1", htype="generic", dtype="int")
ds.labels.extend([0, 1, 2, 3, 4])
ds.categories.extend(["a", "b", "c", "d", "e"])
ds.test1.extend([100, 101, 102, 103, 104])

ds.checkout("dev", create=True)
ds.labels[0] = 11
ds.categories[3] = "haha"
ds.pop(2)
ds.labels.append(100)
ds.categories.append("hello")
ds.test1.append(105)
dev_1 = ds.commit("first on dev")

ds.checkout("main", create=False)
ds.checkout("dev_2", create=True)
ds.delete_tensor("test1")
ds.labels.extend([5, 6])
ds.categories.extend(["f", "g"])
ds.commit("first on dev_2")
ds.labels[1] = 111
ds.categories[1] = "xixixi"
ds.create_tensor(name="test2", htype="generic", dtype="int")
ds.test2.extend([100, 101, 102, 103, 104, 105, 106])
dev_2 = ds.commit("sec on dev_2")

final_df_dict = ds.direct_diff(dev_1, dev_2, as_dataframe=True)
```

In a Jupyter environment, you can view the changes visually.
![image]()

### 5. Diff

`ds.diff()` computes differences across versions for each tensor column and each sample (row). This API primarily supports merge computation, so its output is **not** a simple “absolute diff”; it returns diffs **per commit** relative to the most recent common ancestor (a version-tree style output, similar to `log()`).

Notes:

- If you call `diff()` while you are on one of the versions being compared *and* your current state is the branch HEAD, then **uncommitted HEAD changes** will also be included.
- By default, `diff()` returns **row indices** only. To return actual values, set `show_value=True`. Since loading values requires I/O, consider using `offset` and `limit` for large datasets, especially when the data is stored remotely.
- Use `as_dict=True` to return diffs as a Python dict.

Common usages:

1) Diff between the current state and the previous commit:

```python
ds.diff()  # If nothing changed since the last commit, nothing will be returned.
```

2) Diff between the current state and `id_1`:

```python
ds.diff(id_1=<commit_id>)      # id_1 can be a commit ID
ds.diff(id_1=<branch_name>)    # id_1 can be a branch name (its latest commit)
```

3) Diff between `id_1` and `id_2`:

```python
ds.diff(id_1=<commit_id_1>, id_2=<commit_id_2>)          # commit IDs
ds.diff(id_1=<branch_name_1>, id_2=<branch_name_2>)      # branch names (latest commits)
```

Key parameters:

- **id_1 (str, optional)**: first version ID or branch name.
- **id_2 (str, optional)**: second version ID or branch name.
- **as_dict (bool, optional)**: if `True`, return structured diffs as Python objects.
- **show_value (bool)**: if `True`, return actual values for append/update/pop.
- **offset (int)**: number of items to skip (effective only when `show_value=True`).
- **limit (int)**: maximum number of items to show (effective only when `show_value=True`, default 1000).
- **asrow (bool)**: return values row-wise (list of dicts) when `show_value=True`. This is only applicable when tensor-count and row-count changes are exactly aligned between the two versions; otherwise it may raise an error. In general, prefer `asrow=False` for flexibility.

Example (based on the dataset created in Section 5.2): show indices only.

```python
>>> ds.diff("dev", "main")  # default: do not show values
## MULLER Diff
The 2 diffs are calculated relative to the most recent common ancestor (637adeb5232f0152b866c9a2a49e8da19f00c1da) of the two commits passed.
------------------------------------------------------------------------------------------------------------------------
Diff in dev (target id 1):

********************************************************************************
commit UNCOMMITTED HEAD
Author: public
Date: None
Message: None

No changes were made in this commit.
********************************************************************************
commit 476be1766915dfe652f39e39f5370f9c04528df9
Author: public
Date: 2025-02-28 08:07:48
Message: created categories tensor column and add values.

categories
* Created tensor
* Added 5 samples: [0-5]

------------------------------------------------------------------------------------------------------------------------
Diff in main (target id 2):

********************************************************************************
commit UNCOMMITTED HEAD
Author: public
Date: None
Message: None

No changes were made in this commit.
------------------------------------------------------------------------------------------------------------------------
{}
>>>
```

Example: show actual values.

```python
>>> ds.diff(id_1=third_commit_id, id_2=first_commit_id, show_value=True)
## MULLER Diff
The 2 diffs are calculated relative to the most recent common ancestor (firstdbf9474d461a19e9333c2fd19b46115348f) of the two commits passed.
------------------------------------------------------------------------------------------------------------------------
Diff in 51f5bb41c4a1d74d881269366e3718285b8145db (target id 1):

********************************************************************************
commit 51f5bb41c4a1d74d881269366e3718285b8145db
Author: public
Date: 2025-03-03 04:03:13
Message: created categories tensor column and add values.

categories
* Created tensor
* Added 5 samples: [0-5],
The appended data values are {'created': True, 'cleared': False, 'info_updated': False, 'data_added': [0, 5], 'data_updated': set(), 'data_deleted': SortedSet([]), 'data_deleted_ids': [], 'data_transformed_in_place': False, 'add_value': [array(['agent'], dtype='<U5'), array(['emotion'], dtype='<U7'), array(['generation'], dtype='<U10'), array(['writing'], dtype='<U7'), array(['emotion'], dtype='<U7')], 'updated_values': [], 'data_deleted_values': []}

********************************************************************************
commit 6536edb5ee077de334d6f23133bcf7e209d1be80
Author: public
Date: 2025-03-03 04:03:12
Message: add labels.

labels
* Added 5 samples: [0-5],
The appended data values are {'created': False, 'cleared': False, 'info_updated': False, 'data_added': [0, 5], 'data_updated': set(), 'data_deleted': SortedSet([]), 'data_deleted_ids': [], 'data_transformed_in_place': False, 'add_value': [array([1]), array([2]), array([3]), array([4]), array([5])], 'updated_values': [], 'data_deleted_values': []}

------------------------------------------------------------------------------------------------------------------------
Diff in firstdbf9474d461a19e9333c2fd19b46115348f (target id 2):

No changes were made.

------------------------------------------------------------------------------------------------------------------------
{}
```

Example: return values as a dict.

```python
>>> ds.diff(id_1=third_commit_id, id_2=first_commit_id, as_dict=True, show_value=True)
{'dataset': ([{'commit_id': '51f5bb41c4a1d74d881269366e3718285b8145db',
    'author': 'public',
    'message': 'created categories tensor column and add values.',
    'date': '2025-03-03 04:03:13',
    'info_updated': False,
    'renamed': OrderedDict(),
    'deleted': [],
    'commit_diff_exist': True,
    'tensor_info_updated': False},
   {'commit_id': '6536edb5ee077de334d6f23133bcf7e209d1be80',
    'author': 'public',
    'message': 'add labels.',
    'date': '2025-03-03 04:03:12',
    'info_updated': False,
    'renamed': OrderedDict(),
    'deleted': [],
    'commit_diff_exist': True,
    'tensor_info_updated': False}],
  []),
 'tensor': ([{'commit_id': '51f5bb41c4a1d74d881269366e3718285b8145db',
    'categories': {'created': True,
     'cleared': False,
     'info_updated': False,
     'data_added': [0, 5],
     'data_updated': set(),
     'data_deleted': SortedSet([]),
     'data_deleted_ids': [],
     'data_transformed_in_place': False,
     'add_value': [array(['agent'], dtype='<U5'), array(['emotion'], dtype='<U7'), array(['generation'], dtype='<U10'), array(['writing'], dtype='<U7'), array(['emotion'], dtype='<U7')],
     'updated_values': [],
     'data_deleted_values': []},
    'labels': {'created': False,
     'cleared': False,
     'info_updated': False,
     'data_added': [0, 0],
     'data_updated': set(),
     'data_deleted': set(),
     'data_deleted_ids': [],
     'data_transformed_in_place': False,
     'add_value': [],
     'updated_values': [],
     'data_deleted_values': []}},
   {'commit_id': '6536edb5ee077de334d6f23133bcf7e209d1be80',
    'labels': {'created': False,
     'cleared': False,
     'info_updated': False,
     'data_added': [0, 5],
     'data_updated': set(),
     'data_deleted': SortedSet([]),
     'data_deleted_ids': [],
     'data_transformed_in_place': False,
     'add_value': [array([1]), array([2]), array([3]), array([4]), array([5])],
     'updated_values': [],
     'data_deleted_values': []}}],
  [])}
```

- For details, see [`dataset.diff()`](). For deeper background, see [Detailed MR](https://codehub-y.huawei.com/GTN/MULLER/merge_requests/157).

### 6. What Are HEAD Changes? How to Use `reset()` to Revert Uncommitted Changes

Unlike Git, MULLER version control has **no local staging area**. All changes are immediately synced to the persistent storage location (local or remote). As a result, any dataset change updates the current branch HEAD node immediately. Uncommitted changes do not appear on other branches, but they remain accessible on the current branch until reverted.

In the example below, we return to `main`, which has a `labels` tensor with 5 samples. We then append one sample and use `ds.has_head_changes` to confirm there are HEAD changes.

```python
>>> ds = muller.dataset(path="temp_dataset/", overwrite=True)
>>> with ds:
...     ds.create_tensor("labels", htype="generic")
...     ds.labels.extend([1, 2, 3, 4, 5])
>>> ds.commit()
>>> ds.checkout("dev", create=True)
>>> ds.checkout("main")
>>> with ds:
...     ds.labels.append(100)
>>> ds.has_head_changes
True
```

On the `dev` branch, the `labels` tensor still has 5 samples.

```python
>>> ds.checkout("dev")
>>> print("Dataset in {} branch has {} samples in the labels tensor".format(ds.branch, len(ds.labels)))
Dataset in dev branch has 5 samples in the labels tensor
```

Switch back to `main`, and the uncommitted change is still present (6 samples).

```python
>>> ds.checkout("main")
>>> print("Dataset in {} branch has {} samples in the labels tensor".format(ds.branch, len(ds.labels)))
Dataset in main branch has 6 samples in the labels tensor
>>> ds.has_head_changes
True
```

Use `ds.reset()` to revert uncommitted changes on the current branch.

```python
>>> ds.reset()
>>> ds.has_head_changes
False
>>> print("Dataset in {} branch has {} samples in the labels tensor".format(ds.branch, len(ds.labels)))
Dataset in main branch has 5 samples in the labels tensor
```

- For details, see [`dataset.reset()`]().

### 7. Merge (available in v0.6.7+)

Branch merging is essential for collaborative workflows. Similar to Git, MULLER supports two merge patterns:

- **Fast-forward merge (no conflict)**: if `main` has not moved since creating the feature branch, merging can fast-forward without creating a new merge commit.
  ![image]()
- **3-way merge (may conflict)**: if both branches have diverged, MULLER may need to create a new merge commit and may require conflict resolution.
  ![image]()

In multi-branch development, 3-way merges are common. For example: after creating `dev_1` and `dev_2` from `main`, if `main` merges `dev_1` first, then `main` changes and may conflict with `dev_2`. In other words, after `ds.merge("dev_1")`, merging `dev_2` into `main` can become a 3-way merge.

![image]()

MULLER currently follows the merge workflow below:

![image]()

1) Use `ds.detect_merge_conflict(target_id="...")` to preview possible conflicts. High level:

- Find the most recent common ancestor (e.g., `M1`) between the current branch HEAD (e.g., `M3`) and the target branch head (e.g., `D4`).
- Compute diffs `D4` vs `M1` and `M3` vs `M1`.
- For tensors common to both branches, compare each sample by its UUID to detect conflicts. If conflicts exist, they are printed and also cached (in-memory) for subsequent merge steps.

Conflict types recorded:

- **pop conflicts**: both sides performed `pop`, but on different rows (popping the same row does not conflict).
- **update conflicts**: both sides updated the same row (updating different rows does not conflict).
- **append conflicts**: both sides appended new rows.

2) Merge with `ds.merge(target_id="...")`. You can choose conflict-resolution strategies:

- **append_resolution**: `None` (default, raise), `"ours"`, `"theirs"`, `"both"`
- **update_resolution**: `None` (default, raise), `"ours"`, `"theirs"`
- **pop_resolution**: `None` (default, raise), `"ours"`, `"theirs"`, `"both"`

Additional merge parameters:

- **delete_removed_tensors (bool, default False)**: if `True`, tensors deleted in the current-branch diff are discarded.
- **force (bool, default False)**: relaxes certain rename-related constraints; for example, it may register a renamed tensor as a new tensor if the counterpart tensor is missing on the other side, or merge tensors under certain rename collisions.

Important notes:

- Because `pop` and `append` change dataset length and global indices, conflict detection/resolution is UUID-based.
- The execution order of conflict resolutions affects the final result.
- Be careful to distinguish “conflict” vs “no conflict”. Example: if `main` has 20 rows, you branch to `dev`, delete row 2 in `dev`, then merge `dev` into `main`, this conflicts because row 2 still exists in `main`. You may use `pop_resolution="theirs"` to accept the deletion from `dev`.

Example:

```python
# Create a dataset and add data on main
>>> ds = muller.dataset(path="temp_test", overwrite=True)
>>> ds.create_tensor(name="labels", htype="generic", dtype="int")
>>> ds.labels.extend([0, 1, 2, 3, 4])
>>> ds.create_tensor(name="categories", htype="text")
>>> ds.categories.extend(["a", "b", "c", "d", "e"])

# Create dev-1 and perform add/update/delete operations
>>> ds.checkout("dev-1", create=True)
>>> ds.labels.extend([50, 60, 70])
>>> ds.categories.extend(["ff", "gg", "hh"])
>>> ds.labels[3] = 30
>>> ds.pop(1)
>>> print(ds.labels.numpy())
[[ 0]
 [ 2]
 [30]
 [ 4]
 [50]
 [60]
 [70]]
>>> ds.commit()

# Back to main, create dev-2 and perform add/update/delete operations
>>> ds.checkout("main")
>>> ds.checkout("dev-2", create=True)
>>> ds.labels.extend([500, 600, 700, 800])
>>> ds.categories.extend(["fff", "ggg", "hhh", "iii"])
>>> ds.labels[3] = 300
>>> ds.labels[4] = 400
>>> ds.pop([1, 2])
>>> print(ds.labels.numpy())
[[  0]
 [300]
 [400]
 [500]
 [600]
 [700]
 [800]]
>>> ds.commit()

# Merge dev-1 into main (take dev-1's pop changes)
>>> ds.checkout("main")
>>> ds.merge("dev-1", pop_resolution="theirs")
>>> print(ds.labels.numpy())
[[ 0]
 [ 2]
 [30]
 [ 4]
 [50]
 [60]
 [70]]

# Detect conflicts between main and dev-2
>>> conflict_tensors, conflict_records = ds.detect_merge_conflict("dev-2", show_value=True)
>>> pprint(conflict_records)
{'categories': {'app_ori_idx': [4, 5, 6],
                'app_ori_values': [array(['ff'], dtype='<U2'),
                                   array(['gg'], dtype='<U2'),
                                   array(['hh'], dtype='<U2')],
                'app_tar_idx': [3, 4, 5, 6],
                'app_tar_values': [array(['fff'], dtype='<U3'),
                                   array(['ggg'], dtype='<U3'),
                                   array(['hhh'], dtype='<U3'),
                                   array(['iii'], dtype='<U3')],
                'del_ori_idx': [],
                'del_ori_values': [],
                'del_tar_idx': [2],
                'del_tar_values': [array(['c'], dtype='<U1')],
                'update_values': {'update_ori': [], 'update_tar': []}},
 'labels': {'app_ori_idx': [4, 5, 6],
            'app_ori_values': [array([50]), array([60]), array([70])],
            'app_tar_idx': [3, 4, 5, 6],
            'app_tar_values': [array([500]),
                               array([600]),
                               array([700]),
                               array([800])],
            'del_ori_idx': [],
            'del_ori_values': [],
            'del_tar_idx': [2],
            'del_tar_values': [array([2])],
            'update_values': {'update_ori': [{2: array([30])}],
                              'update_tar': [{1: array([300])}]}}}

# Merge dev-2 into main with explicit resolutions
>>> ds.merge("dev-2", append_resolution="both", pop_resolution="ours", update_resolution="theirs")
>>> print(ds.labels.numpy())
[[  0]
 [  2]
 [300]
 [400]
 [ 50]
 [ 60]
 [ 70]
 [500]
 [600]
 [700]
 [800]]
```

- For details, see [`dataset.detect_merge_conflict()`]().
- For details, see [`dataset.merge()`]().

### 8. Branch Permission Control for the Huashan Platform (available in v0.6.6+)

To meet permission-control requirements on the Huashan platform, the following restrictions apply:

1. Only the user who created the `main` branch has write/delete permissions on **all branches**.
2. Only the `main` creator can delete or rename a dataset via:

```python
ds.delete()
ds.rename()
```

3. Non-creators only have write/delete permissions on **branches they created**, and read-only access on other branches. The following APIs are restricted to “your own branch”:

```python
ds.reset()
ds.create_tensor()
ds.create_tensor_like()
ds.commit()
ds.extend([dataset_object])
ds.<tensor>.extend()
ds.delete_branch()
ds.rechunk()
ds.append([dataset_object])
ds.<tensor>.append
ds.update()  # there are two update methods
ds.merge()
ds.delete_tensor()
ds.pop()
ds.rename()
muller.api_dataset.create_dataset_from_dataframes()
muller.api_dataset.create_dataset_from_file()
ds.<tensor>.clear()
ds.create_index()
```

4. Users can only delete search views they created:

```python
muller.delete_view()
```

- Implementation note: in practice, non-creators have write permission to one file on `main` (`version_control_info.json`) that stores version metadata.
