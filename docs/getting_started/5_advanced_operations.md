## Advanced Operations

#### 7.1. Recommendations for Ingesting Large-Scale Data into MULLER Datasets

If you need to create a dataset or append a large amount of data in the MULLER format, it is recommended to use the `@muller.compute` decorator for parallel ingestion (typically set `num_workers` to 8–32 depending on available resources).

> Notes:
>
> - Multiprocessing/multithreading has overhead; the speedup becomes noticeable mainly for hundreds of thousands to millions of samples.

Example:

```python
def create_cifar10_dataset_parallel(num_workers=4, scheduler="threaded"):
    ds_multi = muller.dataset(path="./temp_test", overwrite=True)
    with ds_multi:
        ds_multi.create_tensor("test1", htype="text")
        ds_multi.create_tensor("test2", htype="text")

    # Add data row-by-row to preserve row-level atomicity
    iter_dict = []
    for i in range(0, 100000):
        iter_dict.append((i, ("hi", "hello")))  # Example only; load any data in practice

    @muller.compute
    def file_to_muller(data_pair, sample_out):
        sample_out.test1.append(data_pair[1][1])
        sample_out.test2.append(data_pair[1][0])
        return sample_out

    with ds_multi:
        file_to_muller().eval(
            iter_dict,
            ds_multi,
            num_workers=num_workers,
            scheduler=scheduler,
            disable_rechunk=True,
        )

    return ds_multi


if __name__ == "__main__":
    ds = create_cifar10_dataset_parallel(num_workers=4, scheduler="processed")
```

For large-scale ingestion, `eval()` also supports `checkpoint_interval=<commit_every_N_samples>`, which checkpoints to disk every N samples to reduce rework after unexpected interruptions. Internally, data is written before metadata; if a crash occurs mid-write, the system can resume from the previous checkpoint instead of restarting from the first sample.

> Note: in this case, versions are stored under the `/versions` directory.

With large datasets, not every sample path is guaranteed to be valid (e.g., invalid paths, wrong file formats such as treating PNG as JPEG). You may choose to ignore such errors via `.eval(..., ignore_errors=True)`; otherwise frequent exception handling can significantly slow down ingestion.

- For details, see `[muller.compute]()`.
- For details, see `[eval()]()`.

#### 7.2. Use `with` for Better Write Performance

1. In MULLER, each independent update is pushed to the target persistent storage immediately (through an LRU cache; see `_set_item()` and `flush()`). If you have many small updates and the data is stored remotely, write time can increase significantly. For example, the following pattern pushes an update on every `.append()`:

```python
for i in range(10):
    ds.my_tensor.append(i)
```

1. Using a `with` block typically improves performance. Updates are batched and flushed when the `with` block completes (or when the local cache is full), reducing fragmented writes:

```python
with ds:
    for i in range(10):
        ds.my_tensor.append(i)  # or other write operations: create, update, etc.
```

#### 7.3. Why a Dataset Become Corrupted, and How to Recover?

If your program is interrupted unexpectedly (e.g., a crash during append/pop), the dataset may become inconsistent: some tensors may have been updated while others were not. In such cases, you can use `ds.reset()` to roll back illegal, uncommitted operations and return to the most recent valid commit.

1. **Scenario A:** The dataset (or some tensors) cannot be read (e.g., you see an error like below).

```text
DatasetCorruptError: Exception occured (see Traceback). The dataset maybe corrupted. Try using `reset=True` to reset HEAD changes and load the previous commit. This will delete all uncommitted changes on the branch you are trying to load.
```

Recovery: reload with `reset=True`.

```python
ds = muller.load(<dataset_path>, reset=True)
```

1. **Scenario B:** The dataset is corrupted (e.g., tensor lengths are inconsistent).

Recovery: load without integrity checking, then reset.

```python
ds = muller.load(<dataset_path>, check_integrity=False)  # skip integrity check
ds.reset()
```

- For details on `check_integrity` during load, see `[muller.dataset()]()` and `[muller.load()]()`.
- For details on reset, see `[dataset.reset()]()`.
- Note: once you reset, all uncommitted changes will be deleted.
- For large datasets, prefer **checkpointing** or **committing frequently** so recovery is easier after unexpected failures.

#### 7.4. Keys to Efficient Operations on OBS

1. Use a sufficiently capable OBS client and sufficient bandwidth.

- The most efficient MULLER usage is typically “local → local” or “local → OBS bucket”. If you use the Huashan production environment with the `huashan://` prefix (or an implicit remote path), the call chain may become “local → OBS bucket A (personal) → OBS bucket B (shared)”. Frequent reads/writes via limited OBS APIs between buckets can significantly impact performance.
- We also expect richer low-level OBS APIs (e.g., batch read/write, batch delete, offset-based partial read/write) to improve small-file transfer efficiency. References: [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html), [Huawei OBS](https://github.com/huaweicloud/huaweicloud-sdk-python-obs/tree/master/examples).

> Note: transferring massive numbers of small files on OBS is inherently costly. Partial read/write can help; see [Huawei OBS partial read (with offset)](https://github.com/huaweicloud/huaweicloud-sdk-python-obs/blob/master/examples/concurrent_upload_part_sample.py).

1. Use enough memory so the LRU cache can hold more data per flush. If needed, increase `DEFAULT_MEMORY_CACHE_SIZE` in `constants.py` (default: 20 GB).

> Note: (1) is typically a prerequisite for (2).

#### 7.5. Concurrency: Write Locks in MULLER

MULLER supports basic concurrent-write protection via **file locks** (including in the Huashan notebook environment). The following locks are used:

1. `version_control_info.lock`: since branch users can write to `version_control_info.json`, this lock ensures only one writer at a time; others wait until the lock is released.
2. `dataset_lock.lock`: once a dataset is created in a path, this lock is created. While it exists, creating a new dataset in the same path (e.g., via `ds.empty()`) is blocked; otherwise you may see:

```text
muller.util.exceptions.DatasetHandlerError: A dataset already exists at the given path (temp_dataset/). If you want to create a new empty dataset, either specify another path or use overwrite=True. If you want to load the dataset that exists at this path, use muller.load() instead.
```

1. `queries.lock`: used in two cases:
  - Created when `ds.save_view()` starts and released when it completes, to prevent concurrent use during view saving.
  - Created when `ds.delete_view()` starts and released when it completes, to prevent concurrent use during deletion. (Currently only the view creator can delete a view, so extreme cases are largely avoided.)

Limitations of file locks:

- Lock creation/deletion performance depends on the underlying OBS (NSP) file I/O performance.
- Lock atomicity depends on the atomicity guarantees of the underlying OBS (NSP) APIs.

MULLER also supports **Redis-based distributed locks**. Three lock types are currently supported:

1. Branch-head lock: locks the head-node resources of a branch.
2. Version-control lock: locks shared resources across dataset versions (version-control metadata).
3. Branch lock: locks predecessor versions of an entire branch (not required in v0.6.7 and earlier).

To avoid deadlocks, if you need multiple lock types, acquire them strictly in the order 1 → 2 → 3.

Note: MULLER currently has no read locks, so **dirty reads** are allowed (e.g., user B may observe partial intermediate states while user A is writing).

#### 7.6. How Is MULLER Different from Deeplake?

Deeplake is closed-source. MULLER adopts a subset of Deeplake-like interfaces but uses its own file layout and includes major performance optimizations and refactoring for version control, loading, and OBS support.

Key differences include:

1. File layout: MULLER’s file organization is self-designed for higher I/O efficiency.
2. Vectorized search acceleration: not available in Deeplake; implemented in MULLER.
3. Version control: Git-for-data style; MULLER provides self-developed `merge` and `diff` for more advanced workflows.
4. Multi-user concurrency handling and locks: implemented in MULLER.
5. Branch permission control: implemented in MULLER.
6. High-performance DataLoader: implemented in MULLER.

#### 7.7. Other

1. Fetch adjacent data in the chunk

```python
# Fetch adjacent data in the chunk -> increases speed when loading sequentially,
# or when a tensor's data fits in the cache.
numeric_label = ds.labels[i].numpy(fetch_chunks=True)
```

> Note: If `True`, full chunks will be retrieved from the storage; otherwise only required bytes will be retrieved. This will always be `True` even if specified as `False` in the following cases: (1) the tensor is ChunkCompressed; (2) the chunk being accessed has more than 128 samples.

