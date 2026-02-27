# Chunk Engine to Numpy Interface Optimization Summary

## Optimization Goal
Enable the three fast methods (`get_samples_continuous`, `get_samples_full`, `get_samples_batch_random_access`) to automatically activate during default calls without requiring users to explicitly pass parameters.

## Key Modifications

### 1. Added Auto-Detection Function `_detect_access_pattern()`
**Location**: [chunk_engine_to_numpy_interface.py:163-203](muller/core/chunk/interface/chunk_engine_to_numpy_interface.py#L163-L203)

This function automatically determines the optimal access pattern based on `index` characteristics:

- **Full Access**: When accessing the entire dataset (`start=0, stop=num_samples, step=1`)
  - Automatically uses `get_samples_full()` method
  - Use cases: `ds[:]` or `ds[0:len(ds)]`

- **Continuous Access**: When accessing a continuous slice (`step=1`)
  - Automatically uses `get_samples_continuous()` method
  - Use cases: `ds[100:200]` or `ds[50:150]`

- **Batch Random Access**: When accessing discrete indices
  - Automatically uses `get_samples_batch_random_access()` method
  - Use cases: `ds[[1, 5, 10, 20]]` or passing `index_list`

### 2. Modified `protected_numpy()` Function
**Location**: [chunk_engine_to_numpy_interface.py:77-160](muller/core/chunk/interface/chunk_engine_to_numpy_interface.py#L77-L160)

After `_validate_batch_samples()` check passes, if the user hasn't explicitly specified access mode parameters, automatically calls `_detect_access_pattern()` for detection:

```python
# Auto-detect access pattern if not explicitly specified
if not continuous and not full and not batch_random_access:
    access_pattern = _detect_access_pattern(chunk_engine, index, index_list)
    continuous = access_pattern['continuous']
    full = access_pattern['full']
    batch_random_access = access_pattern['batch_random_access']
```

### 3. Enhanced `get_samples_batch_random_access()` Function
**Location**: [chunk_engine_to_numpy_interface.py:301-339](muller/core/chunk/interface/chunk_engine_to_numpy_interface.py#L301-L339)

- Added optional `index` parameter
- Automatically extracts index list from `index` when `index_list` is not provided
- Supports tuple-type indices (e.g., `ds[[1, 5, 10]]`)

```python
# Extract index_list from index if not provided
if index_list is None:
    if index is None:
        raise ValueError("Either index or index_list must be provided")
    if isinstance(index.values[0].value, tuple):
        index_list = list(index.values[0].value)
    else:
        index_list = list(index.values[0].indices(chunk_engine.num_samples))
```

## Optimization Impact

### Previous Behavior
Users had to explicitly pass parameters to use fast methods:
```python
# Must explicitly specify
ds.numpy(continuous=True)  # Use get_samples_continuous
ds.numpy(full=True)        # Use get_samples_full
ds.numpy(batch_random_access=True, index_list=[1,5,10])  # Use batch random access
```

### Current Behavior
System automatically detects and uses optimal methods:
```python
# Auto-detection and optimization
ds[:]           # Automatically uses get_samples_full
ds[100:200]     # Automatically uses get_samples_continuous
ds[[1, 5, 10]]  # Automatically uses get_samples_batch_random_access
```

## Applicable Conditions

Optimization only takes effect when all the following conditions are met (checked by `_validate_batch_samples()`):

1. `fetch_chunks=True` (or automatically determined as True)
2. Not a video type (`not chunk_engine.is_video`)
3. Not MemoryProvider storage (`not isinstance(chunk_engine.base_storage, MemoryProvider)`)
4. **Must be UncompressedChunk** (`chunk_compression is None` and `sample_compression is None`)
   - The optimization methods directly access raw bytes via `chunk.data_bytes`
   - Compressed chunks require decompression before accessing individual samples
   - This is a critical requirement for the byte-level optimization to work correctly
5. For continuous and full modes, also requires `chunk_engine.is_fixed_shape=True`

## Backward Compatibility

- All original parameters are preserved (`continuous`, `full`, `batch_random_access`)
- Users can still explicitly specify access modes, which override auto-detection
- When optimization conditions are not met, automatically falls back to the original `get_samples()` method

## Code Standards

All modifications follow PEP8 import order conventions:
1. Standard library imports (bisect, concurrent.futures, functools, typing)
2. Third-party imports (numpy)
3. Local imports (muller.*)
