# Batch Enable Mode Optimization for `@muller.compute`

## Overview

This document summarizes the bug fixes and optimizations made to the `batch_enable=True` mode in the `@muller.compute` decorator for parallel data import.

## Bug Fixes

### 1. `transform_tensor.py` - Fixed batch mode append logic

**File:** `muller/core/transform/transform_tensor.py`

**Problem:** The original code had several issues:
- Called `self.non_numpy_only()` at the beginning of `append()`, which set `self.numpy_only = False`, making the subsequent `if self.numpy_only` check always false
- Incorrectly iterated over individual samples and called `chunk_engine.extend()` on each one, but `extend()` expects a list, not a single element like `numpy.uint32`
- Called non-existent method `self.dataset.rollback()` instead of `self.dataset._rollback()`

**Solution:** Simplified the batch mode logic to directly call `chunk_engine.extend(item)` with the entire list of samples:

```python
def append(self, item):
    if self.is_batch:
        updated_tensor = 0
        try:
            chunk_engine = self.dataset.all_chunk_engines[self.name]
            # In batch mode, item is always a list of samples
            # Directly extend with the entire list
            chunk_engine.extend(
                item,
                pg_callback=self.dataset.pg_callback,
                is_uuid=(self.name == DATASET_UUID_NAME)
            )
            updated_tensor = len(item)
            # Update batch_samples_written for the first non-uuid tensor
            if self.name != DATASET_UUID_NAME and self.dataset.batch_samples_written == 0:
                self.dataset.batch_samples_written = updated_tensor
        except Exception as e:
            self.dataset._rollback({self.name: updated_tensor}, [])
            ...
```

### 2. `transform_dataset.py` - Fixed `__len__` returning 0 in batch mode

**File:** `muller/core/transform/transform_dataset.py`

**Problem:** In batch mode, data is written directly to `chunk_engine`, so `TransformTensor.items` remains empty. This caused `len(dataset)` to return 0, which in turn caused `append_uuid()` to generate 0 UUIDs, triggering `TransformError at index 0`.

**Solution:** Added `batch_samples_written` counter to track the number of samples written in batch mode:

```python
class TransformDataset:
    def __init__(self, ...):
        ...
        self.batch_samples_written = 0  # Track samples written in batch mode

    def __len__(self):
        if self.is_batch and self.batch_samples_written > 0:
            return self.batch_samples_written
        return max(len(self[tensor]) for tensor in self.data)
```

### 3. `transform.py` - Fixed duplicate `cache_size` parameter

**File:** `muller/core/transform/transform.py`

**Problem:** `cache_size` was passed both as an explicit parameter and through `**kwargs`, causing `TypeError: got multiple values for keyword argument 'cache_size'`.

**Solution:** Removed the explicit `cache_size` parameter, letting it pass through `**kwargs` only:

```python
self.run(
    data_in=temp_data_in,
    target_ds=target_ds,
    ...
    # Removed: cache_size=kwargs.get("cache_size", DEFAULT_TRANSFORM_SAMPLE_CACHE_SIZE),
    **kwargs,
)
```

## Optimizations

### 1. `pipeline.py` - Added flush call and fixed progress bar

**File:** `muller/core/transform/pipeline.py`

**Changes:**
- Added `transform_dataset.flush()` in `finally` block for consistency with `_extend_data_slice`
- Fixed progress bar to update once after batch processing completes instead of per-sample updates

```python
def _batch_extend_data_slice(data_slice, offset, transform_dataset, pipeline, pg_callback):
    extend_fn, args, kwargs = pipeline.func, pipeline.args, pipeline.kwargs
    # In batch mode, progress is updated after all samples are written
    batch_pg_callback = pg_callback
    transform_dataset.set_pg_callback(None)  # Disable per-sample progress updates
    try:
        extend_fn(*data_slice, transform_dataset, *args, **kwargs)
        append_uuid(transform_dataset)
        # Update progress after batch processing completes
        if batch_pg_callback is not None:
            num_samples = len(transform_dataset)
            batch_pg_callback(num_samples)
    except Exception as e:
        raise TransformError(offset, suggest=isinstance(e, SampleAppendError), is_batch=True) from e
    finally:
        # Flush to ensure all data is written to storage
        transform_dataset.flush()
```

## Test Cases Added

**File:** `tests/test_dataset.py`

Added comprehensive test cases for `batch_enable=True` mode:

1. `test_batch_enable_optimized_pattern` - Tests the recommended pattern with optimized parameters (`scheduler="processed"`, `cache_size=64`, `disable_rechunk=True`)

2. `test_batch_enable_with_numpy_arrays` - Tests batch mode with numpy arrays using threaded scheduler

3. `test_batch_enable_single_worker` - Tests batch mode with single worker (serial mode)

4. `test_batch_enable_processed_scheduler` - Tests batch mode with multiprocessing scheduler

## Recommended Usage Pattern

```python
@muller.compute(batch_enable=True)
def file_to_muller_optimized(images, labels, sample_out):
    sample_out.images.append(images)
    sample_out.labels.append(labels)
    return sample_out

with ds:
    # Preprocess data
    images = [muller.read(path) for path in image_paths]
    labels = [np.uint32(cls) for cls in label_list]

    file_to_muller_optimized().eval(
        images, labels, ds,
        num_workers=8,
        scheduler="processed",  # Use multiprocessing
        cache_size=64,          # Increase cache size
        disable_rechunk=True    # Disable rechunking
    )
```

## Files Modified

| File | Changes |
|------|---------|
| `muller/core/transform/transform_tensor.py` | Fixed batch mode append logic |
| `muller/core/transform/transform_dataset.py` | Added `batch_samples_written` counter |
| `muller/core/transform/transform.py` | Fixed duplicate `cache_size` parameter |
| `muller/core/transform/pipeline.py` | Added flush call and fixed progress bar |
| `tests/test_dataset.py` | Added batch mode test cases |
