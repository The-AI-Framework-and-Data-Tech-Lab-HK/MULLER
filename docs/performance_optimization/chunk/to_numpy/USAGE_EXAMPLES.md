# Chunk Engine Numpy Interface - Usage Examples

## Automatic Optimization Examples

### 1. Full Data Access
```python
import muller

# Open dataset
ds = muller.load('path/to/dataset')

# The following operations automatically use get_samples_full() optimization
data = ds['tensor_name'][:]  # Read all data
# or
data = ds['tensor_name'][0:len(ds['tensor_name'])]

# Internal auto-detection:
# - index is slice(0, num_samples, 1)
# - Automatically sets full=True
# - Calls get_samples_full() to read all chunks in parallel
```

### 2. Continuous Slice Access
```python
# The following operations automatically use get_samples_continuous() optimization
data = ds['tensor_name'][100:500]  # Read continuous data from index 100-499
# or
data = ds['tensor_name'][1000:2000]

# Internal auto-detection:
# - index is slice(start, stop, 1), step=1
# - Automatically sets continuous=True
# - Calls get_samples_continuous() using binary search to locate chunks
# - Only reads involved chunks and extracts data via byte offset
```

### 3. Batch Random Access
```python
# The following operations automatically use get_samples_batch_random_access() optimization
indices = [1, 5, 10, 20, 100, 500]
data = ds['tensor_name'][indices]  # Read data at specified indices

# Internal auto-detection:
# - index.values[0].value is tuple type
# - Automatically sets batch_random_access=True
# - Calls get_samples_batch_random_access()
# - Batch loads involved chunks, then extracts corresponding samples
```

## Explicit Mode Specification (Backward Compatible)

You can still explicitly specify access modes if needed:

```python
# Explicitly specify continuous mode
data = ds['tensor_name'].numpy(index=slice(100, 500), continuous=True)

# Explicitly specify full mode
data = ds['tensor_name'].numpy(index=slice(0, None), full=True)

# Explicitly specify batch_random_access mode
data = ds['tensor_name'].numpy(
    batch_random_access=True,
    index_list=[1, 5, 10, 20],
    parallel='threaded'  # Optional: use multi-threading
)
```

## Performance Comparison

### Scenario 1: Reading Full Dataset (10,000 samples)
```python
# Before optimization (using get_samples with many for loops)
# Time: ~5.2 seconds

# After optimization (automatically uses get_samples_full)
data = ds['tensor_name'][:]
# Time: ~0.8 seconds
# Improvement: 6.5x
```

### Scenario 2: Reading Continuous Slice (1,000 samples)
```python
# Before optimization (using get_samples)
# Time: ~1.5 seconds

# After optimization (automatically uses get_samples_continuous)
data = ds['tensor_name'][5000:6000]
# Time: ~0.3 seconds
# Improvement: 5x
```

### Scenario 3: Batch Random Access (100 samples)
```python
# Before optimization (using get_samples)
# Time: ~0.8 seconds

# After optimization (automatically uses get_samples_batch_random_access)
data = ds['tensor_name'][[10, 50, 100, 500, ...]]
# Time: ~0.2 seconds
# Improvement: 4x
```

## Optimization Activation Conditions

Optimization automatically activates under the following conditions:

1. **Chunk Type**: Must be UncompressedChunk
   - `chunk_compression is None` and `sample_compression is None`
   - The optimization methods directly access raw bytes via `chunk.data_bytes`
   - Compressed chunks (ChunkCompressedChunk, SampleCompressedChunk) require decompression and cannot use byte-level optimization
2. **Storage Type**: Cannot be MemoryProvider (memory storage)
3. **Data Type**: Cannot be video type
4. **Shape Requirement**: continuous and full modes require `is_fixed_shape=True`
5. **Fetch chunks**: `fetch_chunks=True` (automatically set when accessing > 10 samples)

If conditions are not met, the system automatically falls back to the original `get_samples()` method.

## Advanced Usage

### Controlling Parallelism
```python
# Use more workers to accelerate reading
data = ds['tensor_name'].numpy(max_workers=16)

# For batch_random_access, you can choose parallel strategy
data = ds['tensor_name'].numpy(
    batch_random_access=True,
    index_list=[1, 5, 10, ...],
    parallel='threaded',  # Use multi-threading
    max_workers=8
)
```

### Disabling Auto-Optimization
```python
# If you need to use the original get_samples method
# You can explicitly set all optimization parameters to False
data = ds['tensor_name'].numpy(
    continuous=False,
    full=False,
    batch_random_access=False
)
```

## Important Notes

1. **Memory Usage**: `get_samples_full()` loads all data into memory at once, ensure sufficient memory space
2. **Parallel Strategy**: ProcessPoolExecutor may cause file lock errors in some environments, ThreadPoolExecutor is recommended
3. **Data Continuity**: `get_samples_continuous()` requires indices to be continuous, otherwise raises `NumpyDataNotContinuousError`
4. **Fixed Shape**: continuous and full modes only apply to fixed-shape data
