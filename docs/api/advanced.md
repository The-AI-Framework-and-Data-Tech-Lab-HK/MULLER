# Advanced Features

This page documents advanced features including data transformation pipelines, rechunking operations, and administrative functions.

## Table of Contents

### Data Transformation
- [muller.compute()](#mullercompute)
- [muller.ComputeFunction](#mullercomputefunction)
- [muller.Pipeline](#mullerpipeline)

### Rechunking
- [ds.rechunk()](#dsrechunk)
- [ds.rechunk_if_necessary()](#dsrechunk_if_necessary)

### Administrative
- [ds.enable_admin_mode()](#dsenable_admin_mode)
- [ds.disable_admin_mode()](#dsdisable_admin_mode)
- [ds.lock()](#dslock)
- [ds.unlock()](#dsunlock)

---

## Data Transformation

### muller.compute()

#### Overview

Decorator for creating data transformation functions that can be applied to datasets. This enables parallel processing and efficient data pipelines.

#### Parameters

- **func** (`callable`): The function to decorate. Should take input data and output dataset as parameters.

#### Returns

- **ComputeFunction**: A wrapped function that can be evaluated on datasets.

#### Examples

```python
import muller
import numpy as np

# Define a transformation function
@muller.compute
def preprocess_images(sample_in, sample_out):
    # Normalize images
    image = sample_in["images"]
    normalized = (image - image.mean()) / image.std()
    sample_out["images"].append(normalized)
    sample_out["labels"].append(sample_in["labels"])
    return sample_out

# Apply transformation
source_ds = muller.load("./raw_dataset")
target_ds = muller.dataset("./processed_dataset", overwrite=True)
target_ds.create_tensor("images")
target_ds.create_tensor("labels")

with target_ds:
    preprocess_images().eval(
        source_ds,
        target_ds,
        num_workers=4,
        progressbar=True
    )

# Transformation with augmentation
@muller.compute
def augment_data(sample_in, sample_out):
    image = sample_in["images"]
    label = sample_in["labels"]

    # Original
    sample_out["images"].append(image)
    sample_out["labels"].append(label)

    # Flipped
    sample_out["images"].append(np.fliplr(image))
    sample_out["labels"].append(label)

    return sample_out

# Apply augmentation
with target_ds:
    augment_data().eval(source_ds, target_ds, num_workers=4)

# Filtering transformation
@muller.compute
def filter_high_quality(sample_in, sample_out):
    if sample_in["quality_score"] > 0.8:
        sample_out["images"].append(sample_in["images"])
        sample_out["labels"].append(sample_in["labels"])
    return sample_out
```

---

### muller.ComputeFunction

#### Overview

A class representing a compute function that can be evaluated on datasets. Created by the `@muller.compute` decorator.

#### Methods

- **eval()**: Evaluate the compute function on input data.

#### Parameters for eval()

- **data_in**: Input data (Dataset or iterable).
- **ds_out** (`Dataset`): Output dataset.
- **num_workers** (`int`, optional): Number of parallel workers. Defaults to `0`.
- **scheduler** (`str`, optional): Scheduler type ("threaded", "processed", "serial"). Defaults to `"threaded"`.
- **progressbar** (`bool`, optional): Show progress bar. Defaults to `True`.
- **skip_ok** (`bool`, optional): Skip samples that cause errors. Defaults to `False`.
- **ignore_errors** (`bool`, optional): Continue processing on errors. Defaults to `False`.

#### Examples

```python
import muller

@muller.compute
def transform_data(sample_in, sample_out):
    sample_out["processed"].append(sample_in["raw"] * 2)
    return sample_out

source_ds = muller.load("./source")
target_ds = muller.dataset("./target", overwrite=True)
target_ds.create_tensor("processed")

# Evaluate with different configurations
with target_ds:
    # Serial processing
    transform_data().eval(source_ds, target_ds, num_workers=0)

    # Parallel with threads
    transform_data().eval(
        source_ds,
        target_ds,
        num_workers=4,
        scheduler="threaded"
    )

    # Parallel with processes
    transform_data().eval(
        source_ds,
        target_ds,
        num_workers=4,
        scheduler="processed"
    )

    # With error handling
    transform_data().eval(
        source_ds,
        target_ds,
        num_workers=4,
        skip_ok=True,
        ignore_errors=True
    )
```

---

### muller.Pipeline

#### Overview

Create a pipeline of multiple transformation functions that are applied sequentially.

#### Parameters

- **functions** (`List[ComputeFunction]`): List of compute functions to chain.

#### Returns

- **Pipeline**: A pipeline object that can be evaluated.

#### Examples

```python
import muller
import numpy as np

# Define transformation steps
@muller.compute
def normalize(sample_in, sample_out):
    image = sample_in["images"]
    normalized = (image - image.mean()) / image.std()
    sample_out["images"].append(normalized)
    sample_out["labels"].append(sample_in["labels"])
    return sample_out

@muller.compute
def resize(sample_in, sample_out):
    from PIL import Image
    image = sample_in["images"]
    resized = np.array(Image.fromarray(image).resize((224, 224)))
    sample_out["images"].append(resized)
    sample_out["labels"].append(sample_in["labels"])
    return sample_out

@muller.compute
def augment(sample_in, sample_out):
    image = sample_in["images"]
    # Original
    sample_out["images"].append(image)
    sample_out["labels"].append(sample_in["labels"])
    # Flipped
    sample_out["images"].append(np.fliplr(image))
    sample_out["labels"].append(sample_in["labels"])
    return sample_out

# Create pipeline
from muller.core.transform import Pipeline

pipeline = Pipeline([normalize, resize, augment])

# Apply pipeline
source_ds = muller.load("./raw_data")
target_ds = muller.dataset("./processed_data", overwrite=True)
target_ds.create_tensor("images")
target_ds.create_tensor("labels")

with target_ds:
    pipeline.eval(
        source_ds,
        target_ds,
        num_workers=4,
        progressbar=True
    )
```

---

## Rechunking

### ds.rechunk()

#### Overview

Reorganize the dataset's chunk structure for better storage efficiency and access patterns. This is useful after many modifications or when optimizing for specific access patterns.

#### Parameters

- **tensors** (`List[str]`, optional): List of tensor names to rechunk. If not provided, rechunks all tensors. Defaults to `None`.
- **num_workers** (`int`, optional): Number of parallel workers. Defaults to `0`.
- **progressbar** (`bool`, optional): Show progress bar. Defaults to `True`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Rechunk all tensors
ds.rechunk()

# Rechunk specific tensors
ds.rechunk(tensors=["images", "labels"])

# Rechunk with parallel workers
ds.rechunk(num_workers=4, progressbar=True)

# Rechunk after many modifications
with ds:
    for i in range(1000):
        ds.append({"data": i})
    ds.commit("Added 1000 samples")

# Optimize storage
ds.rechunk(num_workers=4)
ds.commit("Rechunked for optimization")
```

#### When to Use

- After many small append operations
- After deleting many samples
- When optimizing for sequential vs random access
- Before sharing or archiving the dataset

---

### ds.rechunk_if_necessary()

#### Overview

Automatically rechunk the dataset if certain conditions are met (e.g., fragmentation exceeds threshold). This is a smart version of `rechunk()` that only rechunks when beneficial.

#### Parameters

- **tensors** (`List[str]`, optional): List of tensor names to check and rechunk. Defaults to `None`.
- **threshold** (`float`, optional): Fragmentation threshold (0-1) above which rechunking occurs. Defaults to `0.3`.

#### Returns

- **bool**: `True` if rechunking was performed, `False` otherwise.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Automatically rechunk if needed
was_rechunked = ds.rechunk_if_necessary()
if was_rechunked:
    print("Dataset was rechunked")

# Check specific tensors
ds.rechunk_if_necessary(tensors=["images"])

# Use custom threshold
ds.rechunk_if_necessary(threshold=0.5)  # More lenient

# In a maintenance routine
def optimize_dataset(ds):
    if ds.rechunk_if_necessary():
        ds.commit("Auto-rechunked for optimization")
    ds.flush()

optimize_dataset(ds)
```

---

## Administrative

### ds.enable_admin_mode()

#### Overview

Enable administrative mode for the dataset. This grants elevated permissions for operations that are normally restricted.

#### Parameters

None

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Enable admin mode
ds.enable_admin_mode()

# Perform administrative operations
with ds:
    # Operations that require admin privileges
    ds.delete_tensor("sensitive_data", large_ok=True)
    ds.reshard_index("labels", num_shards=20)

# Disable when done
ds.disable_admin_mode()
```

#### Warning

Admin mode bypasses certain safety checks. Use with caution and disable when not needed.

---

### ds.disable_admin_mode()

#### Overview

Disable administrative mode and return to normal operation mode.

#### Parameters

None

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Enable admin mode for specific operations
ds.enable_admin_mode()

try:
    # Perform admin operations
    ds.delete_tensor("old_tensor", large_ok=True)
finally:
    # Always disable admin mode
    ds.disable_admin_mode()

# Check admin mode status
if ds.is_admin_mode:
    ds.disable_admin_mode()
```

---

### ds.lock()

#### Overview

Acquire a write lock on the dataset to prevent concurrent modifications. This is useful in multi-process or distributed environments.

#### Parameters

- **timeout** (`int`, optional): Maximum time to wait for lock in seconds. Defaults to `0` (no wait).

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Acquire lock
ds.lock()

try:
    # Perform operations with exclusive access
    with ds:
        ds.append({"data": new_data})
        ds.commit("Added data with lock")
finally:
    # Always release lock
    ds.unlock()

# Lock with timeout
try:
    ds.lock(timeout=30)  # Wait up to 30 seconds
    with ds:
        ds.extend({"data": batch_data})
        ds.commit("Batch update")
finally:
    ds.unlock()
```

---

### ds.unlock()

#### Overview

Release the write lock on the dataset, allowing other processes to acquire it.

#### Parameters

None

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Lock and unlock pattern
ds.lock()
try:
    with ds:
        ds.append({"data": data})
        ds.commit("Update")
finally:
    ds.unlock()

# Context manager pattern (recommended)
class DatasetLock:
    def __init__(self, dataset):
        self.ds = dataset

    def __enter__(self):
        self.ds.lock()
        return self.ds

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ds.unlock()

# Usage
with DatasetLock(ds):
    with ds:
        ds.append({"data": data})
        ds.commit("Safe update")
```

---

## Advanced Workflow Examples

### Parallel Data Processing Pipeline

```python
import muller
import numpy as np

# Define transformation pipeline
@muller.compute
def preprocess(sample_in, sample_out):
    image = sample_in["raw_image"]
    # Normalize
    normalized = (image - image.mean()) / image.std()
    sample_out["image"].append(normalized)
    sample_out["label"].append(sample_in["label"])
    return sample_out

@muller.compute
def augment(sample_in, sample_out):
    image = sample_in["image"]
    label = sample_in["label"]

    # Original
    sample_out["image"].append(image)
    sample_out["label"].append(label)

    # Augmented versions
    sample_out["image"].append(np.fliplr(image))
    sample_out["label"].append(label)

    sample_out["image"].append(np.flipud(image))
    sample_out["label"].append(label)

    return sample_out

# Process data
source = muller.load("./raw_data")
processed = muller.dataset("./processed_data", overwrite=True)
processed.create_tensor("image")
processed.create_tensor("label")

with processed:
    # Apply preprocessing
    preprocess().eval(source, processed, num_workers=8, progressbar=True)
    processed.commit("Preprocessed data")

    # Rechunk for efficiency
    processed.rechunk(num_workers=4)
    processed.commit("Rechunked")

# Create augmented version
augmented = muller.dataset("./augmented_data", overwrite=True)
augmented.create_tensor("image")
augmented.create_tensor("label")

with augmented:
    augment().eval(processed, augmented, num_workers=8, progressbar=True)
    augmented.commit("Augmented data")
```

### Dataset Maintenance

```python
import muller

def maintain_dataset(path):
    """Perform routine maintenance on a dataset."""
    ds = muller.load(path)

    # Enable admin mode for maintenance
    ds.enable_admin_mode()

    try:
        # Rechunk if necessary
        if ds.rechunk_if_necessary():
            print("Dataset was rechunked")
            ds.commit("Maintenance: rechunked")

        # Optimize indexes
        for tensor_name in ds.indexed_tensors:
            ds.optimize_index(tensor_name)
            print(f"Optimized index for {tensor_name}")

        # Flush all changes
        ds.flush()

        print("Maintenance completed successfully")

    finally:
        # Always disable admin mode
        ds.disable_admin_mode()

# Run maintenance
maintain_dataset("./my_dataset")
```

### Safe Concurrent Access

```python
import muller
from contextlib import contextmanager

@contextmanager
def locked_dataset(path, timeout=30):
    """Context manager for safe concurrent dataset access."""
    ds = muller.load(path)
    ds.lock(timeout=timeout)
    try:
        yield ds
    finally:
        ds.unlock()

# Use in concurrent environment
with locked_dataset("./shared_dataset") as ds:
    with ds:
        ds.append({"data": new_data})
        ds.commit("Concurrent update")
```
