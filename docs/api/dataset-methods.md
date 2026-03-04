# Dataset Core Methods

This page documents core instance methods for Dataset objects, including data operations, tensor management, cache control, and basic information retrieval.

## Table of Contents

### Data Operations
- [append()](#append)
- [extend()](#extend)
- [update()](#update)
- [pop()](#pop)

### Tensor Management
- [create_tensor()](#create_tensor)
- [create_tensor_like()](#create_tensor_like)
- [delete_tensor()](#delete_tensor)
- [rename_tensor()](#rename_tensor)

### Cache and Flush
- [flush()](#flush)
- [maybe_flush()](#maybe_flush)

### Basic Information
- [summary()](#summary)
- [info](#info)
- [statistics()](#statistics)
- [size_approx()](#size_approx)
- [num_samples](#num_samples)
- [tensors](#tensors)

---

## Data Operations

### append()

#### Overview

Append a single sample to the dataset. Each sample is a dictionary where keys are tensor names and values are the data to append.

#### Parameters

- **sample** (`Dict[str, Any]`): Dictionary mapping tensor names to their values.
- **skip_ok** (`bool`, optional): If `True`, allows skipping samples that cause errors. Defaults to `False`.
- **append_empty** (`bool`, optional): If `True`, allows appending empty samples. Defaults to `False`.

#### Returns

- **None**

#### Examples

```python
import muller
import numpy as np

ds = muller.dataset("./my_dataset", overwrite=True)
ds.create_tensor("images")
ds.create_tensor("labels")

# Append a single sample
with ds:
    ds.append({
        "images": np.random.rand(224, 224, 3),
        "labels": 1
    })

# Append multiple samples one by one
with ds:
    for i in range(10):
        ds.append({
            "images": np.random.rand(224, 224, 3),
            "labels": i % 5
        })

# Append with skip_ok to ignore errors
with ds:
    ds.append({"images": invalid_data, "labels": 0}, skip_ok=True)
```

---

### extend()

#### Overview

Extend the dataset with multiple samples at once. This is more efficient than calling `append()` multiple times.

#### Parameters

- **samples** (`Dict[str, Any]`): Dictionary where keys are tensor names and values are lists/arrays of data to append.
- **skip_ok** (`bool`, optional): If `True`, allows skipping samples that cause errors. Defaults to `False`.
- **append_empty** (`bool`, optional): If `True`, allows appending empty samples. Defaults to `False`.
- **ignore_errors** (`bool`, optional): If `True`, continues processing even if errors occur. Defaults to `False`.
- **progressbar** (`bool`, optional): If `True`, displays a progress bar. Defaults to `False`.

#### Returns

- **None**

#### Examples

```python
import muller
import numpy as np

ds = muller.dataset("./my_dataset", overwrite=True)
ds.create_tensor("images")
ds.create_tensor("labels")

# Extend with multiple samples
images_data = [np.random.rand(224, 224, 3) for _ in range(100)]
labels_data = [i % 5 for i in range(100)]

with ds:
    ds.extend({
        "images": images_data,
        "labels": labels_data
    })

# Extend with progress bar
with ds:
    ds.extend({
        "images": images_data,
        "labels": labels_data
    }, progressbar=True)

# Extend with error handling
with ds:
    ds.extend({
        "images": images_data,
        "labels": labels_data
    }, ignore_errors=True, skip_ok=True)
```

---

### update()

#### Overview

Update existing samples in the dataset. This modifies data at specific indices.

#### Parameters

- **sample** (`Dict[str, Any]`): Dictionary mapping tensor names to their new values.

#### Returns

- **None**

#### Examples

```python
import muller
import numpy as np

ds = muller.load("./my_dataset")

# Update a specific sample
with ds:
    ds[0].update({
        "images": np.random.rand(224, 224, 3),
        "labels": 5
    })

# Update multiple samples
with ds:
    for i in range(10):
        ds[i].update({"labels": i * 2})
```

---

### pop()

#### Overview

Remove samples from the dataset at specified indices. If no index is provided, removes the last sample.

#### Parameters

- **index** (`int`, `List[int]`, optional): Index or list of indices to remove. If `None`, removes the last sample. Defaults to `None`.
- **rechunk** (`bool`, optional): If `True`, rechunks the dataset after removal for better storage efficiency. Defaults to `False`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Remove the last sample
with ds:
    ds.pop()

# Remove a specific sample
with ds:
    ds.pop(5)

# Remove multiple samples
with ds:
    ds.pop([0, 2, 4, 6])

# Remove and rechunk for efficiency
with ds:
    ds.pop([10, 20, 30], rechunk=True)
```

---

## Tensor Management

### create_tensor()

#### Overview

Create a new tensor in the dataset. Tensors are the columns of your dataset, each storing a specific type of data.

#### Parameters

- **name** (`str`): Name of the tensor to create.
- **htype** (`str`, optional): The htype (high-level type) of the tensor (e.g., "image", "class_label", "text"). If not specified, defaults to "generic".
- **dtype** (`str` or `np.dtype`, optional): The numpy dtype of the tensor data. If not specified, it's inferred from htype.
- **sample_compression** (`str`, optional): Compression to apply to each sample (e.g., "jpeg", "png"). Defaults to `None`.
- **chunk_compression** (`str`, optional): Compression to apply to chunks. Defaults to `None`.
- **hidden** (`bool`, optional): If `True`, creates a hidden tensor. Defaults to `False`.
- **kwargs**: Additional keyword arguments for tensor configuration.

#### Returns

- **Tensor**: The created tensor object.

#### Examples

```python
import muller

ds = muller.dataset("./my_dataset", overwrite=True)

# Create a generic tensor
ds.create_tensor("data")

# Create an image tensor with JPEG compression
ds.create_tensor("images", htype="image", sample_compression="jpeg")

# Create a text tensor
ds.create_tensor("descriptions", htype="text", dtype="str")

# Create a class label tensor
ds.create_tensor("labels", htype="class_label", dtype="int32")

# Create a bounding box tensor
ds.create_tensor("boxes", htype="bbox", dtype="float32")

# Create with chunk compression
ds.create_tensor("embeddings", htype="embedding", chunk_compression="lz4")
```

---

### create_tensor_like()

#### Overview

Create a new tensor by copying the metadata and configuration from an existing tensor. No samples are copied, only the structure.

#### Parameters

- **name** (`str`): Name of the new tensor to create.
- **source** (`Tensor`): The source tensor to copy metadata from.

#### Returns

- **Tensor**: The created tensor object.

#### Examples

```python
import muller

# Load source dataset
source_ds = muller.load("./source_dataset")

# Create new dataset with similar structure
new_ds = muller.dataset("./new_dataset", overwrite=True)

# Create tensor with same configuration as source
with new_ds:
    new_ds.create_tensor_like("images", source_ds["images"])
    new_ds.create_tensor_like("labels", source_ds["labels"])

# Copy tensor structure within same dataset
with new_ds:
    new_ds.create_tensor_like("images_copy", new_ds["images"])
```

---

### delete_tensor()

#### Overview

Delete a tensor from the dataset. This permanently removes the tensor and all its data.

#### Parameters

- **name** (`str`): Name of the tensor to delete.
- **large_ok** (`bool`, optional): If `True`, allows deletion of large tensors. Defaults to `False`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Delete a tensor
with ds:
    ds.delete_tensor("old_tensor")

# Delete a large tensor
with ds:
    ds.delete_tensor("large_tensor", large_ok=True)

# Delete multiple tensors
with ds:
    for tensor_name in ["temp1", "temp2", "temp3"]:
        ds.delete_tensor(tensor_name)
```

#### Warning

This operation is irreversible. All data in the tensor will be permanently deleted.

---

### rename_tensor()

#### Overview

Rename a tensor in the dataset.

#### Parameters

- **name** (`str`): Current name of the tensor.
- **new_name** (`str`): New name for the tensor.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Rename a tensor
with ds:
    ds.rename_tensor("old_name", "new_name")

# Rename multiple tensors
with ds:
    ds.rename_tensor("img", "images")
    ds.rename_tensor("lbl", "labels")
```

---

## Cache and Flush

### flush()

#### Overview

Flush all pending changes to storage. This ensures that all modifications are written to disk.

#### Parameters

None

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Make changes and flush
with ds:
    ds.append({"images": data, "labels": label})
    ds.flush()

# Flush after batch operations
with ds:
    for i in range(1000):
        ds.append({"data": i})
        if i % 100 == 0:
            ds.flush()  # Periodic flush
```

---

### maybe_flush()

#### Overview

Conditionally flush changes to storage if certain conditions are met (e.g., cache is full). This is called automatically by MULLER but can be invoked manually.

#### Parameters

None

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Manual conditional flush
with ds:
    for i in range(10000):
        ds.append({"data": i})
        ds.maybe_flush()  # Flushes only when needed
```

---

## Basic Information

### summary()

#### Overview

Display a summary of the dataset including tensor names, shapes, dtypes, and sample counts.

#### Parameters

- **force** (`bool`, optional): If `True`, forces regeneration of the summary even if cached. Defaults to `False`.

#### Returns

- **None** (prints summary to console)

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Display dataset summary
ds.summary()

# Force regenerate summary
ds.summary(force=True)
```

#### Example Output

```
Dataset(path='./my_dataset', tensors=['images', 'labels'])

  tensor      htype        shape       dtype  compression
 -------    --------    ----------    -------  -----------
  images     image     (224, 224, 3)   uint8      jpeg
  labels   class_label      ()         int32      None

  Total samples: 1000
```

---

### info

#### Overview

Property that provides access to dataset metadata and information. This can be read and modified.

#### Type

- **DatasetInfo**: Object containing dataset metadata.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Read dataset info
print(ds.info)
print(ds.info.description)

# Modify dataset info
ds.info.description = "My custom dataset"
ds.info.author = "John Doe"

# Access custom metadata
ds.info.custom_field = "custom value"
```

---

### statistics()

#### Overview

Get statistical information about the dataset, including storage size, number of chunks, and other metrics.

#### Parameters

None

#### Returns

- **dict**: Dictionary containing dataset statistics.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Get dataset statistics
stats = ds.statistics()
print(stats)

# Access specific statistics
print(f"Total size: {stats['total_size']} bytes")
print(f"Number of chunks: {stats['num_chunks']}")
```

---

### size_approx()

#### Overview

Get an approximate size of the dataset in bytes. This is faster than computing the exact size.

#### Parameters

None

#### Returns

- **int**: Approximate size in bytes.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Get approximate size
size = ds.size_approx()
print(f"Dataset size: {size / (1024**3):.2f} GB")

# Compare multiple datasets
datasets = ["./ds1", "./ds2", "./ds3"]
for path in datasets:
    ds = muller.load(path)
    size = ds.size_approx()
    print(f"{path}: {size / (1024**2):.2f} MB")
```

---

### num_samples

#### Overview

Property that returns the number of samples in the dataset.

#### Type

- **int**: Number of samples.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Get number of samples
print(f"Dataset has {ds.num_samples} samples")

# Use in loops
for i in range(ds.num_samples):
    sample = ds[i]
    # Process sample

# Check if dataset is empty
if ds.num_samples == 0:
    print("Dataset is empty")
```

---

### tensors

#### Overview

Property that returns a dictionary of all tensors in the dataset.

#### Type

- **Dict[str, Tensor]**: Dictionary mapping tensor names to Tensor objects.

#### Examples

```python
import muller

ds = muller.load("./my_dataset")

# Get all tensor names
tensor_names = list(ds.tensors.keys())
print(f"Tensors: {tensor_names}")

# Iterate over tensors
for name, tensor in ds.tensors.items():
    print(f"{name}: {tensor.shape}, {tensor.dtype}")

# Access specific tensor
images_tensor = ds.tensors["images"]
print(images_tensor.meta)

# Check if tensor exists
if "labels" in ds.tensors:
    print("Labels tensor exists")
```
