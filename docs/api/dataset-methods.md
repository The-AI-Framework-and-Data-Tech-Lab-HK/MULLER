# Dataset Core Methods

This page documents core instance methods for Dataset objects, including data operations, tensor management, cache control, and basic information retrieval.

## Table of Contents

### Data Operations
- [ds.append()](#dsappend)
- [ds.extend()](#dsextend)
- [ds.update()](#dsupdate)
- [ds.pop()](#dspop)
- [ds.delete()](#dsdelete)
- [ds.rename()](#dsrename)

### Tensor Management
- [ds.create_tensor()](#dscreate_tensor)
- [ds.create_tensor_like()](#dscreate_tensor_like)
- [ds.delete_tensor()](#dsdelete_tensor)
- [ds.rename_tensor()](#dsrename_tensor)

### Cache and Flush
- [ds.flush()](#dsflush)
- [ds.maybe_flush()](#dsmaybe_flush)

### Basic Information
- [ds.summary()](#dssummary)
- [ds.info](#dsinfo)
- [ds.statistics()](#dsstatistics)
- [ds.size_approx()](#dssize_approx)
- [ds.num_samples](#dsnum_samples)
- [ds.tensors](#dstensors)

---

## Data Operations

### ds.append()

#### Overview

Append a single sample to the dataset. Each sample is a dictionary where keys are tensor names and values are the data to append.

#### Parameters

- **sample** (`Dict[str, Any]`): Dictionary mapping tensor names to their values.
- **skip_ok** (`bool`, optional): If `True`, tensors missing from `sample` are skipped instead of raising an error. Defaults to `False`.
- **append_empty** (`bool`, optional): If `True`, tensors missing from `sample` receive an empty value so tensor lengths stay aligned. Defaults to `False`.

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

### ds.extend()

#### Overview

Extend the dataset with multiple samples at once. This is more efficient than calling `append()` multiple times.

#### Parameters

- **samples** (`Dict[str, Any]` or `Dataset`): Data to append. Usually a dictionary where keys are tensor names and values are lists, arrays, tensors, or other indexable sequences with the same length. A `Dataset` can also be passed; its tensors are used as the source data.
- **skip_ok** (`bool`, optional): If `True`, tensors missing from `samples` are skipped instead of raising an error. Defaults to `False`.
- **append_empty** (`bool`, optional): If `True`, tensors missing from `samples`, or receiving fewer values than other tensors, are padded with empty values. Defaults to `False`.
- **ignore_errors** (`bool`, optional): If `True`, continues processing per-sample appends when one sample fails. This is ignored when extending from numpy arrays or tensors. Defaults to `False`.
- **progressbar** (`bool`, optional): If `True`, displays a progress bar for per-sample extension. Defaults to `False`.

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

### ds.update()

#### Overview

Update existing samples in the dataset. This modifies data at specific indices.

#### Parameters

- **sample** (`Dict[str, Any]`): Dictionary mapping tensor names to their new values. `ds.update()` updates the current dataset view/index; for partial sample updates on multi-index views, update tensors directly instead.

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

### ds.pop()

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

### ds.delete()

#### Overview

Delete the dataset represented by this `Dataset` object. For local datasets this clears the underlying dataset storage; for views, the view entry is deleted.

#### Parameters

- **large_ok** (`bool`, optional): If `True`, allows deleting datasets larger than the safety threshold. Defaults to `False`.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./datasets/old_dataset")

# Delete a small dataset
ds.delete()

# Delete a large dataset after explicitly acknowledging the safety check
ds.delete(large_ok=True)
```

#### Warning

This operation is irreversible.

---

### ds.rename()

#### Overview

Rename a local dataset to another path in the same parent directory.

#### Parameters

- **path** (`str` or `pathlib.Path`): New dataset path. The implementation accepts local storage rename operations only, and the new path must stay in the same directory as the current dataset path.

#### Returns

- **None**

#### Examples

```python
import muller

ds = muller.load("./datasets/my_dataset")
ds.rename("./datasets/renamed_dataset")
```

---

## Tensor Management

### ds.create_tensor()

#### Overview

Create a new tensor in the dataset. Tensors are the columns of your dataset, each storing a specific type of data.

#### Parameters

- **name** (`str`): Name of the tensor to create. Names are normalized internally and must not conflict with Dataset attributes, Python dictionary methods, or MULLER reserved keywords.
- **htype** (`str`, optional): High-level tensor type. If omitted, MULLER uses the default `generic` htype. Confirmed htypes include `generic`, `image`, `image.rgb`, `image.gray`, `class_label`, `bbox`, `bbox.3d`, `video`, `binary_mask`, `instance_label`, `segment_mask`, `keypoints_coco`, `point`, `audio`, `text`, `json`, `list`, `dicom`, `nifti`, `point_cloud`, `intrinsics`, `polygon`, `mesh`, `embedding`, and `vector`.
- **dtype** (`str` or `np.dtype`, optional): Numpy dtype for tensor samples. If omitted, the dtype is inferred from the selected `htype` when that htype defines a default, such as `uint8` for `image`, `uint32` for `class_label`, `float32` for `bbox`, `embedding`, `point_cloud`, `intrinsics`, and `polygon`, `str` for `text`, and `Any` for `json`.
- **sample_compression** (`str` or `None`, optional): Compression applied to each sample. `image`, `image.rgb`, and `image.gray` support image compression formats and byte compression such as `lz4`; aliases like `jpg` and `jp2` are normalized internally. `dicom` defaults to `dcm`, `mesh` defaults to `ply`, and unspecified compression resolves to no sample compression unless the htype defines a default.
- **chunk_compression** (`str` or `None`, optional): Compression applied at chunk level instead of individual samples. Uses supported MULLER compression names such as `lz4` where valid for the tensor configuration. Defaults to `None`.
- **hidden** (`bool`, optional): If `True`, creates a hidden tensor that is tracked in dataset metadata but not exposed as a normal public tensor. Defaults to `False`.
- **exist_ok** (`bool`, optional): If `True`, returns an existing tensor when its current configuration matches the requested configuration. Defaults to `False`.
- **create_sample_info_tensor** (`bool`, optional): If `True`, creates a hidden sample-info tensor for supported media htypes (`image`, `audio`, `video`, `dicom`, `point_cloud`, `mesh`, `nifti`). Defaults to `False`.
- **create_shape_tensor** (`bool`, optional): If `True`, creates a hidden shape tensor for htypes other than `text` and `json`. Defaults to `False`.
- **create_id_tensor** (`bool`, optional): If `True`, creates a hidden per-sample id tensor. Defaults to `False`.
- **downsampling** (`tuple[int, int]`, optional): Creates hidden downsampled tensors for image and mask htypes. The tuple is `(downsampling_factor, number_of_layers)`.
- **max_chunk_size** (`int`, optional): Maximum chunk size metadata passed to the tensor engine.
- **tiling_threshold** (`int`, optional): Tiling threshold metadata passed to the tensor engine.
- Htype-specific **kwargs** confirmed by code:
  - **intrinsics** (`Any`, optional): Stored in tensor info for `image`.
  - **class_names** (`list`, optional): Stored in tensor info for `class_label` and `segment_mask`.
  - **coords** (`dict`, optional): Stored in tensor info for `bbox` and `bbox.3d`; accepted keys are `type` and `mode` for `bbox`, and `mode` for `bbox.3d`.
  - **keypoints** (`list`, optional) and **connections** (`list`, optional): Stored in tensor info for `keypoints_coco`.
  - **dimension** (`Any`, optional): Stored in tensor info for `vector`.

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

# Create a vector tensor with htype-specific metadata
ds.create_tensor("vectors", htype="vector", dtype="float32", dimension=128)

# Create an image tensor with hidden shape and sample-info tensors
ds.create_tensor(
    "images",
    htype="image",
    sample_compression="jpeg",
    create_shape_tensor=True,
    create_sample_info_tensor=True,
)
```

---

### ds.create_tensor_like()

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

### ds.delete_tensor()

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

### ds.rename_tensor()

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

### ds.flush()

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

### ds.maybe_flush()

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

### ds.summary()

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

### ds.info

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

### ds.statistics()

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

### ds.size_approx()

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

### ds.num_samples

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

### ds.tensors

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
