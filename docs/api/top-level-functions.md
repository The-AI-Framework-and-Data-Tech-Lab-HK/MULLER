# Top-Level Functions

This page documents functions that are accessed directly through the `muller` module.

## Table of Contents

- [read()](#read)
- [tiled()](#tiled)
- [Sample](#sample)

---

## read()

### Overview

Utility that reads raw data from supported files into MULLER format. It can recompress data into the format required by the tensor if permitted by the tensor htype, or copy data directly if the file format matches the sample_compression of the tensor to maximize upload speeds.

### Parameters

- **path** (`str` or `pathlib.Path`): Path to a supported file.
- **verify** (`bool`, optional): If `True`, contents of the file are verified. Defaults to `False`.
- **creds** (`dict`, optional): Credentials for s3, gcp and http urls. Defaults to `None`.
- **compression** (`str`, optional): Format of the file. Only required if path does not have an extension. Defaults to `None`.
- **storage** (`StorageProvider`, optional): Storage provider to use to retrieve remote files. Useful if multiple files are being read from same storage to minimize overhead of creating a new provider. Defaults to `None`.

### Returns

- **Sample**: Sample object. Call `sample.array` to get the `np.ndarray`.

### Examples

```python
import muller

# Read an image file
sample = muller.read("path/to/image.jpg")
array = sample.array

# Read with verification
sample = muller.read("path/to/data.png", verify=True)

# Read from remote storage with credentials
creds = {"aws_access_key_id": "...", "aws_secret_access_key": "..."}
sample = muller.read("s3://bucket/image.jpg", creds=creds)

# Specify compression explicitly
sample = muller.read("path/to/file", compression="png")
```

---

## tiled()

### Overview

Allocates an empty sample of shape `sample_shape`, broken into tiles of shape `tile_shape` (except for edge tiles). This is useful for efficiently storing and accessing large samples by dividing them into smaller tiles.

### Parameters

- **sample_shape** (`Tuple[int, ...]`): Full shape of the sample.
- **tile_shape** (`Tuple[int, ...]`, optional): The sample will be stored as tiles where each tile will have this shape (except edge tiles). If not specified, it will be computed such that each tile is close to half of the tensor's `max_chunk_size` (after compression). Defaults to `None`.
- **dtype** (`str` or `np.dtype`, optional): Dtype for the sample array. Defaults to `np.uint8`.

### Returns

- **PartialSample**: A PartialSample instance which can be appended to a Tensor.

### Examples

```python
import muller
import numpy as np

# Create a dataset with an image tensor
ds = muller.dataset("./my_dataset", overwrite=True)
ds.create_tensor("image", htype="image", sample_compression="png")

# Append a tiled sample
with ds:
    ds.image.append(muller.tiled(sample_shape=(1003, 1103, 3), tile_shape=(10, 10, 3)))
    # Fill part of the tiled sample with data
    ds.image[0][-217:, :212, 1:] = np.random.randint(0, 256, (217, 212, 2), dtype=np.uint8)

# Create a tiled sample with default tile shape
tiled_sample = muller.tiled(sample_shape=(5000, 5000, 3))

# Create a tiled sample with specific dtype
tiled_sample = muller.tiled(
    sample_shape=(2048, 2048),
    tile_shape=(256, 256),
    dtype=np.float32
)
```

---

## Sample

### Overview

The `Sample` class represents a single data sample in MULLER format. It is typically created by the `read()` function and provides access to the underlying data as a numpy array.

### Attributes

- **array** (`np.ndarray`): The numpy array representation of the sample data.
- **path** (`str`): The path to the source file (if applicable).
- **compression** (`str`): The compression format of the sample.

### Examples

```python
import muller

# Create a Sample from a file
sample = muller.read("path/to/image.jpg")

# Access the numpy array
array = sample.array
print(array.shape)
print(array.dtype)

# Use Sample in dataset operations
ds = muller.dataset("./my_dataset", overwrite=True)
ds.create_tensor("images")

with ds:
    ds.images.append(sample)
```

### Notes

- `Sample` objects are typically created automatically by `muller.read()` rather than instantiated directly.
- The `array` property provides lazy loading - the data is only loaded when accessed.
- Samples can be directly appended to tensors in a dataset.
