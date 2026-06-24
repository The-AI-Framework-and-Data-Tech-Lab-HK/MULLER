# Top-Level Functions

This page documents functions that are accessed directly through the `muller` module.

## Table of Contents

- [muller.read()](#mullerread)
- [muller.tiled()](#mullertiled)
- [muller.Sample](#mullersample)

---

### muller.read()

#### Overview

Utility that reads raw data from supported files into MULLER format. It can recompress data into the format required by the tensor if permitted by the tensor htype, or copy data directly if the file format matches the sample_compression of the tensor to maximize upload speeds.

#### Signature

```python
muller.read(
    path,
    verify=False,
    creds=None,
    compression=None,
    storage=None,
)
```

#### Parameters

- **path** (`str` or `pathlib.Path`): Path to a supported file.
- **verify** (`bool`, optional): If `True`, contents of the file are verified. Defaults to `False`.
- **creds** (`dict`, optional): Credentials or connection options for remote paths. HTTP reads use values such as `Authorization` and `proxies`; ROMA reads use values such as `bucket_name`, `region`, `app_token`, and `vendor`. Defaults to `None`.
- **compression** (`str`, optional): Format of the file. Only required if path does not have an extension. Defaults to `None`.
- **storage** (`StorageProvider`, optional): Storage provider to use to retrieve remote files. Useful if multiple files are being read from same storage to minimize overhead of creating a new provider. Defaults to `None`.

#### Returns

- **Sample**: Sample object. Call `sample.array` to get the `np.ndarray`.

#### Examples

```python
import muller

# Read an image file
sample = muller.read("path/to/image.jpg")
array = sample.array

# Read with verification
sample = muller.read("path/to/data.png", verify=True)

# Read from HTTP with proxy settings
creds = {
    "proxies": {
        "http": "http://proxy.example.com:8080",
        "https": "http://proxy.example.com:8080",
    }
}
sample = muller.read("https://example.com/image.jpg", creds=creds)

# Read from ROMA object storage
roma_creds = {
    "bucket_name": "bucket-name",
    "region": "region-name",
    "app_token": "...",
    "vendor": "...",
}
sample = muller.read("roma://path/to/image.jpg", creds=roma_creds)

# Specify compression explicitly
sample = muller.read("path/to/file", compression="png")
```

---

### muller.tiled()

#### Overview

Allocates an empty sample of shape `sample_shape`, broken into tiles of shape `tile_shape` (except for edge tiles). This is useful for efficiently storing and accessing large samples by dividing them into smaller tiles.

#### Parameters

- **sample_shape** (`Tuple[int, ...]`): Full shape of the sample. This is stored as the returned partial sample's `sample_shape` and exposed via its `shape` property.
- **tile_shape** (`Tuple[int, ...]`, optional): The sample will be stored as tiles where each tile will have this shape (except edge tiles). If not specified, it will be computed such that each tile is close to half of the tensor's `max_chunk_size` (after compression). Defaults to `None`.
- **dtype** (`str` or `np.dtype`, optional): Dtype for the sample array. Defaults to `np.uint8`.

#### Returns

- **PartialSample**: A PartialSample instance which can be appended to a Tensor.

#### Examples

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

### muller.Sample

#### Overview

The `Sample` class represents a single data sample in MULLER format. It can wrap a lazy file path, an in-memory NumPy array, or a byte buffer, and provides lazy access to the underlying data as a NumPy array.

#### Signature

```python
muller.Sample(
    path=None,
    array=None,
    buffer=None,
    compression=None,
    verify=False,
    shape=None,
    dtype=None,
    creds=None,
    storage=None,
)
```

#### Parameters

- **path** (`str`, optional): Path to a sample stored on the local file system or a supported remote backend. If `path` is provided, `array` should not be provided. Path-backed samples are lazy until data is accessed.
- **array** (`np.ndarray`, optional): In-memory array for a single sample. If `array` is provided, `path` should not be provided.
- **buffer** (`bytes` or `memoryview`, optional): Byte buffer for a single sample. If the buffer is compressed, provide `compression`.
- **compression** (`str`, optional): Compression or file format for path or buffer data. Useful when the path has no extension.
- **verify** (`bool`, optional): If `True`, verify compressed path or buffer contents and read metadata eagerly. Defaults to `False`.
- **shape** (`Tuple[int, ...]`, optional): Shape metadata for the sample. This can avoid metadata reads for uncompressed buffers.
- **dtype** (`str`, optional): Data type metadata for the sample. This can avoid metadata reads for uncompressed buffers.
- **creds** (`dict`, optional): Credentials or connection options for remote reads.
- **storage** (`StorageProvider`, optional): Storage provider used to retrieve remote files.

#### Attributes

- **array** (`np.ndarray`): The numpy array representation of the sample data.
- **buffer** (`bytes` or `memoryview`): Raw buffer for the sample. Path-backed samples read from storage on first access.
- **path** (`str` or `None`): The path to the source file, if the sample was created from a path.
- **compression** (`str` or `None`): Compression or file format of the sample.
- **shape** (`Tuple[int, ...]`): Shape of the sample.
- **dtype** (`str`): Data type of the sample.
- **is_lazy** (`bool`): `True` when the sample has not loaded array data yet.
- **is_empty** (`bool`): `True` when at least one dimension of `shape` is `0`.
- **is_text_like** (`bool`): `True` for text-like htypes such as `text`, `list`, `json`, or `tag`.
- **pil** (`PIL.Image.Image`): PIL image representation for image samples.
- **meta** (`dict`): Metadata such as `shape`, `format`, filename, and image EXIF or DICOM metadata when available.

#### Examples

```python
import muller
import numpy as np

# Create a Sample from a file
sample = muller.read("path/to/image.jpg")

# Access the numpy array
array = sample.array
print(array.shape)
print(array.dtype)

# Create a Sample from an array
sample = muller.Sample(array=np.zeros((32, 32, 3), dtype=np.uint8))
print(sample.shape)
print(sample.dtype)

# Create a Sample from an uncompressed buffer with explicit metadata
buffer = np.zeros((10, 10), dtype=np.float32).tobytes()
sample = muller.Sample(buffer=buffer, shape=(10, 10), dtype="float32")

# Use Sample in dataset operations
ds = muller.dataset("./my_dataset", overwrite=True)
ds.create_tensor("images")

with ds:
    ds.images.append(sample)
```

#### Notes

- `Sample` objects are typically created automatically by `muller.read()` rather than instantiated directly.
- The `array` property provides lazy loading - the data is only loaded when accessed.
- Samples can be directly appended to tensors in a dataset.
- For array-backed samples, `shape` and `dtype` are inferred from the provided NumPy array.
