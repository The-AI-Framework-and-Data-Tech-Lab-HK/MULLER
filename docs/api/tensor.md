# Tensor API

This page documents public APIs on `Tensor` objects. A tensor is a dataset column, accessed with `ds.<tensor_name>`, `ds["<tensor_name>"]`, or `ds.tensors["<tensor_name>"]`.

## Table of Contents

### Writing Data
- [Tensor.append()](#tensorappend)
- [Tensor.extend()](#tensorextend)
- [Tensor.clear()](#tensorclear)
- [Index assignment](#index-assignment)

### Reading Data
- [Tensor.numpy()](#tensornumpy)
- [Tensor.data()](#tensordata)
- [Tensor.text()](#tensortext)
- [Tensor.dict()](#tensordict)
- [Tensor.list()](#tensorlist)
- [Tensor.tobytes()](#tensortobytes)
- [Index access](#index-access)

### Properties
- [Tensor.htype](#tensorhtype)
- [Tensor.dtype](#tensordtype)
- [Tensor.shape](#tensorshape)
- [Tensor.shape_interval](#tensorshape_interval)
- [Tensor.ndim](#tensorndim)
- [Tensor.num_samples](#tensornum_samples)
- [Tensor.sample_info](#tensorsample_info)

---

## Writing Data

### Tensor.append()

#### Signature

```python
Tensor.append(sample, ignore_errors=False)
```

#### Parameters

- **sample** (`InputSample`): One sample to append to this tensor. Accepted sample values depend on the tensor htype and compression configuration.
- **ignore_errors** (`bool`, optional): If `True`, skips samples that fail during append processing where supported by the underlying chunk engine. Defaults to `False`.

#### Returns

- **None**

#### Example

```python
with ds:
    ds.labels.append(1)
    ds.text.append("a short caption")
```

---

### Tensor.extend()

#### Signature

```python
Tensor.extend(samples, progressbar=False, ignore_errors=False)
```

#### Parameters

- **samples** (`np.ndarray`, `Sequence[InputSample]`, or `Tensor`): Samples to append to this tensor. Passing another `Tensor` copies data from that tensor view.
- **progressbar** (`bool`, optional): If `True`, shows progress while extending. Defaults to `False`.
- **ignore_errors** (`bool`, optional): If `True`, continues where supported when an individual sample fails. Defaults to `False`.

#### Returns

- **None**

#### Example

```python
with ds:
    ds.labels.extend(np.array([0, 1, 2], dtype=np.int32))
    ds.text.extend(["cat", "dog", "tree"])
```

---

### Tensor.clear()

#### Signature

```python
Tensor.clear()
```

#### Parameters

None.

#### Returns

- **None**

#### Example

```python
with ds:
    ds.temp_values.clear()
```

---

### Index Assignment

#### Signature

```python
tensor[item] = value
```

#### Parameters

- **item** (`int` or `slice`): Sample position or contiguous sample range to update.
- **value** (`Any`): Replacement sample or samples. When assigning a `Tensor`, MULLER reads it with `value.numpy(aslist=True)` before updating.

#### Returns

- **None**

#### Notes

- Assignment updates existing samples and marks the dataset as no longer append-only.
- If random assignment is enabled in MULLER constants, assigning to an integer index beyond the current length pads missing samples before appending the new value.

#### Example

```python
with ds:
    ds.labels[0] = 5
    ds.labels[1:3] = np.array([6, 7], dtype=np.int32)
```

---

## Reading Data

### Tensor.numpy()

#### Signature

```python
Tensor.numpy(aslist=False, fetch_chunks=False, max_workers=MAX_WORKERS_FOR_CHUNK_ENGINE)
```

#### Parameters

- **aslist** (`bool`, optional): If `True`, returns a list of `np.ndarray` objects. This is useful for dynamically shaped tensors. If `False`, returns a single `np.ndarray` unless the selected samples cannot be represented as one regular array. Defaults to `False`.
- **fetch_chunks** (`bool`, optional): If `True`, fetches complete chunks from storage. If `False`, fetches only required bytes where possible. MULLER may still fetch chunks internally for iteration, chunk-compressed tensors, or large chunks. Defaults to `False`.
- **max_workers** (`int`, optional): Maximum worker count used by the chunk engine while reading. Defaults to `MAX_WORKERS_FOR_CHUNK_ENGINE`.

#### Returns

- **np.ndarray** or **List[np.ndarray]**

#### Notes

- `Tensor.numpy()` does not accept an `asrow` parameter.
- For tensors with htype `polygon`, `aslist` is always treated as `True`.

#### Example

```python
arr = ds.labels[:10].numpy()
samples = ds.images[:10].numpy(aslist=True)
```

---

### Tensor.data()

#### Signature

```python
Tensor.data(aslist=False, fetch_chunks=False)
```

#### Parameters

- **aslist** (`bool`, optional): Passed to `Tensor.numpy()` for htypes that return numpy-backed values. Defaults to `False`.
- **fetch_chunks** (`bool`, optional): Passed to the underlying read path. Defaults to `False`.

#### Returns

- **dict**: A htype-specific dictionary.

#### Return Format

- `text`, `json`, `list`, and `tag` tensors return `{"value": ...}` using `Tensor.text()`, `Tensor.dict()`, or `Tensor.list()`.
- `video` tensors return `{"frames": ..., "timestamps": ..., "sample_info": ...}`.
- `class_label` tensors return `{"value": ...}` and, when class names are configured, `"text"`.
- `image`, `image.rgb`, `image.gray`, `dicom`, and `nifti` tensors return `{"value": ..., "sample_info": ...}`.
- Other htypes return `{"value": ...}` using numpy-backed data.

#### Example

```python
value = ds.labels[0].data()["value"]
image = ds.images[0].data()
```

---

### Tensor.text()

#### Signature

```python
Tensor.text(fetch_chunks=False)
```

#### Parameters

- **fetch_chunks** (`bool`, optional): Passed to the underlying numpy read. Defaults to `False`.

#### Returns

- **Any**: Text value for one selected sample, or a list of text values for multiple selected samples.

#### Raises

- **Exception**: If the tensor base htype is not `text`.

#### Example

```python
caption = ds.captions[0].text()
captions = ds.captions[:3].text()
```

---

### Tensor.dict()

#### Signature

```python
Tensor.dict(fetch_chunks=False)
```

#### Parameters

- **fetch_chunks** (`bool`, optional): Passed to the underlying numpy read. Defaults to `False`.

#### Returns

- **Any**: JSON value for one selected sample, or a list of JSON values for multiple selected samples.

#### Raises

- **Exception**: If the tensor base htype is not `json`.

#### Example

```python
metadata = ds.metadata[0].dict()
```

---

### Tensor.list()

#### Signature

```python
Tensor.list(fetch_chunks=False)
```

#### Parameters

- **fetch_chunks** (`bool`, optional): Passed to the underlying numpy read. Defaults to `False`.

#### Returns

- **list**: List data for tensors with base htype `list` or `tag`.

#### Raises

- **Exception**: If the tensor base htype is not `list` or `tag`.

#### Example

```python
tags = ds.tags[0].list()
```

---

### Tensor.tobytes()

#### Signature

```python
Tensor.tobytes()
```

#### Parameters

None.

#### Returns

- **bytes**: Raw bytes for exactly one selected sample. For uncompressed tensors this is the numpy-array bytes; for sample-compressed tensors this is the compressed sample bytes.

#### Raises

- **ValueError**: If the tensor view selects zero, multiple, or sliced samples.

#### Example

```python
raw = ds.images[0].tobytes()
```

---

### Index Access

#### Signature

```python
tensor[item]
```

#### Parameters

- **item** (`int`, `np.integer`, `slice`, `list`, `tuple`, `Ellipsis`, or `Index`): Index expression used to create a tensor view.

#### Returns

- **Tensor**: A lazy tensor view over the selected samples or dimensions. Data is loaded only when a read method such as `numpy()`, `data()`, or `tobytes()` is called.

#### Raises

- **InvalidKeyTypeError**: If the index type is unsupported.

#### Example

```python
one_sample = ds.images[0]
batch = ds.images[10:20]
array = batch.numpy(aslist=True)
```

---

## Properties

### Tensor.htype

#### Type

- **str**

#### Description

High-level tensor type, such as `generic`, `image`, `text`, `json`, `list`, `class_label`, or `vector`. This property can be assigned for supported htype conversions; unsupported conversions raise an error.

---

### Tensor.dtype

#### Type

- **np.dtype** or **None**

#### Description

Numpy dtype for the tensor. For base htypes `json`, `list`, and `tag`, this returns `np.dtype(str)`. If dtype metadata is unavailable, it returns `None`.

---

### Tensor.shape

#### Type

- **Tuple[Optional[int], ...]**

#### Description

Shape of the current tensor view. Dynamic dimensions are represented with `None`.

---

### Tensor.shape_interval

#### Type

- **ShapeInterval**

#### Description

Minimum and maximum shape information for the current tensor view. Use this to inspect whether samples have dynamic shapes.

---

### Tensor.ndim

#### Type

- **int**

#### Description

Number of dimensions in the current tensor view.

---

### Tensor.num_samples

#### Type

- **int**

#### Description

Total number of samples in the tensor. This ignores the current tensor view index and returns the length of the primary axis for the full tensor.

---

### Tensor.sample_info

#### Type

- **dict**, **List[dict]**, or **None**

#### Description

Sample metadata for the current tensor view. A single-sample view returns a dict, a multi-sample view returns a list of dicts, and tensors without a sample-info tensor return `None`.

Sample info is available only when the dataset was created with the corresponding hidden sample-info tensor, for example by passing `create_sample_info_tensor=True` to `create_tensor()` for supported media htypes.
