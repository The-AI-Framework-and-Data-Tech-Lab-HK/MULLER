# Htypes Reference

MULLER supports various high-level data types (htypes) for different kinds of data. Each htype has specific configurations, supported compressions, and data formats.

---

## Overview

| Htype | Default dtype | Supported Compressions | Use Case |
|-------|---------------|------------------------|----------|
| `generic` | None (inferred) | None, lz4 | General numeric data |
| `text` | str | None, lz4 | String/text data |
| `image` | uint8 | bmp, dib, gif, ico, jpg, jpeg, jpeg2000, pcx, png, ppm, sgi, tga, tiff, webp, wmf, xbm, eps, fli, im, msp, mpo | Image files |
| `video` | uint8 | mp4, mkv, avi | Video files |
| `audio` | float64 | mp3, wav, flac | Audio files |
| `class_label` | uint32 | None, lz4 | Classification labels |
| `bbox` | float32 | None, lz4 | Bounding boxes (2D) |
| `bbox.3d` | float32 | None, lz4 | Bounding boxes (3D) |
| `json` | Any | None, lz4 | JSON objects |
| `list` | List | None, lz4 | List data |
| `vector` | float32 | None, lz4 | Embedding vectors |
| `embedding` | float32 | None, lz4 | Embedding vectors (alias) |

---

## generic

### Suggested Use Case

`htype="generic"` is assigned to numeric tensors with unspecified htype. However, it is not recommended for general use because it does not provide an informative description of the data or efficiently store it. Use `generic` only when you are uncertain about the role of the data in the dataset but still want to record it.

### Creating a generic tensor

```python
ds.create_tensor("generic_data", htype="generic", dtype="float32", sample_compression="lz4")
```

**Parameters:**
- `dtype`: Default is `None`, and a value will be assigned once a sample is appended. Supported dtypes: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `float32`, `float64`, `bool`.
- `sample_compression`: Supported values: `{None, "lz4"}`.

### Appending generic data

**Example 1:** Appending numpy arrays

```python
ds.generic_data.append(np.array([[1, 2, 3], [4, 5, 6]]))
ds.generic_data.append(np.array([1.0, 2.0, 3.0]))
ds.generic_data.append(np.array([True, True, False]))
```

**Example 2:** Appending lists

```python
ds.generic_data.append([[1, 2, 3], [4, 5, 6]])
ds.generic_data.append([1.0, 2.0, 3.0])
ds.generic_data.append([True, True, False])
```

---

## text

### Suggested Use Case

Use `htype="text"` when storing string information in the dataset.

### Creating a text tensor

```python
ds.create_tensor("text_data", htype="text", sample_compression="lz4")
```

**Parameters:**
- `dtype`: Default is `str`.
- `sample_compression`: Supported values: `{None, "lz4"}`.

### Appending text data

```python
ds.text_data.append("Thanks for using MULLER!")
```

---

## image

### Suggested Use Case

Use `htype="image"` for storing image data. For example, in the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), images should be stored with this htype.

### Creating an image tensor

```python
ds.create_tensor("image_data", htype="image", sample_compression="jpeg")
```

**Parameters:**
- `dtype`: Default is `uint8`.
- `sample_compression`: Supported values: `{None, "bmp", "dib", "eps", "fli", "gif", "ico", "im", "jpeg", "jpg", "jpeg2000", "msp", "mpo", "pcx", "png", "ppm", "sgi", "tga", "tiff", "webp", "wmf", "xbm"}`.

### Appending image data

**Example 1:** Appending `Sample` read by `muller.read()` (recommended - keep `sample_compression` consistent with file format)

```python
ds.create_tensor("image_data", htype="image", sample_compression="jpeg")
ds.image_data.append(muller.read("/path/to/file.jpg"))

ds.create_tensor("image_data_2", htype="image", sample_compression="png")
ds.image_data_2.append(muller.read("/path/to/file.png"))
```

**Example 2:** Appending raw numpy arrays (compression not supported for raw data)

```python
ds.create_tensor("image_data", htype="image", sample_compression=None)
ds.image_data.append(np.zeros((400, 300, 3), dtype=np.uint8))
```

**Example 3:** Appending raw lists (compression not supported for raw data)

```python
ds.create_tensor("image_data", htype="image", sample_compression=None)
ds.image_data.append(np.zeros((400, 300, 3), dtype=np.uint8).tolist())
```

---

## video

### Suggested Use Case

Use `htype="video"` for storing video files. For example, datasets like [YouTube-8M](https://research.google.com/youtube8m/index.html) should use this htype.

### Creating a video tensor

```python
ds.create_tensor("video_data", htype="video", sample_compression="mp4")
```

**Parameters:**
- `dtype`: Default is `uint8`.
- `sample_compression`: Supported values: `{None, "mp4", "mkv", "avi"}`.

### Appending video data

**Example 1:** Appending `Sample` read by `muller.read()` (recommended)

```python
ds.create_tensor("video_data", htype="video", sample_compression="mp4")
ds.video_data.append(muller.read("/path/to/file.mp4"))

ds.create_tensor("video_data_2", htype="video", sample_compression="mkv")
ds.video_data_2.append(muller.read("/path/to/file.mkv"))
```

**Example 2:** Appending raw numpy arrays (compression not supported)

```python
ds.create_tensor("video_data", htype="video", sample_compression=None)
ds.video_data.append(np.zeros((100, 200, 300, 3), dtype=np.uint8))
```

---

## audio

### Suggested Use Case

Use `htype="audio"` for storing audio files. For example, [speech datasets](https://huggingface.co/learn/audio-course/en/chapter5/choosing_dataset#a-summary-of-datasets-on-the-hub) should use this htype.

### Creating an audio tensor

```python
ds.create_tensor("audio_data", htype="audio", sample_compression="mp3")
```

**Parameters:**
- `dtype`: Default is `float64`.
- `sample_compression`: Supported values: `{None, "mp3", "wav", "flac"}`.

### Appending audio data

**Example 1:** Appending `Sample` read by `muller.read()` (recommended)

```python
ds.create_tensor("audio_data", htype="audio", sample_compression="mp3")
ds.audio_data.append(muller.read("/path/to/file.mp3"))

ds.create_tensor("audio_data_2", htype="audio", sample_compression="wav")
ds.audio_data_2.append(muller.read("/path/to/file.wav"))
```

**Example 2:** Appending raw numpy arrays (compression not supported)

```python
ds.create_tensor("audio_data", htype="audio", sample_compression=None)
ds.audio_data.append(np.zeros((5, 3), dtype=np.float64))
```

---

## class_label

### Suggested Use Case

Use `htype="class_label"` for storing classification labels. For example, in the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), there are 10 categories: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"], typically represented as integers 0-9.

### Creating a class_label tensor

```python
ds.create_tensor("labels", htype="class_label", sample_compression="lz4")
```

**Parameters:**
- `dtype`: Default is `uint32`.
- `sample_compression`: Supported values: `{None, "lz4"}`.
- `class_names` (optional): A list of strings mapping integers to category names.

**With class names:**

```python
ds.create_tensor("labels", htype="class_label", sample_compression="lz4",
                 class_names=["airplane", "automobile", "bird", "cat", "deer",
                             "dog", "frog", "horse", "ship", "truck"])
```

Or update after creation:

```python
ds.labels.info.update(class_names=["airplane", "automobile", "bird", "cat", "deer",
                                    "dog", "frog", "horse", "ship", "truck"])
```

### Appending class_label data

**Example 1:** Appending numpy arrays

```python
ds.labels.append(np.array([1, 2, 3], dtype="uint32"))
```

**Example 2:** Appending lists

```python
ds.labels.append([1, 2, 3])
```

**Example 3:** Appending strings (automatically mapped to integers)

```python
ds.labels.append(["airplane", "cat", "dog"])
```

---

## bbox

### Suggested Use Case

Use `htype="bbox"` for storing 2D bounding box annotations. For example, [COCO dataset](https://cocodataset.org/#home) bounding boxes should use this htype.

### Creating a bbox tensor

```python
ds.create_tensor("bbox_data", htype="bbox", sample_compression="lz4")
```

**Parameters:**
- `dtype`: Default is `float32`.
- `sample_compression`: Supported values: `{None, "lz4"}`.
- `coords` (optional): A dictionary specifying coordinate format.

**With coordinate specification:**

```python
ds.create_tensor("bbox_data", htype="bbox", sample_compression="lz4",
                 coords={"type": "fractional", "mode": "CCWH"})
```

**coords dictionary:**
- `type`: `"pixel"` (coordinates in pixels) or `"fractional"` (coordinates relative to image dimensions, like YOLO format)
- `mode`:
  - `"LTRB"`: left_x, top_y, right_x, bottom_y
  - `"LTWH"`: left_x, top_y, width, height
  - `"CCWH"`: center_x, center_y, width, height

Or update after creation:

```python
ds.bbox_data.info.update(coords={"type": "fractional", "mode": "CCWH"})
```

### Appending bbox data

**Example 1:** Appending numpy arrays

```python
ds.bbox_data.append(np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
```

**Example 2:** Appending lists

```python
ds.bbox_data.append([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
```

**Example 3:** Appending list of arrays

```python
ds.bbox_data.append([np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)])
```

---

## bbox.3d

### Suggested Use Case

Use `htype="bbox.3d"` for storing 3D bounding box annotations in autonomous driving or 3D object detection datasets.

### Creating a bbox.3d tensor

```python
ds.create_tensor("bbox_3d_data", htype="bbox.3d", sample_compression="lz4",
                 coords={"mode": "XYZWHD"})
```

**Parameters:**
- `dtype`: Default is `float32`.
- `sample_compression`: Supported values: `{None, "lz4"}`.
- `coords` (optional): A dictionary specifying the coordinate mode.

### Appending bbox.3d data

```python
ds.bbox_3d_data.append(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32))
```

---

## json

### Suggested Use Case

Use `htype="json"` for storing arbitrary JSON objects or dictionaries.

### Creating a json tensor

```python
ds.create_tensor("json_data", htype="json", sample_compression="lz4")
```

**Parameters:**
- `dtype`: Default is `Any`.
- `sample_compression`: Supported values: `{None, "lz4"}`.

### Appending json data

```python
ds.json_data.append({"key": "value", "number": 42, "nested": {"a": 1}})
```

---

## list

### Suggested Use Case

Use `htype="list"` for storing list data structures.

### Creating a list tensor

```python
ds.create_tensor("list_data", htype="list", sample_compression="lz4")
```

**Parameters:**
- `dtype`: Default is `List`.
- `sample_compression`: Supported values: `{None, "lz4"}`.

### Appending list data

```python
ds.list_data.append([1, 2, 3, 4, 5])
ds.list_data.append(["a", "b", "c"])
```

---

## vector

### Suggested Use Case

Use `htype="vector"` for storing embedding vectors for similarity search and vector indexing.

### Creating a vector tensor

```python
ds.create_tensor("embeddings", htype="vector", dtype="float32", dimension=128)
```

**Parameters:**
- `dtype`: Default is `float32`.
- `sample_compression`: Supported values: `{None, "lz4"}`.
- `dimension` (required): The dimensionality of the vectors.

### Appending vector data

```python
ds.embeddings.append(np.random.rand(128).astype(np.float32))
```

---

## embedding

### Suggested Use Case

`htype="embedding"` is an alias for `vector`. Use for storing embedding vectors.

### Creating an embedding tensor

```python
ds.create_tensor("embeddings", htype="embedding", dtype="float32")
```

**Parameters:**
- `dtype`: Default is `float32`.
- `sample_compression`: Supported values: `{None, "lz4"}`.

### Appending embedding data

```python
ds.embeddings.append(np.random.rand(128).astype(np.float32))
```

---

## See Also

- [Dataset Methods: create_tensor()](../dataset-methods/#create_tensor) - Detailed API for creating tensors
- [Getting Started: Creating a MULLER Dataset](../../getting_started/2_create_muller_dataset/) - Tutorial on dataset creation
