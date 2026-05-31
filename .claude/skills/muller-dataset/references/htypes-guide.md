# Data Types (htypes) Guide

## Overview

MULLER supports 12+ high-level data types (htypes) for different kinds of data.

## Supported htypes

### image
**Use:** Image files (photos, diagrams, etc.)
**dtype:** uint8
**Compression:** jpg, png, webp, bmp, gif, tiff, etc.

```python
ds.create_tensor("photos", htype="image", sample_compression="jpg")
ds.photos.append(muller.read("photo.jpg"))
```

### video
**Use:** Video files
**dtype:** uint8
**Compression:** mp4, mkv, avi

```python
ds.create_tensor("videos", htype="video", sample_compression="mp4")
ds.videos.append(muller.read("video.mp4"))
```

### audio
**Use:** Audio files
**dtype:** float64
**Compression:** mp3, wav, flac

```python
ds.create_tensor("audio", htype="audio", sample_compression="mp3")
ds.audio.append(muller.read("audio.mp3"))
```

### text
**Use:** Text/string data
**dtype:** str
**Compression:** None, lz4 (optional)

```python
ds.create_tensor("descriptions", htype="text")
ds.descriptions.append("A beautiful sunset")
```

### vector / embedding
**Use:** Embedding vectors, feature vectors
**dtype:** float32
**Compression:** None, lz4 (optional)

```python
ds.create_tensor("embeddings", htype="vector", dtype="float32")
ds.embeddings.append(np.array([0.1, 0.2, 0.3]))
```

### class_label
**Use:** Classification labels
**dtype:** uint32
**Compression:** None, lz4 (optional)

```python
ds.create_tensor("labels", htype="class_label", dtype="uint32")
ds.labels.append(5)
```

### bbox
**Use:** 2D bounding boxes
**dtype:** float32
**Compression:** None, lz4 (optional)

```python
ds.create_tensor("boxes", htype="bbox", dtype="float32")
ds.boxes.append([x, y, width, height])
```

### json
**Use:** JSON objects
**Compression:** None, lz4 (optional)

```python
ds.create_tensor("metadata", htype="json")
ds.metadata.append({"key": "value", "count": 42})
```

### list
**Use:** List data
**Compression:** None, lz4 (optional)

```python
ds.create_tensor("tags", htype="list")
ds.tags.append(["tag1", "tag2", "tag3"])
```

### generic
**Use:** Generic numeric data
**dtype:** Specify explicitly (int32, float32, etc.)
**Compression:** None, lz4 (optional)

```python
ds.create_tensor("scores", htype="generic", dtype="float32")
ds.scores.append(0.95)
```

## Compression Guidelines

### Always use compression:
- **image:** jpg, png, webp (reduces size 10-100x)
- **video:** mp4, mkv (reduces size 10-100x)
- **audio:** mp3, flac (reduces size 5-20x)

### Avoid compression (unless storage critical):
- **text:** Adds overhead, minimal benefit
- **class_label:** Small data, overhead not worth it
- **bbox:** Small data, overhead not worth it
- **generic:** Depends on data size

### Optional compression (lz4):
- Use when storage is critical
- Adds compression/decompression overhead
- Test performance impact

## Quick Reference Table

| htype | dtype | Best Compression | Use Case |
|-------|-------|------------------|----------|
| image | uint8 | jpg, png | Photos, images |
| video | uint8 | mp4 | Videos |
| audio | float64 | mp3, wav | Audio files |
| text | str | None | Text, strings |
| vector | float32 | None | Embeddings |
| class_label | uint32 | None | Labels |
| bbox | float32 | None | Bounding boxes |
| json | - | None | JSON objects |
| list | - | None | Lists |
| generic | specify | None | Numeric data |

## Examples

### Image Classification Dataset
```python
ds.create_tensor("images", htype="image", sample_compression="jpg")
ds.create_tensor("labels", htype="class_label", dtype="uint32")
```

### Text Dataset with Embeddings
```python
ds.create_tensor("text", htype="text")
ds.create_tensor("embeddings", htype="vector", dtype="float32")
```

### Object Detection Dataset
```python
ds.create_tensor("images", htype="image", sample_compression="jpg")
ds.create_tensor("boxes", htype="bbox", dtype="float32")
ds.create_tensor("labels", htype="class_label", dtype="uint32")
```

For complete details, see [../../docs/api/htypes.md](../../docs/api/htypes.md)
