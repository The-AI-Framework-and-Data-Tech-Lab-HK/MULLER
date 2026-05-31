# Quick Start Guide

## 5-Minute MULLER Tutorial

### 1. Create a Dataset

```python
import muller

# Create empty dataset
ds = muller.dataset("./my_dataset", overwrite=True)

# Create tensors
ds.create_tensor("images", htype="image", sample_compression="jpg")
ds.create_tensor("labels", htype="class_label", dtype="uint32")
```

**Via skill:**
```bash
uv run scripts/dataset_manager.py create --path ./my_dataset \
  --tensors "images:image:jpg,labels:class_label:uint32"
```

### 2. Add Data

```python
import numpy as np

with ds:
    # Append single sample
    ds.append({
        "images": muller.read("photo.jpg"),
        "labels": 1
    })

    # Extend with multiple samples
    ds.extend({
        "images": [muller.read(f"img{i}.jpg") for i in range(10)],
        "labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    })
```

**Via skill:**
```bash
uv run scripts/data_operations.py extend --path ./my_dataset \
  --data-file samples.json
```

### 3. Query Data

```python
# Filter samples
result = ds.filter_vectorized([("labels", ">", 5)])

# Access samples
for sample in result:
    print(sample["labels"].numpy())
```

**Via skill:**
```bash
uv run scripts/data_operations.py query --path ./my_dataset \
  --filter "labels > 5"
```

### 4. Inspect Dataset

```python
# Summary
ds.summary()

# Statistics
stats = ds.statistics()
print(f"Samples: {ds.num_samples}")
```

**Via skill:**
```bash
uv run scripts/dataset_manager.py info --path ./my_dataset
uv run scripts/dataset_manager.py stats --path ./my_dataset
```

## Common Patterns

### Import from Files

```python
# From JSONL
ds = muller.from_file("data.jsonl", "./my_dataset")

# From dataframes
data = [{"text": "hello", "label": 1}, {"text": "world", "label": 2}]
ds = muller.from_dataframes(data, "./my_dataset")
```

### Update and Delete

```python
with ds:
    # Update sample
    ds[0].update({"labels": 10})

    # Delete samples
    ds.pop([0, 5, 10])
```

### Tensor Management

```python
with ds:
    # Create tensor
    ds.create_tensor("embeddings", htype="vector", dtype="float32")

    # Rename tensor
    ds.rename_tensor("old_name", "new_name")

    # Delete tensor
    ds.delete_tensor("temp_tensor")
```

## Storage Backends

### Local Storage
```python
ds = muller.dataset("./my_dataset")
ds = muller.dataset("/absolute/path/dataset")
```

### S3 Storage
```python
creds = {
    "bucket_name": "my-bucket",
    "endpoint": "http://s3.example.com",
    "ak": "access_key",
    "sk": "secret_key"
}
ds = muller.dataset("s3://my_dataset", creds=creds)
```

## Best Practices

1. **Use `with ds:` for writes** - Improves performance
2. **Choose appropriate compression** - jpg for images, mp4 for video
3. **Avoid compression for text/labels** - Unless storage is critical
4. **Use `extend()` over `append()`** - More efficient for batches
5. **Check `ds.summary()`** - Verify schema before adding data

## Next Steps

- See [api-cheatsheet.md](api-cheatsheet.md) for complete API reference
- See [htypes-guide.md](htypes-guide.md) for data type details
- See [../../docs/](../../docs/) for full documentation
