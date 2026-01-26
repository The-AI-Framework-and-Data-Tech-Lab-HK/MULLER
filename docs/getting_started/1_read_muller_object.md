## 1. Loading Multimodal Files into MULLER Objects

MULLER provides the `muller.read()` interface for loading files from different storage backends via the `path` argument.  
Below, we use image loading as an example.

**Example 1: Loading from a Local File Path**
```python
>>> img_1 = muller.read(path="/your/data/path/xxx.jpg")
>>> img_1.array  # Access the tensor representation of the multimodal object via `.array`.
array([[[183, 160, 118],
        [223, 201, 160],
        [211, 189, 150],
        ...,
        [143, 146,  79],
        [186, 189, 122],
        [142, 145,  78]]], shape=(400, 500, 3), dtype=uint8)

>>> img_1.shape  # Retrieve the tensor shape via the `.shape` attribute.
(400, 500, 3)  # This JPEG image has a resolution of 400×500 (height × width) with three RGB channels.

>>> img_1.dtype  # Retrieve the underlying data type via the `.dtype` attribute.
'uint8'  # JPEG images represent pixel values in the range [0, 255], and are therefore stored as uint8.
```

**Example 2: Loading from an HTTP/HTTPS URL**  
(If access is restricted, the `creds` argument can be used to provide proxy settings.)
```python
>>> credential_2 = {
...     "proxies": {
...         "http": "http://account:pwd@proxy.xxx.com:port",
...         "https": "http://account:pwd@proxy.xxx.com:port"
...     }
... }
>>> img_2 = muller.read(
...     path="https://xxx.com/xxx.jpg",
...     creds=credential_2
... )
```

**Example 3: Loading from a ROMA/S3 object storage bucket**
```python
>>> credential_3 = {
...     "bucket_name": "bucket-name",
...     "region": "xxx",
...     "app_token": "xxx",
...     "vendor": "xxx"
... }
>>> img_3 = muller.read(
...     path="roma://your/data/path/xxx.jpg",
...     creds=credential_3
... )
```

- `muller.read()` currently supports multiple multimodal file types, including images, audio, and video, covering more than 20 file formats.
- The optional `creds` argument can be used to access remote or cloud storage backends (e.g., HTTP/HTTPS, ROMA, OBS), and may include authentication or connection information such as `proxies`, `bucket_name`, `access_key`, `secret_key`, and `token`.
- For the complete list of input parameters, supported file types, and usage details of `muller.read()`, please refer to the API documentation.
