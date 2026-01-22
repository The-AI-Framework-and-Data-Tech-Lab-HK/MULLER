<center><img src="figures/logo.png" width="500"></center>

## MULLER: A Multimodal Data Lake Format for Collaborative AI Data Workflows

MULLER is a novel Multimodal data lake format designed for collaborative AI data workflows, with the following key features:
* **Mutimodal data support** with than 12 data types of different modalities, including scalars, vectors, text, images, videos, and audio, with 20+ compression formats (e.g., LZ4, JPG, PNG, MP3, MP4, AVI, WAV).
* **Data sampling, exploration, and Analysis** through low-latency random access and fast scan.
* **Array-oriented hybrid search engine** that jointly queries vector, text, and scalar data.
* **Git-like data versioning** with support for commit, checkout, diff, conflict detection and resolution, as well as merge. Specifically, to the best of our knowledge, MULLER is the first data lake format to support _fine-grained row-level updates and three-way merges_ across multiple coexisting data branches.
* **Seamless integration with LLM/MLLM data training and processing pipelines**.


## Getting Started

#### Prerequisites

- Python >= 3.11
- CMake >= 3.22.1 (required for building C++ extensions)
- A C++17 compatible compiler (tested with gcc 11.4.0)
- Linux or macOS (tested on Ubuntu 22.04)

#### 1. (Recommended) Create a new Conda environment
```bash
conda create -n muller python=3.11
conda activate muller
```
#### 2. Installation
* First, clone the MULLER repository.
```bash
git clone https://github.com/The-AI-Framework-and-Data-Tech-Lab-HK/MULLER.git
cd MULLER
chmod 777 muller/util/sparsehash/build_proj.sh  # You may need to modify the script permissions.
```
* [Dafault] Install from code
```
pip install .   # Use `pip install . -v` to view detailed build logs
```
* [Optional] Development (editable) installation
```bash
pip install -e .
```
* [Optional] Skip building C++ modules

The Python implementation provides the same core functionality as the C++ modules.
If you only need the basic features of MULLER, you may skip building the C++ extensions:
```bash
BUILD_CPP=false pip install .
```
#### 3. Verify the Installation
```python
import muller
print(muller.__version__)
```

## Examples
#### 1. Creating a MULLER Dataset
* Note: MULLER support 12+ data types of different modalities, including scalars, vectors, text, images, videos, and audio, with 20+ compression formats (e.g., LZ4, JPG, PNG, MP3, MP4, AVI, WAV).
```python
import muller

# Create an empty MULLER datatset
ds = muller.dataset(path='test_dataset/', overwrite=True)

# Create Columns
ds.create_tensor(name='my_images', htype='image', sample_compression='jpg')
ds.create_tensor('labels', htype='generic', dtype='int')
ds.create_tensor('description', htype='text')

# Append data
with ds:
    ds.my_images.extend([muller.read(img_path_0), muller.read(img_path_1), muller.read(img_path_2), muller.read(img_path_3), muller.read(img_path_4)])
    ds.labels.extend([0, 1, 2, 3, 4])
    ds.description.extend(["A majestic long-haired Maine Coon cat perched on a wooden bookshelf, staring intently at a tree outside with its bright amber eyes.", 
                           "A domestic short-hair cat with a distinctive tuxedo pattern stretching lazily across a velvet sofa in a dimly lit living room.", 
                           "An energetic Golden Retriever with bright amber eyes sprinting across a vibrant green meadow, its fur glistening under the afternoon sun as it chases a bright yellow tennis ball.", 
                           "A focused German Shepherd sitting patiently on a cobblestone street, wearing a professional service harness and looking up at its handler for the next command.", 
                           "A soft, white lop-eared rabbit with bright eyes nestled in a patch of clover, twitching its pink nose while nibbling on a fresh garden carrot."])

# Check the metadata and schema of the dataset
ds.summary() 
```

#### 2. 

## Reproduction steps for the experiment results in our paper

Please refer to [exp_scripts/README.md](https://github.com/spencerr221/MULLER/blob/main/exp_scripts/README.md) for the detailed steps.
