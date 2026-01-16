<center><img src="figures/logo.png" width="500"></center>

## MULLER: A Multimodal Data Lake Format for Collaborative AI Data Workflows

MULLER is a novel Multimodal data lake format designed for collaborative AI data workflows, with the following key features:
* **Mutimodal data support** with than 12 data types of different modalities, including scalars, vectors, text, images, videos, and audio, with 20+ compression formats (e.g., LZ4, JPG, PNG, MP3, MP4, AVI, WAV).
* **Data sampling, exploration, and Analysis** through low-latency random access and fast scan.
* **Array-oriented hybrid search engine** that jointly queries vector, text, and scalar data.
* **Git-like data versioning** with support for commit, checkout, diff, conflict detection and resolution, as well as merge. Specifically, to the best of our knowledge, MULLER is the first data lake format to support _fine-grained row-level updates and three-way merges_ across multiple coexisting data branches.
* **Seamless integration with LLM/MLLM data training and processing pipelines**.


## Getting Started
MULLER requires Python 3.11 or higher.

* Set up a conda environment
```bash
conda create -n muller python=3.11
conda activate muller
```
* Installation from code
```bash
pip install . 
```
[Optional] Development Installation
```bash
pip install -e .
```
[Optional] Skip C++ Module Building
```bash
BUILD_CPP=false pip install .
```
* Verify Installation
```python
import muller
print(muller.__version__)
```
