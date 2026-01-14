<img src="figures/logo.png" width="300">

### MULLER: A Multimodal Data Lake Format for Collaborative AI Data Workflows

MULLER is a novel Multimodal data lake format designed for collaborative AI data workflows. 
Specifically, MULLER supports (1) a vectorized hybrid search engine that jointly queries vector, text, and scalar data, (2) Git-like data versioning with support for commit, checkout, diff, conflict detection and resolution, as well as merge, (3) efficient data sampling and exploration through low-latency random access and high-throughput traversal, and (4) seamless integration with deep learning frameworks.

### Getting Started
MULLER requires Python 3.11 or higher.

* Set up a conda environment
```bash
conda create -n gtn_f python=3.11
conda activate gtn_f
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
import gtn_f
print(gtn_f.__version__)
```
