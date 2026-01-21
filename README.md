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

#### 1. (Recommended) Set up a new conda environment
```bash
conda create -n muller python=3.11
conda activate muller
```
#### 2. Clone the MULLER project, and install MULLER from code
```bash
git clone https://github.com/The-AI-Framework-and-Data-Tech-Lab-HK/MULLER.git
cd MULLER
chmod 777 muller/util/sparsehash/build_proj.sh  # You may need to change mode of the shell script.
pip install .   # You may also use pip install . -v to check the build process
```
* [Optional] Development Installation
```bash
pip install -e .
```
* [Optional] Skip C++ Module Building
The Python modules provides all the same functions as the C++ modules, so you may consider to skip all the C++ module building if you only need to investigate the basic functions provided by MULLER.
```bash
BUILD_CPP=false pip install .
```
#### 3. Verify Installation
```python
import muller
print(muller.__version__)
```

## Reproduction steps for the experiment results in the paper

Please refer to [exp_scripts/README.md](https://github.com/spencerr221/MULLER/blob/main/exp_scripts/README.md) for the detailed steps.
