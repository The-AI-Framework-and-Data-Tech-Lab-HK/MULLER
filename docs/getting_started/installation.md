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
