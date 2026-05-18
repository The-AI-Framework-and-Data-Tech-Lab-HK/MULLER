#!/bin/bash
# Copyright (c) 2026 Bingyu Liu. All rights reserved.

set -e

# Obtain the directory of the script. Avoid ``readlink -f`` because it is not
# available on a default macOS installation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARSEHASH_DIR="${SCRIPT_DIR}"  # MULLER/muller/util/sparsehash
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd) # MULLER
BUILD_DIR="${SPARSEHASH_DIR}/build"
THIRDPARTY_DIR="${PROJECT_ROOT}/thirdparty"
OUTPUT_DIR="${BUILD_DIR}"  # MULLER/muller/util/sparsehash/build
PYTHON_BIN="${PYTHON:-$(command -v python || command -v python3)}"

# Default args
CPU_NUM=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
JOB_NUM=$((CPU_NUM + 1))
BUILD_TYPE="Release"

# Show help
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build the custom_hash_map project with dependencies

Options:
    -j NUM    Number of parallel jobs (default: ${JOB_NUM})
    -t TYPE   Build type: Debug|Release (default: ${BUILD_TYPE})
    -c        Clean build
    -d        Download and build dependencies only
    -s        Skip dependency build
    -h        Show this help

Examples:
    $0                    # Full build
    $0 -j 8 -t Debug     # Debug build with 8 jobs
    $0 -c                # Clean and rebuild everything
    $0 -s                # Skip dependency build

EOF
}

# Build args
CLEAN_BUILD=0
DEPS_ONLY=0
SKIP_DEPS=0

while getopts "j:t:cdsh" opt; do
    case ${opt} in
        j)
            JOB_NUM="${OPTARG}"
            ;;
        t)
            BUILD_TYPE="${OPTARG}"
            ;;
        c)
            CLEAN_BUILD=1
            ;;
        d)
            DEPS_ONLY=1
            ;;
        s)
            SKIP_DEPS=1
            ;;
        h)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

# Clean the build caches
if [ ${CLEAN_BUILD} -eq 1 ]; then
    echo "=== Cleaning build directories ==="
    rm -rf "${BUILD_DIR}"
    rm -f "${OUTPUT_DIR}"/custom_hash_map*.so
    rm -f "${OUTPUT_DIR}"/custom_hash_map*.pyd
fi

# Building
if [ ${SKIP_DEPS} -eq 0 ]; then
    echo "=== Building dependencies ==="
    DEPS_SCRIPT="${THIRDPARTY_DIR}/build_deps.sh"
    if [ -f "${DEPS_SCRIPT}" ]; then
        cd "${THIRDPARTY_DIR}"
        bash build_deps.sh -j ${JOB_NUM}
        cd "${SCRIPT_DIR}"
    else
        echo "Error: build_deps.sh not found at ${DEPS_SCRIPT}"
        exit 1
    fi
fi

# If only the building of dependencies is needed, quit now.
if [ ${DEPS_ONLY} -eq 1 ]; then
    echo "Dependencies built successfully"
    exit 0
fi

# Build the project
echo "=== Building custom_hash_map project ==="
echo "Build type: ${BUILD_TYPE}"
echo "Parallel jobs: ${JOB_NUM}"
echo "Build directory: ${BUILD_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# Construct the build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configuration
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DPython3_EXECUTABLE="${PYTHON_BIN}"

# Building
echo "Building..."
cmake --build . --config ${BUILD_TYPE} -- -j${JOB_NUM}

# Check the outputs
echo "=== Build completed ==="
MODULE_FILE=$(find "${BUILD_DIR}" -name "custom_hash_map*.so" -o -name "custom_hash_map*.pyd" | head -1)

if [ -n "${MODULE_FILE}" ]; then
    echo "Success! Module built at: ${MODULE_FILE}"

else
    echo "Error: Module not found in build directory"
    exit 1
fi

# Tips for the following steps
echo ""
echo "Build successful! You can now use the module in Python:"
echo "  cd ${SPARSEHASH_DIR}/../../.."  # Go back to the MULLER directory
echo "  python -c \"from muller.util.sparsehash.build import custom_hash_map\""

cd "${PROJECT_ROOT}"
rm -rf build/ dist/ *.egg-info muller.egg-info/
echo "  Finish Cleaning the whl."