#!/bin/bash
# Copyright (c) 2026 Bingyu Liu. All rights reserved.

set -e

CUR_DIR=$(dirname $(readlink -f "$0"))
THIRDPARTY_DIR="${CUR_DIR}"  # MULLER/thirdparty
VENDOR_DIR="${THIRDPARTY_DIR}/vendor"
BUILD_OUT_DIR="${VENDOR_DIR}/lib"
DEPS_SCRIPT="${CUR_DIR}/download_opensource.sh"

# Default number of parallel compilation jobs.
CPU_NUM=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
JOB_NUM=$((CPU_NUM + 1))

while getopts 'j:cd' opt; do
    case "$opt" in
    j)
        if [ ${OPTARG} -gt $((CPU_NUM * 2)) ]; then
            echo "Warning: The -j ${OPTARG} is over the max logical cpu count($CPU_NUM) * 2"
        fi
        JOB_NUM="${OPTARG}"
        ;;
    c)
        CLEAN_BUILD=1
        ;;
    d)
        DOWNLOAD_ONLY=1
        ;;
    *)
        echo "Usage: $0 [-j jobs] [-c] [-d]"
        echo "  -j: Number of parallel jobs"
        echo "  -c: Clean build"
        echo "  -d: Download only, no build"
        exit 1
        ;;
    esac
done

# Downloading dependencies
echo "=== Downloading dependencies ==="
if [ -f "${DEPS_SCRIPT}" ]; then
    bash "${DEPS_SCRIPT}" -T "${VENDOR_DIR}" -F "${CUR_DIR}/dependencies.csv"
else
    echo "Error: download_opensource.sh not found at ${DEPS_SCRIPT}"
    exit 1
fi

# If it is download_only, then you can skip the building
if [ "${DOWNLOAD_ONLY}" = "1" ]; then
    echo "Download completed. Skipping build."
    exit 0
fi

# Make a build directory
mkdir -p "${BUILD_OUT_DIR}"

# Clean the build files
if [ "${CLEAN_BUILD}" = "1" ]; then
    echo "=== Cleaning previous builds ==="
    find "${VENDOR_DIR}" -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
    rm -rf "${BUILD_OUT_DIR}"/*
fi

# Build cppjieba
build_cppjieba() {
    echo "=== Building cppjieba ==="
    local cppjieba_dir="${VENDOR_DIR}/cppjieba"

    if [ ! -d "${cppjieba_dir}" ]; then
        echo "Error: cppjieba not found at ${cppjieba_dir}"
        return 1
    fi

    cd "${cppjieba_dir}"

    # Initialize submodules.
    if [ -f ".gitmodules" ]; then
        git submodule init
        git submodule update
    fi

    # Create and enter the build directory.
    mkdir -p build
    cd build

    # Configure and compile.
    cmake ..
    make -j${JOB_NUM}

    echo "cppjieba build completed"
}

# Construct sparsehash
build_sparsehash() {
    echo "=== Building sparsehash ==="
    local sparsehash_dir="${VENDOR_DIR}/sparsehash"

    if [ ! -d "${sparsehash_dir}" ]; then
        echo "Error: sparsehash not found at ${sparsehash_dir}"
        return 1
    fi

    cd "${sparsehash_dir}"

    # Perform the standard three-step build process.
    echo "Running ./configure..."
    ./configure --prefix="${VENDOR_DIR}/sparsehash/install"

    echo "Running make..."
    make -j${JOB_NUM}

    echo "Running make install..."
    make install

    echo "sparsehash build completed"
}

# Construct boost
build_boost() {
    echo "=== Building boost ==="
    local boost_dir="${VENDOR_DIR}/boost"

    if [ ! -d "${boost_dir}" ]; then
        echo "Error: boost not found at ${boost_dir}"
        return 1
    fi

    cd "${boost_dir}"

    # Bootstrap
    ./bootstrap.sh --prefix="${boost_dir}/install"

    # Compile only the required libraries.
    ./b2 -j${JOB_NUM} \
        --with-system \
        variant=release \
        link=static \
        threading=multi \
        install

    echo "boost build completed"
}

# Check for header-only libraries.
check_header_only_libs() {
    echo "=== Checking header-only libraries ==="

    # murmurhash - Check the key files
    if [ -d "${VENDOR_DIR}/murmurhash" ]; then
        if [ -f "${VENDOR_DIR}/murmurhash/murmurhash/include/murmurhash/MurmurHash3.h" ] && \
           [ -f "${VENDOR_DIR}/murmurhash/murmurhash/MurmurHash3.cpp" ]; then
            echo "✓ murmurhash: found required files"
            echo "  - Header: ${VENDOR_DIR}/murmurhash/murmurhash/include/murmurhash/MurmurHash3.h"
            echo "  - Source: ${VENDOR_DIR}/murmurhash/murmurhash/MurmurHash3.cpp"

            # Copy source files to the third_party directory for compilation.
            cp "${VENDOR_DIR}/murmurhash/murmurhash/MurmurHash3.cpp" "${THIRDPARTY_DIR}"
        else
            echo "✗ murmurhash: missing required files"
            return 1
        fi
    else
        echo "✗ murmurhash: directory not found"
        return 1
    fi

    # pybind11 - Check the header files
    if [ -d "${VENDOR_DIR}/pybind11" ]; then
        if [ -d "${VENDOR_DIR}/pybind11/include/pybind11" ]; then
            echo "✓ pybind11: header-only library found"
            echo "  - Headers: ${VENDOR_DIR}/pybind11/include/pybind11/"
        else
            echo "✗ pybind11: headers not found in expected location"
            return 1
        fi
    else
        echo "✗ pybind11: directory not found"
        return 1
    fi
}

# Main build process.
echo "=== Starting build process ==="
echo "Parallel jobs: ${JOB_NUM}"

# Build each library.
build_cppjieba
build_sparsehash
build_boost

# Check for header-only libraries.
check_header_only_libs

echo "=== Build completed ==="
echo "Libraries are in: ${BUILD_OUT_DIR}"