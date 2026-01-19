#!/bin/bash
# Copyright (c) 2026 Bingyu Liu. All rights reserved.

set -e

CUR_DIR=$(dirname $(readlink -f "$0"))
THIRDPARTY_DIR="${CUR_DIR}"  # GTN-F/thirdparty
VENDOR_DIR="${THIRDPARTY_DIR}/vendor"
BUILD_OUT_DIR="${VENDOR_DIR}/lib"
DEPS_SCRIPT="${CUR_DIR}/download_opensource.sh"

# 默认并行编译数
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

# 下载依赖
echo "=== Downloading dependencies ==="
if [ -f "${DEPS_SCRIPT}" ]; then
    bash "${DEPS_SCRIPT}" -T "${VENDOR_DIR}" -F "${CUR_DIR}/dependencies.csv"
else
    echo "Error: download_opensource.sh not found at ${DEPS_SCRIPT}"
    exit 1
fi

# 如果只下载，退出
if [ "${DOWNLOAD_ONLY}" = "1" ]; then
    echo "Download completed. Skipping build."
    exit 0
fi

# 创建输出目录
mkdir -p "${BUILD_OUT_DIR}"

# 清理构建
if [ "${CLEAN_BUILD}" = "1" ]; then
    echo "=== Cleaning previous builds ==="
    find "${VENDOR_DIR}" -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
    rm -rf "${BUILD_OUT_DIR}"/*
fi

# 构建 cppjieba
build_cppjieba() {
    echo "=== Building cppjieba ==="
    local cppjieba_dir="${VENDOR_DIR}/cppjieba"

    if [ ! -d "${cppjieba_dir}" ]; then
        echo "Error: cppjieba not found at ${cppjieba_dir}"
        return 1
    fi

    cd "${cppjieba_dir}"

    # 初始化子模块
    if [ -f ".gitmodules" ]; then
        git submodule init
        git submodule update
    fi

    # 创建并进入构建目录
    mkdir -p build
    cd build

    # 配置和编译
    cmake ..
    make -j${JOB_NUM}

    echo "cppjieba build completed"
}

# 构建 sparsehash
build_sparsehash() {
    echo "=== Building sparsehash ==="
    local sparsehash_dir="${VENDOR_DIR}/sparsehash"

    if [ ! -d "${sparsehash_dir}" ]; then
        echo "Error: sparsehash not found at ${sparsehash_dir}"
        return 1
    fi

    cd "${sparsehash_dir}"

    # 执行标准的三步构建
    echo "Running ./configure..."
    ./configure --prefix="${VENDOR_DIR}/sparsehash/install"

    echo "Running make..."
    make -j${JOB_NUM}

    echo "Running make install..."
    make install

    echo "sparsehash build completed"
}

# 构建 boost
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

    # 只编译需要的库
    ./b2 -j${JOB_NUM} \
        --with-system \
        variant=release \
        link=static \
        threading=multi \
        install

    echo "boost build completed"
}

# 检查仅头文件的库
check_header_only_libs() {
    echo "=== Checking header-only libraries ==="

    # murmurhash - 检查关键文件
    if [ -d "${VENDOR_DIR}/murmurhash" ]; then
        if [ -f "${VENDOR_DIR}/murmurhash/murmurhash/include/murmurhash/MurmurHash3.h" ] && \
           [ -f "${VENDOR_DIR}/murmurhash/murmurhash/MurmurHash3.cpp" ]; then
            echo "✓ murmurhash: found required files"
            echo "  - Header: ${VENDOR_DIR}/murmurhash/murmurhash/include/murmurhash/MurmurHash3.h"
            echo "  - Source: ${VENDOR_DIR}/murmurhash/murmurhash/MurmurHash3.cpp"

            # 复制源文件到third_party目录以便编译
            cp "${VENDOR_DIR}/murmurhash/murmurhash/MurmurHash3.cpp" "${THIRDPARTY_DIR}"
        else
            echo "✗ murmurhash: missing required files"
            return 1
        fi
    else
        echo "✗ murmurhash: directory not found"
        return 1
    fi

    # pybind11 - 检查头文件目录
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

# 主构建流程
echo "=== Starting build process ==="
echo "Parallel jobs: ${JOB_NUM}"

# 构建各个库
build_cppjieba
build_sparsehash
build_boost

# 检查仅头文件的库
check_header_only_libs

echo "=== Build completed ==="
echo "Libraries are in: ${BUILD_OUT_DIR}"