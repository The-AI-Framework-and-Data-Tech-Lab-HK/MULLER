cmake_minimum_required(VERSION 3.4)
project(CustomHashMapPybind)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 设置基础路径
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(THIRDPARTY_DIR "${PROJECT_ROOT}/thirdparty")
set(VENDOR_DIR "${THIRDPARTY_DIR}/vendor")

# 依赖路径
set(PYBIND11_INCLUDE_DIR "${VENDOR_DIR}/pybind11/include")
set(CPPJIEBA_ROOT_DIR "${VENDOR_DIR}/cppjieba")
set(SPARSEHASH_INCLUDE_DIR "${VENDOR_DIR}/sparsehash/install/include")
set(BOOST_INCLUDE_DIR "${VENDOR_DIR}/boost/install/include")
set(BOOST_LIBRARY_DIR "${VENDOR_DIR}/boost/install/lib")
set(MURMURHASH_INCLUDE_DIR "${VENDOR_DIR}/murmurhash/murmurhash/include")
set(JIEBA_DICT_DIR "${CPPJIEBA_ROOT_DIR}/dict/")

# 定义结巴词典路径
add_definitions(-DJIEBA_DICT_DIR="${JIEBA_DICT_DIR}")

# 查找Python
find_package(Python3 3.11 EXACT REQUIRED COMPONENTS Development)

# 源文件
set(SOURCES
    src/value_set.cpp
    src/custom_hash_map.cpp
    src/thread_pool.cpp
    src/jieba_utils.cpp
    src/index_utils.cpp
    src/index_processor.cpp
    src/chunk.cpp
    src/chunk_id_encoder.cpp
    src/reader.cpp
    src/search_utils.cpp
    src/pybind_module.cpp
    src/gil_manager.cpp
    src/sparse_key_dense_value_map.cpp
    ${THIRDPARTY_DIR}/MurmurHash3.cpp
)

# 创建Python模块
add_library(custom_hash_map MODULE ${SOURCES})

# 设置Python模块属性
set_target_properties(custom_hash_map PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# 包含目录
target_include_directories(custom_hash_map PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${THIRDPARTY_DIR}
    ${CPPJIEBA_ROOT_DIR}/include
    ${CPPJIEBA_ROOT_DIR}/deps/limonp/include
    ${THIRDPARTY_DIR}/vendor/cppjieba/deps
    ${SPARSEHASH_INCLUDE_DIR}
    ${MURMURHASH_INCLUDE_DIR}
    ${BOOST_INCLUDE_DIR}
    ${PYBIND11_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)

# 编译定义
target_compile_definitions(custom_hash_map PRIVATE
    LOGGING_LEVEL=LL_WARNING
)

# 链接库
target_link_libraries(custom_hash_map PRIVATE
    ${BOOST_LIBRARY_DIR}/libboost_system.a
    ${Python3_LIBRARIES}
    uuid
    stdc++fs
)

# Release优化
target_compile_options(custom_hash_map PRIVATE -O3 -DNDEBUG)