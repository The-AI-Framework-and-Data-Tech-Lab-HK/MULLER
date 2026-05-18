cmake_minimum_required(VERSION 3.5...4.0)
project(CustomHashMapPybind)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Set the basic paths
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(THIRDPARTY_DIR "${PROJECT_ROOT}/thirdparty")
set(VENDOR_DIR "${THIRDPARTY_DIR}/vendor")

# Set the dependency paths
set(PYBIND11_INCLUDE_DIR "${VENDOR_DIR}/pybind11/include")
set(CPPJIEBA_ROOT_DIR "${VENDOR_DIR}/cppjieba")
set(SPARSEHASH_INCLUDE_DIR "${VENDOR_DIR}/sparsehash/sparsehash-install/include")
set(BOOST_INCLUDE_DIR "${VENDOR_DIR}/boost/boost-install/include")
set(BOOST_LIBRARY_DIR "${VENDOR_DIR}/boost/boost-install/lib")
set(MURMURHASH_INCLUDE_DIR "${VENDOR_DIR}/murmurhash/murmurhash/include")
set(JIEBA_DICT_DIR "${CPPJIEBA_ROOT_DIR}/dict/")

# Set the jiaba dictionary path
add_definitions(-DJIEBA_DICT_DIR="${JIEBA_DICT_DIR}")

# Search for the Python interpreter selected by the build script, so editable
# installs inside a conda environment build against the same Python.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# The source files
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

# Construct the Python modules
add_library(custom_hash_map MODULE ${SOURCES})

# Set the Python module attribbutes
set_target_properties(custom_hash_map PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Include the directories
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

# Compile definition
target_compile_definitions(custom_hash_map PRIVATE
    LOGGING_LEVEL=LL_WARNING
)

# Link Libraries
target_link_libraries(custom_hash_map PRIVATE
    ${BOOST_LIBRARY_DIR}/libboost_system.a
    Python3::Module
)

if(NOT APPLE)
    target_link_libraries(custom_hash_map PRIVATE uuid stdc++fs)
endif()

# Release
target_compile_options(custom_hash_map PRIVATE -O3 -DNDEBUG)