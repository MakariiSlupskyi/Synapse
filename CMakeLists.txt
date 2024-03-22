cmake_minimum_required(VERSION 3.28)
project(Synapse
    VERSION 0.0.1
    LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)

# Set C++ Standart
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Enable testing
include(CTest)

# Add subdirectories
add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(test)