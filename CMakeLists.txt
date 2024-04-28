cmake_minimum_required(VERSION 3.22)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type")

project(Synapse
    VERSION 0.0.1
    LANGUAGES CXX)

# Set C++ Standart
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
include(CTest)

# Add subdirectories
add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(test)