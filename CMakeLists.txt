cmake_minimum_required(VERSION 3.28)
project(Test 
    VERSION 0.0.1
    LANGUAGES CXX)

# Set C++ Standart
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Enable testing
include(CTest)
enable_testing()

# Add subdirectories
add_subdirectory(external)
add_subdirectory(src/AI)
add_subdirectory(src/Linear)
add_subdirectory(test)

# Link OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
#    target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
endif()