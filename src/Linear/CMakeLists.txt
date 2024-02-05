# Linear algebra library

set(INCROOT ${PROJECT_SOURCE_DIR}/include/MLL/Linear)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src/Linear)

set(SOURCES
    ${SRCROOT}/Tensor.cpp
    ${SRCROOT}/Matrix.cpp
    ${SRCROOT}/Vector.cpp
)

set(HEADERS
    ${INCROOT}/Tensor.h
    ${INCROOT}/Matrix.h
    ${INCROOT}/Vector.h
)

find_package(OpenMP)

# Add dynamic Linear algebra library
add_library(ml-linear SHARED ${SOURCES} ${HEADERS})
target_include_directories(ml-linear PRIVATE ${INCROOT})

# Add static Linear algebra library
add_library(ml-linear-s STATIC ${SOURCES} ${HEADERS})
target_include_directories(ml-linear-s PRIVATE ${INCROOT})

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(ml-linear PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(ml-linear-s PRIVATE OpenMP::OpenMP_CXX)
endif()

# Add compiler optimizations
target_compile_options(ml-linear PRIVATE -Ofast -march=native)
target_compile_options(ml-linear-s PRIVATE -Ofast -march=native)