# Linear algebra module

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

# Add dynamic Linear algebra library
add_library(ml-linear SHARED ${SOURCES} ${HEADERS})
target_include_directories(ml-linear PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Add static Linear algebra library
add_library(ml-linear-s STATIC ${SOURCES} ${HEADERS})
target_include_directories(ml-linear-s PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Add compiler optimizations
target_compile_options(ml-linear PRIVATE -Ofast -march=native)
target_compile_options(ml-linear-s PRIVATE -Ofast -march=native)