set(INCROOT ${PROJECT_SOURCE_DIR}/include/Synapse/linear)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src/linear)

set(SOURCES
    ${SRCROOT}/tensor.cpp
    ${SRCROOT}/tensor_funcs.cpp
)

set(HEADERS
    ${INCROOT}/tensor.h
    ${SRCROOT}/tensor_funcs.cpp
)

# Add dynamic Linear algebra library
add_library(synapse-linear SHARED ${SOURCES} ${HEADERS})
target_include_directories(synapse-linear PUBLIC ${PROJECT_SOURCE_DIR}/include)