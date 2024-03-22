set(INCROOT ${PROJECT_SOURCE_DIR}/include/Synapse/linear)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src/linear)

set(SOURCES
    ${SRCROOT}/tensor.cpp
    ${SRCROOT}/matrix.cpp
    ${SRCROOT}/vector.cpp
)

set(HEADERS
    ${INCROOT}/tensor.h
    ${INCROOT}/matrix.h
    ${INCROOT}/vector.h
)

# Add dynamic Linear algebra library
add_library(synapse-linear SHARED ${SOURCES} ${HEADERS})
target_include_directories(synapse-linear PUBLIC ${PROJECT_SOURCE_DIR}/include)