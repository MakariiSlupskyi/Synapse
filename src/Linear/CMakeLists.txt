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
add_library(ml-linear SHARED ${SOURCES} ${HEADERS})
target_include_directories(ml-linear PUBLIC ${PROJECT_SOURCE_DIR}/include)

# # Add static Linear algebra library
add_library(ml-linear-s STATIC ${SOURCES} ${HEADERS})
target_include_directories(ml-linear-s PUBLIC ${PROJECT_SOURCE_DIR}/include)