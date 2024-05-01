set(INCROOT ${PROJECT_SOURCE_DIR}/include/Synapse)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src)

set(SOURCES
    # Linear Algebra
    ${SRCROOT}/linear/tensor.cpp
    ${SRCROOT}/linear/tensor_funcs.cpp

    # AI
    ${SRCROOT}/AI/automated/RL/generation.cpp
    ${SRCROOT}/AI/automated/RL/entity.cpp
    ${SRCROOT}/AI/automated/SL/create.cpp
    ${SRCROOT}/AI/automated/SL/model_builder.cpp

    ${SRCROOT}/AI/functions/activ_funcs.cpp
    ${SRCROOT}/AI/functions/loss_funcs.cpp
    ${SRCROOT}/AI/functions/other_funcs.cpp

    ${SRCROOT}/AI/layers/activation.cpp
    ${SRCROOT}/AI/layers/convolutional.cpp
    ${SRCROOT}/AI/layers/dense.cpp
    ${SRCROOT}/AI/layers/flatten.cpp
    ${SRCROOT}/AI/layers/pooling.cpp

    ${SRCROOT}/AI/data.cpp
    ${SRCROOT}/AI/dataset.cpp
    ${SRCROOT}/AI/functions.cpp
    ${SRCROOT}/AI/layers.cpp
    ${SRCROOT}/AI/model.cpp
    ${SRCROOT}/AI/optimizers.cpp
)

set(HEADERS
    # Linear Algebra
    ${INCROOT}/linear/tensor.h
    ${INCROOT}/linear/tensor_funcs.h
    ${INCROOT}/linear.h
    ${INCROOT}/AI.h

    # AI
    ${INCROOT}/AI/automated/RL/generation.h
    ${INCROOT}/AI/automated/RL/entity.h
    ${INCROOT}/AI/automated/SL/create.h
    ${INCROOT}/AI/automated/SL/model_builder.h
    ${INCROOT}/AI/automated/RL.h
    ${INCROOT}/AI/automated/SL.h

    ${INCROOT}/AI/functions/activ_funcs.h
    ${INCROOT}/AI/functions/loss_funcs.h
    ${INCROOT}/AI/functions/other_funcs.h

    ${INCROOT}/AI/layers/layer.h
    ${INCROOT}/AI/layers/activation.h
    ${INCROOT}/AI/layers/convolutional.h
    ${INCROOT}/AI/layers/dense.h
    ${INCROOT}/AI/layers/flatten.h
    ${INCROOT}/AI/layers/pooling.h
    
    ${INCROOT}/AI/optimizers/optimizer.h
    ${INCROOT}/AI/optimizers/GD.h
    ${INCROOT}/AI/optimizers/SGD.h

    ${INCROOT}/AI/automated.h
    ${INCROOT}/AI/data.h
    ${INCROOT}/AI/dataset.h
    ${INCROOT}/AI/functions.h
    ${INCROOT}/AI/layers.h
    ${INCROOT}/AI/model.h
    ${INCROOT}/AI/optimizers.h
)

# Add dynamic synapce library
add_library(synapse SHARED ${SOURCES} ${HEADERS})
target_include_directories(synapse PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Add static synapce library
add_library(synapse-s STATIC ${SOURCES} ${HEADERS})
target_include_directories(synapse-s PUBLIC ${PROJECT_SOURCE_DIR}/include)

set_target_properties(synapse-s synapse
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Synapse/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Synapse/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Synapse/lib"
)