# Linear algebra submodule's directories
set(LA_INCROOT ${PROJECT_SOURCE_DIR}/include/Synapse/linear)
set(LA_SRCROOT ${PROJECT_SOURCE_DIR}/src/linear)

# AI submodule's directories
set(AI_INCROOT ${PROJECT_SOURCE_DIR}/include/Synapse/AI)
set(AI_SRCROOT ${PROJECT_SOURCE_DIR}/src/AI)

set(SOURCES
    ${LA_SRCROOT}/tensor.cpp
    ${LA_SRCROOT}/tensor_funcs.cpp

    ${AI_SRCROOT}/functions/activ_funcs.cpp
    ${AI_SRCROOT}/functions/loss_funcs.cpp
    ${AI_SRCROOT}/functions/other_funcs.cpp

    ${AI_SRCROOT}/layers/activation.cpp
    ${AI_SRCROOT}/layers/convolutional.cpp
    ${AI_SRCROOT}/layers/dense.cpp
    ${AI_SRCROOT}/layers/flatten.cpp
    ${AI_SRCROOT}/layers/pooling.cpp

    ${AI_SRCROOT}/data.cpp
    ${AI_SRCROOT}/dataset.cpp
    ${AI_SRCROOT}/functions.cpp
    ${AI_SRCROOT}/layers.cpp
    ${AI_SRCROOT}/model.cpp
    ${AI_SRCROOT}/optimizers.cpp
)

set(HEADERS
    ${LA_INCROOT}/tensor.h
    ${LA_INCROOT}/tensor_funcs.h

    ${AI_INCROOT}/functions/activ_funcs.h
    ${AI_INCROOT}/functions/loss_funcs.h
    ${AI_INCROOT}/functions/other_funcs.h

    ${AI_INCROOT}/layers/layer.h
    ${AI_INCROOT}/layers/activation.h
    ${AI_INCROOT}/layers/convolutional.h
    ${AI_INCROOT}/layers/dense.h
    ${AI_INCROOT}/layers/flatten.h
    ${AI_INCROOT}/layers/pooling.h
    
    ${AI_INCROOT}/optimizers/optimizer.h
    ${AI_INCROOT}/optimizers/GD.h
    ${AI_INCROOT}/optimizers/SGD.h

    ${AI_INCROOT}/data.h
    ${AI_INCROOT}/dataset.h
    ${AI_INCROOT}/functions.h
    ${AI_INCROOT}/layers.h
    ${AI_INCROOT}/model.h
    ${AI_INCROOT}/optimizers.h
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