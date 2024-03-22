set(INCROOT ${PROJECT_SOURCE_DIR}/include/Synapse/AI)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src/AI)

set(SOURCES
    ${SRCROOT}/functions/activ_funcs.cpp
    ${SRCROOT}/functions/loss_funcs.cpp
    ${SRCROOT}/functions/other_funcs.cpp

    ${SRCROOT}/layers/activation.cpp
    ${SRCROOT}/layers/convolutional.cpp
    ${SRCROOT}/layers/dense.cpp
    ${SRCROOT}/layers/flatten.cpp
    ${SRCROOT}/layers/pooling.cpp

    ${SRCROOT}/data.cpp
    ${SRCROOT}/dataset.cpp
    ${SRCROOT}/functions.cpp
    ${SRCROOT}/layers.cpp
    ${SRCROOT}/model.cpp
    ${SRCROOT}/optimizers.cpp
)

set(HEADERS
    ${INCROOT}/functions/activ_funcs.h
    ${INCROOT}/functions/loss_funcs.h
    ${INCROOT}/functions/other_funcs.h

    ${INCROOT}/layers/layer.h
    ${INCROOT}/layers/activation.h
    ${INCROOT}/layers/convolutional.h
    ${INCROOT}/layers/dense.h
    ${INCROOT}/layers/flatten.h
    ${INCROOT}/layers/pooling.h
    
    ${INCROOT}/optimizers/optimizer.h
    ${INCROOT}/optimizers/GD.h
    ${INCROOT}/optimizers/SGD.h

    ${INCROOT}/data.h
    ${INCROOT}/dataset.h
    ${INCROOT}/functions.h
    ${INCROOT}/layers.h
    ${INCROOT}/model.h
    ${INCROOT}/optimizers.h
)

# Add dynamic Machine learning library
add_library(synapse-mll SHARED ${SOURCES} ${HEADERS})
target_include_directories(synapse-mll PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(synapse-mll PUBLIC synapse-linear)