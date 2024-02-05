# Machine learning Module

set(INCROOT ${PROJECT_SOURCE_DIR}/include/MLL/AI)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src/AI)

set(SOURCES
    ${SRCROOT}/Model.cpp
    ${SRCROOT}/Data.cpp
    ${SRCROOT}/Dense.cpp
    ${SRCROOT}/Activation.cpp
)

set(HEADERS
    ${INCROOT}/Model.h
    ${INCROOT}/Data.h
    ${INCROOT}/Layers/Layer.h
    ${INCROOT}/Layers/Dense.h
    ${INCROOT}/Layers/Activation.h
)

# Add dynamic Machine learning library
add_library(ml-ai SHARED ${SOURCES} ${HEADERS})
target_include_directories(ml-ai PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ml-ai PRIVATE ml-linear)

# Add static Machine learning library
add_library(ml-ai-s STATIC ${SOURCES} ${HEADERS})
target_include_directories(ml-ai-s PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ml-ai-s PRIVATE ml-linear-s)

# Add compiler optimizations
target_compile_options(ml-ai PRIVATE -Ofast -march=native)
target_compile_options(ml-ai-s PRIVATE -Ofast -march=native)