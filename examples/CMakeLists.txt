# Set example folder path
set(EXAMPLE_PATH ${PROJECT_SOURCE_DIR}/examples)

# Define a macro that will create a Synapse example
macro(add_synapse_example TARGET_NAME SOURCES)
    # Create an executable
    add_executable(${TARGET_NAME} ${SOURCES})

    # Set the output directory for the target
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Synapse/examples")

    # Include Synapse
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # Link Synapse
    target_link_libraries(${TARGET_NAME} PRIVATE synapse-s)
endmacro(add_synapse_example)


# Add examples subdirectories
add_subdirectory(trivial-usage)
add_subdirectory(creating-cnn)
add_subdirectory(parsing-data)