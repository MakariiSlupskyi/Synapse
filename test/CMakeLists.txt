if (BUILD_TESTING)
    add_executable(tests tests.cpp)

    target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(tests PRIVATE GTest::gtest_main synapse-s)

    include(GoogleTest)

    # Finds all the Google tests associated with the executable
    gtest_discover_tests(tests)
endif()

# Add test program executable
add_executable(test_program test_program.cpp)
target_include_directories(test_program PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link Synapse
target_link_libraries(test_program PRIVATE synapse-s)