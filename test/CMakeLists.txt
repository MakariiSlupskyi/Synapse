if (BUILD_TESTING)
    add_executable(my_test my_test.cpp)

    target_include_directories(my_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(my_test PRIVATE GTest::gtest_main synapse-mll synapse-linear)

    include(GoogleTest)

    # Finds all the Google tests associated with the executable
    gtest_discover_tests(my_test)
endif()

# Add test program executable
add_executable(test_prog test_main.cpp)
target_include_directories(test_prog PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link Synapse
target_link_libraries(test_prog PRIVATE synapse-mll synapse-linear)
