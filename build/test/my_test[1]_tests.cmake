add_test([=[TensorTesting.SettingData]=]  D:/dev/MLL/build/bin/my_test.exe [==[--gtest_filter=TensorTesting.SettingData]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TensorTesting.SettingData]=]  PROPERTIES WORKING_DIRECTORY D:/dev/MLL/build/test SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
add_test([=[TensorTesting.Comparising]=]  D:/dev/MLL/build/bin/my_test.exe [==[--gtest_filter=TensorTesting.Comparising]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TensorTesting.Comparising]=]  PROPERTIES WORKING_DIRECTORY D:/dev/MLL/build/test SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
add_test([=[TensorTesting.TrivialAddition]=]  D:/dev/MLL/build/bin/my_test.exe [==[--gtest_filter=TensorTesting.TrivialAddition]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TensorTesting.TrivialAddition]=]  PROPERTIES WORKING_DIRECTORY D:/dev/MLL/build/test SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
add_test([=[TensorTesting.TrivialSubtraction]=]  D:/dev/MLL/build/bin/my_test.exe [==[--gtest_filter=TensorTesting.TrivialSubtraction]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TensorTesting.TrivialSubtraction]=]  PROPERTIES WORKING_DIRECTORY D:/dev/MLL/build/test SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  my_test_TESTS TensorTesting.SettingData TensorTesting.Comparising TensorTesting.TrivialAddition TensorTesting.TrivialSubtraction)
