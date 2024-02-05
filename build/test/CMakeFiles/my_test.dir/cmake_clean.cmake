file(REMOVE_RECURSE
  "../bin/my_test.exe"
  "../bin/my_test.exe.manifest"
  "../bin/my_test.pdb"
  "../lib/libmy_test.dll.a"
  "CMakeFiles/my_test.dir/my_test.cpp.obj"
  "CMakeFiles/my_test.dir/my_test.cpp.obj.d"
  "my_test[1]_tests.cmake"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/my_test.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
