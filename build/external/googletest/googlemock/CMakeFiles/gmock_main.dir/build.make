# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\dev\MLL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\dev\MLL\build

# Include any dependencies generated for this target.
include external/googletest/googlemock/CMakeFiles/gmock_main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include external/googletest/googlemock/CMakeFiles/gmock_main.dir/compiler_depend.make

# Include the progress variables for this target.
include external/googletest/googlemock/CMakeFiles/gmock_main.dir/progress.make

# Include the compile flags for this target's objects.
include external/googletest/googlemock/CMakeFiles/gmock_main.dir/flags.make

external/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.obj: external/googletest/googlemock/CMakeFiles/gmock_main.dir/flags.make
external/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.obj: external/googletest/googlemock/CMakeFiles/gmock_main.dir/includes_CXX.rsp
external/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.obj: D:/dev/MLL/external/googletest/googlemock/src/gmock_main.cc
external/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.obj: external/googletest/googlemock/CMakeFiles/gmock_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=D:\dev\MLL\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.obj"
	cd /d D:\dev\MLL\build\external\googletest\googlemock && D:\w64devkit\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.obj -MF CMakeFiles\gmock_main.dir\src\gmock_main.cc.obj.d -o CMakeFiles\gmock_main.dir\src\gmock_main.cc.obj -c D:\dev\MLL\external\googletest\googlemock\src\gmock_main.cc

external/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/gmock_main.dir/src/gmock_main.cc.i"
	cd /d D:\dev\MLL\build\external\googletest\googlemock && D:\w64devkit\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\dev\MLL\external\googletest\googlemock\src\gmock_main.cc > CMakeFiles\gmock_main.dir\src\gmock_main.cc.i

external/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/gmock_main.dir/src/gmock_main.cc.s"
	cd /d D:\dev\MLL\build\external\googletest\googlemock && D:\w64devkit\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\dev\MLL\external\googletest\googlemock\src\gmock_main.cc -o CMakeFiles\gmock_main.dir\src\gmock_main.cc.s

# Object files for target gmock_main
gmock_main_OBJECTS = \
"CMakeFiles/gmock_main.dir/src/gmock_main.cc.obj"

# External object files for target gmock_main
gmock_main_EXTERNAL_OBJECTS =

lib/libgmock_main.a: external/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.obj
lib/libgmock_main.a: external/googletest/googlemock/CMakeFiles/gmock_main.dir/build.make
lib/libgmock_main.a: external/googletest/googlemock/CMakeFiles/gmock_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=D:\dev\MLL\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ..\..\..\lib\libgmock_main.a"
	cd /d D:\dev\MLL\build\external\googletest\googlemock && $(CMAKE_COMMAND) -P CMakeFiles\gmock_main.dir\cmake_clean_target.cmake
	cd /d D:\dev\MLL\build\external\googletest\googlemock && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\gmock_main.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/googletest/googlemock/CMakeFiles/gmock_main.dir/build: lib/libgmock_main.a
.PHONY : external/googletest/googlemock/CMakeFiles/gmock_main.dir/build

external/googletest/googlemock/CMakeFiles/gmock_main.dir/clean:
	cd /d D:\dev\MLL\build\external\googletest\googlemock && $(CMAKE_COMMAND) -P CMakeFiles\gmock_main.dir\cmake_clean.cmake
.PHONY : external/googletest/googlemock/CMakeFiles/gmock_main.dir/clean

external/googletest/googlemock/CMakeFiles/gmock_main.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\dev\MLL D:\dev\MLL\external\googletest\googlemock D:\dev\MLL\build D:\dev\MLL\build\external\googletest\googlemock D:\dev\MLL\build\external\googletest\googlemock\CMakeFiles\gmock_main.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : external/googletest/googlemock/CMakeFiles/gmock_main.dir/depend

