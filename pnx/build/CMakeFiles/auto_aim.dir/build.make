# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mozijun/Mycode_c/pnx

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mozijun/Mycode_c/pnx/build

# Include any dependencies generated for this target.
include CMakeFiles/auto_aim.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/auto_aim.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/auto_aim.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/auto_aim.dir/flags.make

CMakeFiles/auto_aim.dir/main.cpp.o: CMakeFiles/auto_aim.dir/flags.make
CMakeFiles/auto_aim.dir/main.cpp.o: ../main.cpp
CMakeFiles/auto_aim.dir/main.cpp.o: CMakeFiles/auto_aim.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mozijun/Mycode_c/pnx/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/auto_aim.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/auto_aim.dir/main.cpp.o -MF CMakeFiles/auto_aim.dir/main.cpp.o.d -o CMakeFiles/auto_aim.dir/main.cpp.o -c /home/mozijun/Mycode_c/pnx/main.cpp

CMakeFiles/auto_aim.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/auto_aim.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mozijun/Mycode_c/pnx/main.cpp > CMakeFiles/auto_aim.dir/main.cpp.i

CMakeFiles/auto_aim.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/auto_aim.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mozijun/Mycode_c/pnx/main.cpp -o CMakeFiles/auto_aim.dir/main.cpp.s

CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.o: CMakeFiles/auto_aim.dir/flags.make
CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.o: ../armor_detector/src/detector.cpp
CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.o: CMakeFiles/auto_aim.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mozijun/Mycode_c/pnx/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.o -MF CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.o.d -o CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.o -c /home/mozijun/Mycode_c/pnx/armor_detector/src/detector.cpp

CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mozijun/Mycode_c/pnx/armor_detector/src/detector.cpp > CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.i

CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mozijun/Mycode_c/pnx/armor_detector/src/detector.cpp -o CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.s

CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.o: CMakeFiles/auto_aim.dir/flags.make
CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.o: ../armor_detector/src/number_classifier.cpp
CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.o: CMakeFiles/auto_aim.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mozijun/Mycode_c/pnx/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.o -MF CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.o.d -o CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.o -c /home/mozijun/Mycode_c/pnx/armor_detector/src/number_classifier.cpp

CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mozijun/Mycode_c/pnx/armor_detector/src/number_classifier.cpp > CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.i

CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mozijun/Mycode_c/pnx/armor_detector/src/number_classifier.cpp -o CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.s

CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.o: CMakeFiles/auto_aim.dir/flags.make
CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.o: ../armor_detector/src/pnp_solver.cpp
CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.o: CMakeFiles/auto_aim.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mozijun/Mycode_c/pnx/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.o -MF CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.o.d -o CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.o -c /home/mozijun/Mycode_c/pnx/armor_detector/src/pnp_solver.cpp

CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mozijun/Mycode_c/pnx/armor_detector/src/pnp_solver.cpp > CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.i

CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mozijun/Mycode_c/pnx/armor_detector/src/pnp_solver.cpp -o CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.s

CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.o: CMakeFiles/auto_aim.dir/flags.make
CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.o: ../armor_detector/src/armor.cpp
CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.o: CMakeFiles/auto_aim.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mozijun/Mycode_c/pnx/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.o -MF CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.o.d -o CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.o -c /home/mozijun/Mycode_c/pnx/armor_detector/src/armor.cpp

CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mozijun/Mycode_c/pnx/armor_detector/src/armor.cpp > CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.i

CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mozijun/Mycode_c/pnx/armor_detector/src/armor.cpp -o CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.s

CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.o: CMakeFiles/auto_aim.dir/flags.make
CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.o: ../armor_tracker/src/tracker.cpp
CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.o: CMakeFiles/auto_aim.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mozijun/Mycode_c/pnx/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.o -MF CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.o.d -o CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.o -c /home/mozijun/Mycode_c/pnx/armor_tracker/src/tracker.cpp

CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mozijun/Mycode_c/pnx/armor_tracker/src/tracker.cpp > CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.i

CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mozijun/Mycode_c/pnx/armor_tracker/src/tracker.cpp -o CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.s

# Object files for target auto_aim
auto_aim_OBJECTS = \
"CMakeFiles/auto_aim.dir/main.cpp.o" \
"CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.o" \
"CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.o" \
"CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.o" \
"CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.o" \
"CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.o"

# External object files for target auto_aim
auto_aim_EXTERNAL_OBJECTS =

../bin/auto_aim: CMakeFiles/auto_aim.dir/main.cpp.o
../bin/auto_aim: CMakeFiles/auto_aim.dir/armor_detector/src/detector.cpp.o
../bin/auto_aim: CMakeFiles/auto_aim.dir/armor_detector/src/number_classifier.cpp.o
../bin/auto_aim: CMakeFiles/auto_aim.dir/armor_detector/src/pnp_solver.cpp.o
../bin/auto_aim: CMakeFiles/auto_aim.dir/armor_detector/src/armor.cpp.o
../bin/auto_aim: CMakeFiles/auto_aim.dir/armor_tracker/src/tracker.cpp.o
../bin/auto_aim: CMakeFiles/auto_aim.dir/build.make
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
../bin/auto_aim: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
../bin/auto_aim: CMakeFiles/auto_aim.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mozijun/Mycode_c/pnx/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable ../bin/auto_aim"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/auto_aim.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/auto_aim.dir/build: ../bin/auto_aim
.PHONY : CMakeFiles/auto_aim.dir/build

CMakeFiles/auto_aim.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/auto_aim.dir/cmake_clean.cmake
.PHONY : CMakeFiles/auto_aim.dir/clean

CMakeFiles/auto_aim.dir/depend:
	cd /home/mozijun/Mycode_c/pnx/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mozijun/Mycode_c/pnx /home/mozijun/Mycode_c/pnx /home/mozijun/Mycode_c/pnx/build /home/mozijun/Mycode_c/pnx/build /home/mozijun/Mycode_c/pnx/build/CMakeFiles/auto_aim.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/auto_aim.dir/depend

