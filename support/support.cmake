#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.4
https://github.com/FrancoisCarouge/Kalman

SPDX-License-Identifier: Unlicense

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org> ]]

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Workaround: CMake does not record MSVC as supporting the C++26 language
  # dialect yet, so requesting it through `CMAKE_CXX_STANDARD` hard-fails
  # configuration, including inside fetched dependencies' own checks (for
  # example mp-units' `check_cxx_feature_supported`). Leave the standard unset
  # on MSVC and rely on `/std:c++latest`, set in `support/CMakeLists.txt`, to
  # opt in to C++26 features directly.
  unset(CMAKE_CXX_STANDARD)
endif()

check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT _)

# Interprocedural optimization defaults on for the benchmark drivers, unless
# explicitly disabled: coverage instrumentation produces no data from link-time
# optimized objects.
if(DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION
   AND NOT CMAKE_INTERPROCEDURAL_OPTIMIZATION)
  set(IPO_SUPPORTED OFF)
endif()

# Add a given sample.
#
# * NAME The name of the sample file without extension.
# * BACKENDS Optional list of backends to use against the sample.
function(sample NAME)
  set(multiValueArgs BACKENDS)
  cmake_parse_arguments(PARSE_ARGV 0 SAMPLE "" "${oneValueArgs}"
                        "${multiValueArgs}")

  if(NOT SAMPLE_BACKENDS)
    add_executable(kalman_sample_${NAME}_driver "${NAME}.cpp")
    target_link_libraries(
      kalman_sample_${NAME}_driver
      PRIVATE kalman kalman_main kalman_support_options kalman_unit_mp_units)
    separate_arguments(SAMPLE_COMMAND UNIX_COMMAND $ENV{COMMAND})
    add_test(
      NAME kalman_sample_${NAME}
      COMMAND ${SAMPLE_COMMAND} $<TARGET_FILE:kalman_sample_${NAME}_driver>)
  else()
    foreach(BACKEND IN ITEMS ${SAMPLE_BACKENDS})
      add_executable(kalman_sample_${BACKEND}_${NAME}_driver "${NAME}.cpp")
      target_link_libraries(
        kalman_sample_${BACKEND}_${NAME}_driver
        PRIVATE
          kalman kalman_main kalman_linalg_${BACKEND} kalman_support_options)
      separate_arguments(SAMPLE_COMMAND UNIX_COMMAND $ENV{COMMAND})
      add_test(
        NAME kalman_sample_${BACKEND}_${NAME}
        COMMAND
          ${SAMPLE_COMMAND}
          $<TARGET_FILE:kalman_sample_${BACKEND}_${NAME}_driver>)
    endforeach()
  endif()
endfunction(sample)

# Add a given run-time test.
#
# The test passes when its file compiles, links, and runs to completion: its
# verifications are assertions. The test is named after the directory it is
# declared in, its feature, and its file name.
#
# * NAME The name of the test file without extension.
# * BACKENDS Optional list of backends to use against the test.
function(pass NAME)
  set(multiValueArgs BACKENDS)
  cmake_parse_arguments(PARSE_ARGV 0 TEST "" "${oneValueArgs}"
                        "${multiValueArgs}")

  get_filename_component(CALLER "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

  if(NOT TEST_BACKENDS)
    add_executable(kalman_${CALLER}_${NAME}_pass_driver "${NAME}.cpp")
    target_link_libraries(
      kalman_${CALLER}_${NAME}_pass_driver
      PRIVATE kalman kalman_main kalman_support_options kalman_unit_mp_units)
    separate_arguments(TEST_COMMAND UNIX_COMMAND $ENV{COMMAND})
    add_test(
      NAME kalman_${CALLER}_${NAME}_pass
      COMMAND
        ${TEST_COMMAND} $<TARGET_FILE:kalman_${CALLER}_${NAME}_pass_driver>)
  else()
    foreach(BACKEND IN ITEMS ${TEST_BACKENDS})
      add_executable(kalman_${BACKEND}_${CALLER}_${NAME}_pass_driver
                     "${NAME}.cpp")
      target_link_libraries(
        kalman_${BACKEND}_${CALLER}_${NAME}_pass_driver
        PRIVATE
          kalman kalman_main kalman_linalg_${BACKEND} kalman_support_options)
      separate_arguments(TEST_COMMAND UNIX_COMMAND $ENV{COMMAND})
      add_test(
        NAME kalman_${BACKEND}_${CALLER}_${NAME}_pass
        COMMAND
          ${TEST_COMMAND}
          $<TARGET_FILE:kalman_${BACKEND}_${CALLER}_${NAME}_pass_driver>)
    endforeach()
  endif()
endfunction(pass)

# Add a given compile-time-only test.
#
# The test passes when its file compiles: its verifications are static
# assertions. No executable is linked nor run. The test is named after the
# directory it is declared in, its feature, and its file name.
#
# * NAME The name of the test file without extension.
# * BACKENDS Optional list of backends to use against the test.
function(build NAME)
  set(multiValueArgs BACKENDS)
  cmake_parse_arguments(PARSE_ARGV 0 TEST "" "${oneValueArgs}"
                        "${multiValueArgs}")

  get_filename_component(CALLER "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

  if(NOT TEST_BACKENDS)
    add_library(kalman_${CALLER}_${NAME}_build_driver OBJECT "${NAME}.cpp")
    target_link_libraries(
      kalman_${CALLER}_${NAME}_build_driver
      PRIVATE kalman kalman_support_options kalman_unit_mp_units)
    set_target_properties(
      kalman_${CALLER}_${NAME}_build_driver
      PROPERTIES EXCLUDE_FROM_ALL TRUE)
    add_test(
      NAME kalman_${CALLER}_${NAME}_pass
      COMMAND
        ${CMAKE_COMMAND}
        --build
        ${CMAKE_BINARY_DIR}
        --config
        $<CONFIG>
        --target
        kalman_${CALLER}_${NAME}_build_driver)
    set_tests_properties(
      kalman_${CALLER}_${NAME}_pass
      PROPERTIES RESOURCE_LOCK "kalman_build_tree")
  else()
    foreach(BACKEND IN ITEMS ${TEST_BACKENDS})
      add_library(
        kalman_${BACKEND}_${CALLER}_${NAME}_build_driver
        OBJECT "${NAME}.cpp")
      target_link_libraries(
        kalman_${BACKEND}_${CALLER}_${NAME}_build_driver
        PRIVATE kalman kalman_linalg_${BACKEND} kalman_support_options)
      set_target_properties(
        kalman_${BACKEND}_${CALLER}_${NAME}_build_driver
        PROPERTIES EXCLUDE_FROM_ALL TRUE)
      add_test(
        NAME kalman_${BACKEND}_${CALLER}_${NAME}_pass
        COMMAND
          ${CMAKE_COMMAND}
          --build
          ${CMAKE_BINARY_DIR}
          --config
          $<CONFIG>
          --target
          kalman_${BACKEND}_${CALLER}_${NAME}_build_driver)
      set_tests_properties(
        kalman_${BACKEND}_${CALLER}_${NAME}_pass
        PROPERTIES RESOURCE_LOCK "kalman_build_tree")
    endforeach()
  endif()
endfunction(build)

# Add a given compile-fail test.
#
# The test passes when its file fails to compile. By convention, the file name
# ends in `_fail`, and the valid form of the rejected line is kept in an
# `// Intended:` comment above it. The test is named after the directory it is
# declared in, its feature, and its file name.
#
# * NAME The name of the test file without extension.
# * BACKENDS Optional list of backends to use against the test.
function(fail NAME)
  set(multiValueArgs BACKENDS)
  cmake_parse_arguments(PARSE_ARGV 0 TEST "" "${oneValueArgs}"
                        "${multiValueArgs}")

  get_filename_component(CALLER "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

  if(NOT TEST_BACKENDS)
    add_library(kalman_${CALLER}_${NAME}_driver OBJECT "${NAME}.cpp")
    target_link_libraries(
      kalman_${CALLER}_${NAME}_driver
      PRIVATE kalman kalman_support_options kalman_unit_mp_units)
    set_target_properties(
      kalman_${CALLER}_${NAME}_driver
      PROPERTIES EXCLUDE_FROM_ALL TRUE)
    add_test(
      NAME kalman_${CALLER}_${NAME}
      COMMAND
        ${CMAKE_COMMAND}
        --build
        ${CMAKE_BINARY_DIR}
        --config
        $<CONFIG>
        --target
        kalman_${CALLER}_${NAME}_driver)
    set_tests_properties(
      kalman_${CALLER}_${NAME}
      PROPERTIES RESOURCE_LOCK "kalman_build_tree" WILL_FAIL TRUE)
  else()
    foreach(BACKEND IN ITEMS ${TEST_BACKENDS})
      add_library(
        kalman_${BACKEND}_${CALLER}_${NAME}_driver
        OBJECT "${NAME}.cpp")
      target_link_libraries(
        kalman_${BACKEND}_${CALLER}_${NAME}_driver
        PRIVATE kalman kalman_linalg_${BACKEND} kalman_support_options)
      set_target_properties(
        kalman_${BACKEND}_${CALLER}_${NAME}_driver
        PROPERTIES EXCLUDE_FROM_ALL TRUE)
      add_test(
        NAME kalman_${BACKEND}_${CALLER}_${NAME}
        COMMAND
          ${CMAKE_COMMAND}
          --build
          ${CMAKE_BINARY_DIR}
          --config
          $<CONFIG>
          --target
          kalman_${BACKEND}_${CALLER}_${NAME}_driver)
      set_tests_properties(
        kalman_${BACKEND}_${CALLER}_${NAME}
        PROPERTIES RESOURCE_LOCK "kalman_build_tree" WILL_FAIL TRUE)
    endforeach()
  endif()
endfunction(fail)

# Add a given sample benchmark.
#
# The sample is compiled unchanged into an object, measured for its compile
# time and sizes, and into a driver, with the benchmark header included ahead,
# measuring its filter's first prediction and update. The test writes the
# sample's `<name>_benchmark.md` documentation: in its sample directory with
# `BUILD_BENCHMARKING`, in the build tree otherwise. The test is named after the
# sample: `kalman_sample_[<backend>_]<name>_bench`. The calling directory
# provides the `nanobench` target, the benchmark `HEADER` compile option, the
# `IGNORELIST` sanitizer option, and the `bench.cmake` script.
#
# * NAME The name of the sample.
# * BACKENDS Optional list of backends to use against the sample.
function(bench NAME)
  cmake_parse_arguments(PARSE_ARGV 1 BENCH "" "" "BACKENDS")

  if(NOT BENCH_BACKENDS)
    set(BENCH_BACKENDS "none")
  endif()

  set(SOURCE "${PROJECT_SOURCE_DIR}/sample/${NAME}/${NAME}.cpp")
  set(DOCUMENT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}_benchmark.md")
  if(BUILD_BENCHMARKING)
    set(DOCUMENT "${PROJECT_SOURCE_DIR}/sample/${NAME}/${NAME}_benchmark.md")
  endif()
  list(JOIN BENCH_BACKENDS "," BACKENDS)

  foreach(BACKEND IN LISTS BENCH_BACKENDS)
    if(BACKEND STREQUAL "none")
      set(SAMPLE kalman_sample_${NAME})
      set(LIBRARY kalman_unit_mp_units)
    else()
      set(SAMPLE kalman_sample_${BACKEND}_${NAME})
      set(LIBRARY kalman_linalg_${BACKEND})
    endif()

    add_library(${SAMPLE}_bench_object OBJECT "${SOURCE}")
    target_link_libraries(
      ${SAMPLE}_bench_object
      PRIVATE kalman kalman_realtime ${LIBRARY} kalman_support_options)

    add_executable(${SAMPLE}_bench_driver "${SOURCE}")
    target_compile_options(
      ${SAMPLE}_bench_driver
      PRIVATE ${HEADER} ${IGNORELIST})
    target_link_libraries(
      ${SAMPLE}_bench_driver
      PRIVATE
        kalman kalman_main ${LIBRARY} kalman_support_options
        nanobench::nanobench)
    if(IPO_SUPPORTED)
      set_target_properties(
        ${SAMPLE}_bench_driver
        PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()

    add_test(
      NAME ${SAMPLE}_bench
      COMMAND
        ${CMAKE_COMMAND}
        "-DBINARY_DIR=${CMAKE_BINARY_DIR}"
        "-DCONFIG=$<CONFIG>"
        "-DOBJECT_TARGET=${SAMPLE}_bench_object"
        "-DOBJECT=$<TARGET_OBJECTS:${SAMPLE}_bench_object>"
        "-DREADELF=${CMAKE_READELF}"
        "-DDRIVER=$<TARGET_FILE:${SAMPLE}_bench_driver>"
        "-DBACKEND=${BACKEND}"
        "-DBACKENDS=${BACKENDS}"
        "-DMEASUREMENTS=${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
        "-DCOMPILER=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        "-DDOCUMENT=${DOCUMENT}"
        -P
        "${CMAKE_CURRENT_SOURCE_DIR}/bench.cmake")
    set_tests_properties(${SAMPLE}_bench PROPERTIES RUN_SERIAL TRUE)
  endforeach()
endfunction(bench)
