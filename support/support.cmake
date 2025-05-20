#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
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

# Add a given sample.
#
# * NAME The name of the sample file without extension.
# * BACKENDS Optional list of backends to use against the sample.
function(sample SAMPLE_NAME)
  set(multiValueArgs BACKENDS)
  cmake_parse_arguments(PARSE_ARGV 0 SAMPLE "" "${oneValueArgs}"
                        "${multiValueArgs}")

  if(NOT SAMPLE_BACKENDS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      message(STATUS "${SAMPLE_NAME} not yet compatible with MSVC/mp-units.")
      return()
    endif()

    add_executable(kalman_sample_${SAMPLE_NAME}_driver "${SAMPLE_NAME}.cpp")
    target_link_libraries(
      kalman_sample_${SAMPLE_NAME}_driver
      PRIVATE kalman kalman_main kalman_support_options kalman_unit_mp_units)
    separate_arguments(SAMPLE_COMMAND UNIX_COMMAND $ENV{COMMAND})
    add_test(NAME kalman_sample_${SAMPLE_NAME}
             COMMAND ${SAMPLE_COMMAND}
                     $<TARGET_FILE:kalman_sample_${SAMPLE_NAME}_driver>)
  else()
    foreach(BACKEND IN ITEMS ${SAMPLE_BACKENDS})
      if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND (BACKEND STREQUAL
                                                      "quantity"))
        message(STATUS "${SAMPLE_NAME} not yet compatible with MSVC/mp-units.")
        continue()
      endif()

      add_executable(kalman_sample_${BACKEND}_${SAMPLE_NAME}_driver
                     "${SAMPLE_NAME}.cpp")
      target_link_libraries(
        kalman_sample_${BACKEND}_${SAMPLE_NAME}_driver
        PRIVATE kalman kalman_main kalman_linalg_${BACKEND}
                kalman_support_options)
      separate_arguments(SAMPLE_COMMAND UNIX_COMMAND $ENV{COMMAND})
      add_test(
        NAME kalman_sample_${BACKEND}_${SAMPLE_NAME}
        COMMAND ${SAMPLE_COMMAND}
                $<TARGET_FILE:kalman_sample_${BACKEND}_${SAMPLE_NAME}_driver>)
    endforeach()
  endif()
endfunction(sample)

# Add a given test.
#
# * NAME The name of the test file without extension.
# * BACKENDS Optional list of backends to use against the test.
function(test TEST_NAME)
  set(multiValueArgs BACKENDS)
  cmake_parse_arguments(PARSE_ARGV 0 TEST "" "${oneValueArgs}"
                        "${multiValueArgs}")

  if(NOT TEST_BACKENDS)
    add_executable(kalman_test_${TEST_NAME}_driver "${TEST_NAME}.cpp")
    target_link_libraries(
      kalman_test_${TEST_NAME}_driver
      PRIVATE kalman kalman_main kalman_support_options kalman_unit_mp_units)
    separate_arguments(TEST_COMMAND UNIX_COMMAND $ENV{COMMAND})
    add_test(NAME kalman_test_${TEST_NAME}
             COMMAND ${TEST_COMMAND}
                     $<TARGET_FILE:kalman_test_${TEST_NAME}_driver>)
  else()
    foreach(BACKEND IN ITEMS ${TEST_BACKENDS})
      if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND (BACKEND STREQUAL
                                                      "quantity"))
        message(STATUS "${TEST_NAME} not yet compatible with MSVC/mp-units.")
        continue()
      endif()

      add_executable(kalman_test_${BACKEND}_${TEST_NAME}_driver
                     "${TEST_NAME}.cpp")
      target_link_libraries(
        kalman_test_${BACKEND}_${TEST_NAME}_driver
        PRIVATE kalman kalman_main kalman_linalg_${BACKEND}
                kalman_support_options)
      separate_arguments(TEST_COMMAND UNIX_COMMAND $ENV{COMMAND})
      add_test(
        NAME kalman_test_${BACKEND}_${TEST_NAME}
        COMMAND ${TEST_COMMAND}
                $<TARGET_FILE:kalman_test_${BACKEND}_${TEST_NAME}_driver>)
    endforeach()
  endif()
endfunction(test)
