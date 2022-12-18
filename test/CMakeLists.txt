#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
Version 0.2.0
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

foreach(TEST "f.cpp" "format.cpp" "h.cpp" "initialization.cpp"
             "kf_1x1x0_building_height.cpp")
  get_filename_component(NAME ${TEST} NAME_WE)
  add_executable(kalman_test_${NAME}_driver ${TEST})
  target_link_libraries(kalman_test_${NAME}_driver PRIVATE kalman kalman_main)
  add_test(kalman_test_${NAME} kalman_test_${NAME}_driver)
  add_test(
    NAME kalman_valgrind_${NAME}
    COMMAND valgrind --error-exitcode=1 --leak-check=full --track-origins=yes
            $<TARGET_FILE:kalman_test_${NAME}_driver>)
endforeach()

foreach(TEST_EIGEN "eigen_f.cpp" "eigen_h.cpp" "eigen_initialization.cpp")
  get_filename_component(NAME ${TEST_EIGEN} NAME_WE)
  add_executable(kalman_test_eigen_${NAME}_driver ${TEST_EIGEN})
  target_link_libraries(kalman_test_eigen_${NAME}_driver
                        PRIVATE Eigen3::Eigen kalman kalman_main)
  add_test(kalman_test_eigen_${NAME} kalman_test_eigen_${NAME}_driver)
  add_test(
    NAME kalman_valgrind_eigen_${NAME}
    COMMAND valgrind --error-exitcode=1 --leak-check=full --track-origins=yes
            $<TARGET_FILE:kalman_test_eigen_${NAME}_driver>)
endforeach()
