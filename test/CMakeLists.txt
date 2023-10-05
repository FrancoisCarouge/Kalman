#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.3.0
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

include(FetchContent)

foreach(
  TEST
  "kalman_constructor_default_float_1x1x1.cpp"
  "kalman_constructor_default.cpp"
  "kalman_f.cpp"
  "kalman_format_arguments.cpp"
  "kalman_format_float_1x1x1.cpp"
  "kalman_format.cpp"
  "kalman_h.cpp")
  get_filename_component(NAME ${TEST} NAME_WE)
  add_executable(kalman_test_${NAME}_driver ${TEST})
  set_target_properties(
    kalman_test_${NAME}_driver
    PROPERTIES CXX_STANDARD 23
               CXX_EXTENSIONS OFF
               INTERPROCEDURAL_OPTIMIZATION TRUE)
  target_link_libraries(kalman_test_${NAME}_driver PRIVATE kalman kalman_main
                                                           kalman_options)
  add_test(kalman_test_${NAME} kalman_test_${NAME}_driver)
  add_test(
    NAME kalman_valgrind_${NAME}
    COMMAND valgrind --error-exitcode=1 --leak-check=full --track-origins=yes
            $<TARGET_FILE:kalman_test_${NAME}_driver>)
endforeach()

foreach(BACKEND IN ITEMS "eigen" "lazy" "naive")
  foreach(
    TEST
    "linalg_addition.cpp"
    "linalg_assign.cpp"
    "linalg_constructor_1x1_array.cpp"
    "linalg_constructor_1x1.cpp"
    "linalg_constructor_1xn_array.cpp"
    "linalg_constructor_1xn.cpp"
    "linalg_constructor_initializer_lists.cpp"
    "linalg_constructor_nx1_array.cpp"
    "linalg_constructor_nx1.cpp"
    "linalg_copy.cpp"
    "linalg_identity_default.cpp"
    "linalg_identity.cpp"
    "linalg_multiplication_arithmetic.cpp"
    "linalg_multiplication_rxc.cpp"
    "linalg_multiplication_sxc.cpp"
    "linalg_operator_bracket.cpp"
    "linalg_operator_equality.cpp"
    "linalg_zero_default.cpp"
    "linalg_zero.cpp")
    get_filename_component(NAME ${TEST} NAME_WE)
    add_executable(kalman_test_${BACKEND}_${NAME}_driver ${TEST})
    set_target_properties(
      kalman_test_${BACKEND}_${NAME}_driver
      PROPERTIES CXX_STANDARD 23
                 CXX_EXTENSIONS OFF
                 INTERPROCEDURAL_OPTIMIZATION TRUE)
    target_link_libraries(
      kalman_test_${BACKEND}_${NAME}_driver
      PRIVATE kalman kalman_main kalman_linalg_${BACKEND} kalman_options)
    add_test(kalman_test_${BACKEND}_${NAME}
             kalman_test_${BACKEND}_${NAME}_driver)
    add_test(
      NAME kalman_${BACKEND}_valgrind_${NAME}
      COMMAND valgrind --error-exitcode=1 --leak-check=full --track-origins=yes
              $<TARGET_FILE:kalman_test_${BACKEND}_${NAME}_driver>)
  endforeach()
endforeach()

foreach(BACKEND IN ITEMS "eigen")
  foreach(
    TEST
    "kalman_constructor_default_1x1x3.cpp"
    "kalman_constructor_default_1x4x1.cpp"
    "kalman_constructor_default_1x4x3.cpp"
    "kalman_constructor_default_5x1x1.cpp"
    "kalman_constructor_default_5x1x3.cpp"
    "kalman_constructor_default_5x4x0.cpp"
    "kalman_constructor_default_5x4x1.cpp"
    "kalman_constructor_default_5x4x3.cpp"
    "kalman_f_5x4x3.cpp"
    "kalman_h_5x4x3.cpp")
    get_filename_component(NAME ${TEST} NAME_WE)
    add_executable(kalman_test_${BACKEND}_${NAME}_driver ${TEST})
    set_target_properties(kalman_test_${BACKEND}_${NAME}_driver
                          PROPERTIES CXX_STANDARD 23 CXX_EXTENSIONS OFF)
    target_link_libraries(
      kalman_test_${BACKEND}_${NAME}_driver
      PRIVATE kalman kalman_main kalman_linalg_${BACKEND} kalman_options)
    add_test(kalman_test_${BACKEND}_${NAME}
             kalman_test_${BACKEND}_${NAME}_driver)
    add_test(
      NAME kalman_${BACKEND}_valgrind_${NAME}
      COMMAND valgrind --error-exitcode=1 --leak-check=full --track-origins=yes
              $<TARGET_FILE:kalman_test_${BACKEND}_${NAME}_driver>)
  endforeach()
endforeach()

foreach(BACKEND IN ITEMS "lazy")
  foreach(TEST "linalg_constructor_initializer_lists_deduction.cpp")
    get_filename_component(NAME ${TEST} NAME_WE)
    add_executable(kalman_test_${BACKEND}_${NAME}_driver ${TEST})
    set_target_properties(kalman_test_${BACKEND}_${NAME}_driver
                          PROPERTIES CXX_STANDARD 23 CXX_EXTENSIONS OFF)
    target_link_libraries(
      kalman_test_${BACKEND}_${NAME}_driver
      PRIVATE kalman kalman_main kalman_linalg_${BACKEND} kalman_options)
    add_test(kalman_test_${BACKEND}_${NAME}
             kalman_test_${BACKEND}_${NAME}_driver)
    add_test(
      NAME kalman_${BACKEND}_valgrind_${NAME}
      COMMAND valgrind --error-exitcode=1 --leak-check=full --track-origins=yes
              $<TARGET_FILE:kalman_test_${BACKEND}_${NAME}_driver>)
  endforeach()
endforeach()
