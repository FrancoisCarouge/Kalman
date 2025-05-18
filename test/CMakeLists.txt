#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.1
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

if(NOT BUILD_TESTING)
  return()
endif()

test("kalman_assign_move_5x4x3" BACKENDS "eigen" "eigexed")
test("kalman_constructor_default_1x1x3" BACKENDS "eigen" "eigexed")
test("kalman_constructor_default_1x4x1" BACKENDS "eigen" "eigexed")
test("kalman_constructor_default_1x4x3" BACKENDS "eigen" "eigexed")
test("kalman_constructor_default_5x1x1" BACKENDS "eigen" "eigexed")
test("kalman_constructor_default_5x1x3" BACKENDS "eigen" "eigexed")
test("kalman_constructor_default_5x4x0" BACKENDS "eigen" "eigexed")
test("kalman_constructor_default_5x4x1" BACKENDS "eigen" "eigexed")
test("kalman_constructor_default_5x4x3" BACKENDS "eigen" "eigexed")
test("kalman_constructor_default_float_1x1x1")
test("kalman_constructor_default")
test("kalman_constructor_move_5x4x3" BACKENDS "eigen" "eigexed")
test("kalman_f_5x4x3" BACKENDS "eigen" "eigexed")
test("kalman_f")
test("kalman_format_1x4x3" BACKENDS "eigen" "eigexed")
test("kalman_format_5x4x3" BACKENDS "eigen" "eigexed")
test("kalman_format_arguments")
test("kalman_format_float_1x1x1")
test("kalman_format")
test("kalman_h_5x4x3" BACKENDS "eigen" "eigexed")
test("kalman_println_1x1x0")
test("linalg_addition" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_assign" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_constructor_1x1_array" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_constructor_1x1" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_constructor_1xn_array" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_constructor_1xn" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_constructor_initializer_lists" BACKENDS "eigen" "eigexed" "naive"
     "naixed")
test("linalg_constructor_nx1_array" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_constructor_nx1" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_copy" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_format_1xn" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_format_mx1" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_format_mxn" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_identity" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_multiplication_arithmetic" BACKENDS "eigen" "eigexed" "naive"
     "naixed")
test("linalg_multiplication_rxc" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_multiplication_sxc" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_operator_bracket" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_operator_equality" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("linalg_zero" BACKENDS "eigen" "eigexed" "naive" "naixed")
test("printer_1x1x0")
test("printer_2x3x4" BACKENDS "eigen" "eigexed")
test("utility_identity_default")
test("utility_zero_default")
