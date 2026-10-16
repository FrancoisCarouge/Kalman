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

build("arguments_1x1x0")
build("constant_1x1x0")
build("rows")
build("traits_1x1x0")
build("traits_5x4x3" BACKENDS "eigen" "eigen_typed")
fail("rows_fail")
pass("characteristics_1x1x1")
pass("characteristics_2x1x0" BACKENDS "eigen" "eigen_typed")
pass("copy_5x4x3" BACKENDS "eigen" "eigen_typed")
pass("copy_callable_1x1x0")
pass("default_1x1x0")
pass("default_1x1x3" BACKENDS "eigen" "eigen_typed")
pass("default_1x4x1" BACKENDS "eigen" "eigen_typed")
pass("default_1x4x3" BACKENDS "eigen" "eigen_typed")
pass("default_5x1x1" BACKENDS "eigen" "eigen_typed")
pass("default_5x1x3" BACKENDS "eigen" "eigen_typed")
pass("default_5x4x0" BACKENDS "eigen" "eigen_typed")
pass("default_5x4x1" BACKENDS "eigen" "eigen_typed")
pass("default_5x4x3" BACKENDS "eigen" "eigen_typed")
pass("default_float_1x1x1")
pass("move_5x4x3" BACKENDS "eigen" "eigen_typed")
pass("output_type_2x1x0" BACKENDS "eigen" "eigen_typed")
pass("state_1x1x0")
pass("static_1x1x0")
pass("stored_2x1x0" BACKENDS "eigen" "eigen_typed")
pass("stored_2x2x1" BACKENDS "eigen" "eigen_typed")
pass("uncertainties_1x1x0")

# The filter of the static storage test is constructed in a translation unit
# linked after the one defining its owner, dynamically initialized first.
target_sources(
  kalman_constructor_static_1x1x0_pass_driver
  PRIVATE "static_1x1x0_holder.cpp")
