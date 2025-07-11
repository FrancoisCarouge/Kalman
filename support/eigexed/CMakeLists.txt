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

FetchContent_Declare(
  fcarouge-typed-linear-algebra
  GIT_REPOSITORY "https://github.com/FrancoisCarouge/TypedLinearAlgebra"
  GIT_TAG "952d6790384e62ee8fd0d6b88b3ed1d782abcc46"
  FIND_PACKAGE_ARGS NAMES fcarouge-typed-linear-algebra)
FetchContent_MakeAvailable(fcarouge-typed-linear-algebra)

add_library(kalman_linalg_eigexed INTERFACE)
target_sources(
  kalman_linalg_eigexed INTERFACE FILE_SET "linalg_headers" TYPE "HEADERS"
                                  FILES "fcarouge/linalg.hpp")
target_link_libraries(
  kalman_linalg_eigexed INTERFACE kalman kalman_linalg_eigen
                                  fcarouge-typed-linear-algebra::linalg)
