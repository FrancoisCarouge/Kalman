#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.2
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
  gsl-lite
  GIT_REPOSITORY "https://github.com/gsl-lite/gsl-lite"
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS NAMES gsl-lite)
FetchContent_MakeAvailable(gsl-lite)

FetchContent_Declare(
  mp-units
  GIT_REPOSITORY "https://github.com/mpusz/mp-units"
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR "src" FIND_PACKAGE_ARGS NAMES mp-units)
FetchContent_MakeAvailable(mp-units)

target_compile_options(mp-units INTERFACE $<IF:$<CXX_COMPILER_ID:MSVC>, ,
                                          -Wno-double-promotion -Wno-undef>)

add_library(kalman_unit_mp_units INTERFACE)
target_sources(
  kalman_unit_mp_units INTERFACE FILE_SET "unit_headers" TYPE "HEADERS" FILES
                                 "fcarouge/unit.hpp")
target_link_libraries(kalman_unit_mp_units INTERFACE kalman_support_options
                                                     mp-units::mp-units)
