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

if(NOT BUILD_TESTING)
  return()
endif()

add_subdirectory("eigen")
add_subdirectory("eigexed")
add_subdirectory("main")
add_subdirectory("mp_units")
add_subdirectory("naive")
add_subdirectory("naixed")
add_subdirectory("quantity")

add_library(kalman_support_options INTERFACE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(
    kalman_support_options
    INTERFACE "-D_FORTIFY_SOURCE=3"
              "-D_GLIBCXX_ASSERTIONS"
              "-fcf-protection=full"
              "-fno-check-new"
              "-fno-common"
              "-fPIE"
              "-fstack-protector-strong"
              "-fstrict-aliasing"
              "-ftrivial-auto-var-init=pattern"
              "-pedantic"
              "-UNDEBUG"
              "-Wall"
              "-Wcast-align"
              "-Wdouble-promotion"
              "-Wenum-conversion"
              "-Werror"
              "-Wextra"
              "-Wformat-security"
              "-Wno-long-long"
              "-Wno-psabi"
              "-Wno-variadic-macros"
              "-Wnon-virtual-dtor"
              "-Wpointer-arith"
              "-Wshadow"
              "-Wundef"
              "-Wunused-local-typedefs"
              "-Wwrite-strings")
  target_link_options(kalman_support_options INTERFACE "LINKER:-pie")
  target_link_libraries(kalman_support_options INTERFACE "stdc++")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    kalman_support_options
    INTERFACE "-D_FORTIFY_SOURCE=3"
              "-D_GLIBCXX_ASSERTIONS"
              "-fcf-protection=full"
              "-fno-check-new"
              "-fno-common"
              "-fPIE"
              "-fstack-clash-protection"
              "-fstack-protector-strong"
              "-fstrict-aliasing"
              "-ftrivial-auto-var-init=pattern"
              "-pedantic"
              "-UNDEBUG"
              "-Wall"
              "-Wcast-align"
              "-Wdouble-promotion"
              "-Wenum-conversion"
              "-Werror"
              "-Wextra"
              "-Wformat-security"
              "-Wno-long-long"
              "-Wno-psabi"
              "-Wno-variadic-macros"
              "-Wnon-virtual-dtor"
              "-Wpointer-arith"
              "-Wshadow"
              "-Wundef"
              "-Wunused-local-typedefs"
              "-Wwrite-strings")
  target_link_options(kalman_support_options INTERFACE "LINKER:-pie"
                      "LINKER:-z,relro,-z,now")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(kalman_support_options INTERFACE "/EHsc" "/utf-8"
                                                          "/UNDEBUG" "/W4")
endif()
