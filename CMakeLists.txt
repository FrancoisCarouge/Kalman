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

cmake_minimum_required(VERSION "3.24")

project(
  kalman
  DESCRIPTION "Kalman Filter for C++"
  HOMEPAGE_URL "https://github.com/FrancoisCarouge/Kalman"
  LANGUAGES "CXX"
  VERSION "0.2.0")

set(CPACK_PACKAGE_CHECKSUM "SHA512")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.txt")

include(CMakePackageConfigHelpers)
include(CPack)
include(CTest)
include(FetchContent)
include(GNUInstallDirs)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY "https://gitlab.com/libeigen/eigen"
  GIT_TAG "23524ab6fcbf557b4ae1885d25d867727330c0c2"
  FIND_PACKAGE_ARGS NAMES Eigen3)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY "https://github.com/fmtlib/fmt"
  FIND_PACKAGE_ARGS NAMES fmt)
option(FMT_INSTALL "Enable installation for the {fmt} project." ON)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
  google_benchmark
  GIT_REPOSITORY "https://github.com/google/benchmark"
  GIT_TAG "main"
  FIND_PACKAGE_ARGS NAMES benchmark)

FetchContent_Declare(
  google_test
  GIT_REPOSITORY "https://github.com/google/googletest"
  GIT_TAG "main"
  FIND_PACKAGE_ARGS NAMES GTest)
FetchContent_MakeAvailable(google_test google_benchmark)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY "https://github.com/kokkos/mdspan"
  GIT_TAG "stable"
  FIND_PACKAGE_ARGS NAMES mdspan)
FetchContent_MakeAvailable(mdspan)

add_subdirectory(benchmark)
add_subdirectory(cmake)
add_subdirectory(include)
add_subdirectory(pkgconfig)
add_subdirectory(sample)
add_subdirectory(source)
add_subdirectory(support)
add_subdirectory(test)
