#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
Version 0.1.0
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
  VERSION "0.1.0"
  DESCRIPTION "Kalman Filter for C++"
  LANGUAGES "CXX")

set(CMAKE_CXX_STANDARD "23" CMAKE_CXX_STANDARD_REQUIRED ON CMAKE_CXX_EXTENSIONS
                       OFF)

include(CTest)
include(FetchContent)

FetchContent_Declare(
  google_test
  URL "https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz"
  URL_HASH MD5=e82199374acdfda3f425331028eb4e2a)

FetchContent_Declare(
  google_benchmark
  URL "https://github.com/google/benchmark/archive/refs/tags/v1.6.2.tar.gz"
  URL_HASH MD5=14d14849e075af116143a161bc3b927b)

FetchContent_Declare(
  eigen
  URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2"
  URL_HASH MD5=132dde48fe2b563211675626d29f1707)

FetchContent_Declare(
  fmt
  URL "https://github.com/fmtlib/fmt/archive/refs/tags/9.0.0.tar.gz"
  URL_HASH MD5=d56c8b0612b049bb1854f07c8b133f3c)

FetchContent_MakeAvailable(google_test google_benchmark eigen fmt)

add_subdirectory(benchmark)
add_subdirectory(include)
add_subdirectory(sample)
add_subdirectory(source)
add_subdirectory(support)
add_subdirectory(test)
