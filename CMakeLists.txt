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

cmake_minimum_required(VERSION "3.28")

project(
  kalman
  DESCRIPTION "Kalman Filter"
  HOMEPAGE_URL "https://github.com/FrancoisCarouge/Kalman"
  LANGUAGES "CXX"
  VERSION "0.3.0")

set(CPACK_PACKAGE_CHECKSUM "SHA512")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.txt")
set(CPACK_SOURCE_GENERATOR "TBZ2")
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CXX_STANDARD 20)

# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") include("cmake/gcc_modules.cmake")
# endif() set(CMAKE_CXX_STANDARD 23) set(CXX_SCAN_FOR_MODULES TRUE)

# message(STATUS "CMAKE VERSION: " ${CMAKE_VERSION})

# if(${CMAKE_VERSION} VERSION_EQUAL "3.25")
#   set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "3c375311-a3c9-4396-a187-3227ef642046") 
# elseif(${CMAKE_VERSION} VERSION_EQUAL "3.26") 
#   set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "2182bf5c-ef0d-489a-91da-49dbc3090d2a") 
# elseif(${CMAKE_VERSION} VERSION_EQUAL "3.27") 
#   set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "aa1f7df0-828a-4fcd-9afc-2dc80491aca7") 
# endif()

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "aa1f7df0-828a-4fcd-9afc-2dc80491aca7")
# set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)

include(CMakePackageConfigHelpers)
include(CPack)
include(CTest)
include(GNUInstallDirs)

add_subdirectory("benchmark")
add_subdirectory("cmake")
add_subdirectory("include")
add_subdirectory("linalg")
add_subdirectory("pkgconfig")
add_subdirectory("sample")
add_subdirectory("source")
add_subdirectory("support")
add_subdirectory("test")
