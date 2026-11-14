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

# Verify the installed package is usable on its own: install the project into
# a fresh staging prefix, then configure and build a consumer project against
# it through the documented `find_package` interface. The installed public
# header includes every implementation file, so a header missing from the
# installed file set fails the consumer build.
set(PREFIX "${CMAKE_CURRENT_BINARY_DIR}/prefix")
set(CONSUMER "${CMAKE_CURRENT_BINARY_DIR}/consumer")

add_test(
  NAME kalman_install_clean
  COMMAND
    ${CMAKE_COMMAND}
    -E
    rm
    -r
    -f
    "${PREFIX}"
    "${CONSUMER}")
set_tests_properties(
  kalman_install_clean
  PROPERTIES FIXTURES_SETUP kalman_install_clean)

add_test(
  NAME kalman_install_install
  COMMAND
    ${CMAKE_COMMAND}
    --install
    "${CMAKE_BINARY_DIR}"
    --prefix
    "${PREFIX}"
    --config
    $<CONFIG>)
set_tests_properties(
  kalman_install_install
  PROPERTIES
    FIXTURES_REQUIRED kalman_install_clean FIXTURES_SETUP
    kalman_install_installed)

add_test(
  NAME kalman_install_find_package_pass
  COMMAND
    ${CMAKE_CTEST_COMMAND}
    --build-and-test
    "${CMAKE_CURRENT_SOURCE_DIR}/consumer"
    "${CONSUMER}"
    --build-generator
    "${CMAKE_GENERATOR}"
    --build-generator-platform
    "${CMAKE_GENERATOR_PLATFORM}"
    --build-generator-toolset
    "${CMAKE_GENERATOR_TOOLSET}"
    --build-makeprogram
    "${CMAKE_MAKE_PROGRAM}"
    --build-config
    $<CONFIG>
    --build-options
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_PREFIX_PATH=${PREFIX}")
set_tests_properties(
  kalman_install_find_package_pass
  PROPERTIES FIXTURES_REQUIRED kalman_install_installed)
