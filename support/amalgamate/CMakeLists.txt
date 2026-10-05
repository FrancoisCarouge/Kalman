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

# Single-header amalgamation.
#
# * NAME The generated header's file name, without extension.
# * ENTRY The header to start amalgamating from.
# * INCLUDE_DIRS Fallback roots to resolve quoted includes against, in order,
#   after the including file's own directory.
function(amalgamate NAME ENTRY)
  set(multiValueArgs INCLUDE_DIRS)
  cmake_parse_arguments(PARSE_ARGV 2 AMALGAMATION "" "" "${multiValueArgs}")

  set(OUTPUT_HEADER "${CMAKE_BINARY_DIR}/amalgamate/fcarouge/${NAME}.h")
  add_custom_command(
    OUTPUT "${OUTPUT_HEADER}"
    COMMAND
      "${CMAKE_COMMAND}"
      -D
      "INPUT=${ENTRY}"
      -D
      "INCLUDE_DIRS=${AMALGAMATION_INCLUDE_DIRS}"
      -D
      "OUTPUT=${OUTPUT_HEADER}"
      -P
      "${CMAKE_CURRENT_SOURCE_DIR}/amalgamate.cmake"
    DEPENDS "${ENTRY}"
    DEPFILE "${OUTPUT_HEADER}.d"
    COMMENT "Generating amalgamated ${NAME}.h header"
    VERBATIM)
  add_custom_target("amalgamate_${NAME}" DEPENDS "${OUTPUT_HEADER}")
endfunction(amalgamate)

amalgamate(kalman "${CMAKE_SOURCE_DIR}/include/fcarouge/kalman.hpp" INCLUDE_DIRS
           "${CMAKE_SOURCE_DIR}/include")

add_custom_target(amalgamate DEPENDS amalgamate_kalman)
