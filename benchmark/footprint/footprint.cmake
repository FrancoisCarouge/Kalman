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

# Measure the footprint of a step entry point image, write its documentation
# table row, and verify it against its expected sizes.
#
# The entry points are flattened: their configuration object holds no other
# function. Their only possible callees are the leaf memory routines, the image
# being linked without any library. The worst-case stack is the entry point
# frame and the largest frame of the leaf routines linked in the image.

execute_process(
  COMMAND "${READELF}" --section-headers --wide "${IMAGE}"
  OUTPUT_VARIABLE HEADERS
  COMMAND_ERROR_IS_FATAL ANY)

foreach(SECTION IN ITEMS text rodata data bss)
  set(${SECTION} 0)
  if(HEADERS MATCHES
     "\\] \\.${SECTION} +[A-Z]+ +[0-9a-f]+ [0-9a-f]+ ([0-9a-f]+)")
    math(EXPR ${SECTION} "0x${CMAKE_MATCH_1}")
  endif()
endforeach()

math(EXPR MEASURED_FLASH "${text} + ${rodata} + ${data}")
math(EXPR MEASURED_RAM "${data} + ${bss}")

string(REGEX REPLACE "\\.o(bj)?$" ".su" ENTRY_USAGE "${ENTRY_USAGE}")
string(REGEX REPLACE "\\.o(bj)?$" ".su" LEAF_USAGE "${LEAF_USAGE}")
file(STRINGS "${ENTRY_USAGE}" ENTRY_FRAMES)
file(STRINGS "${LEAF_USAGE}" LEAF_FRAMES)

foreach(FRAME IN LISTS ENTRY_FRAMES LEAF_FRAMES)
  if(NOT FRAME MATCHES "\t[0-9]+\tstatic$")
    message(FATAL_ERROR "Unbounded stack frame: ${FRAME}")
  endif()
endforeach()

set(ENTRY_FRAME "")
foreach(FRAME IN LISTS ENTRY_FRAMES)
  if(NOT FRAME MATCHES "_step\\(")
    message(FATAL_ERROR "Function not flattened in its entry point: ${FRAME}")
  endif()
  if(FRAME MATCHES " ${POLICY}_step\\(.*\t([0-9]+)\tstatic$")
    set(ENTRY_FRAME ${CMAKE_MATCH_1})
  endif()
endforeach()

if(ENTRY_FRAME STREQUAL "")
  message(FATAL_ERROR "No ${POLICY}_step entry point in: ${ENTRY_USAGE}")
endif()

execute_process(
  COMMAND "${NM}" --defined-only "${IMAGE}"
  OUTPUT_VARIABLE SYMBOLS
  COMMAND_ERROR_IS_FATAL ANY)

set(LEAF_FRAME 0)
foreach(FRAME IN LISTS LEAF_FRAMES)
  string(REGEX MATCH " ([a-z_]+)\\(.*\t([0-9]+)\tstatic$" _ "${FRAME}")
  set(SIZE ${CMAKE_MATCH_2})
  if(SYMBOLS MATCHES " ${CMAKE_MATCH_1}\n" AND SIZE GREATER LEAF_FRAME)
    set(LEAF_FRAME ${SIZE})
  endif()
endforeach()

math(EXPR MEASURED_STACK "${ENTRY_FRAME} + ${LEAF_FRAME}")

set(CELLS
    ${NAME}
    ${POLICY}
    -${OPTIMIZATION}
    ${text}
    ${rodata}
    ${data}
    ${bss}
    ${MEASURED_FLASH}
    ${MEASURED_RAM}
    ${MEASURED_STACK})
list(JOIN CELLS "</td><td>" CELLS)
file(WRITE "${ROW}" "<tr><td>${CELLS}</td></tr>\n")

set(FAILURES "")
foreach(METRIC IN ITEMS FLASH RAM STACK)
  if(NOT MEASURED_${METRIC} EQUAL ${METRIC})
    string(
      APPEND FAILURES
      "\n${NAME} ${POLICY} ${OPTIMIZATION} ${METRIC} ${MEASURED_${METRIC}}, expected ${${METRIC}}"
    )
  endif()
endforeach()

if(FAILURES)
  message(FATAL_ERROR "Footprint changed:${FAILURES}")
endif()
