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

# Measure a sample on one backend, then write the sample's benchmark
# documentation from the measurements of each of its backends.

# The compile time: the wall time to rebuild the sample's object.
file(REMOVE "${OBJECT}")
string(TIMESTAMP START "%s%f" UTC)
execute_process(
  COMMAND
    "${CMAKE_COMMAND}"
    --build
    "${BINARY_DIR}"
    --config
    "${CONFIG}"
    --target
    "${OBJECT_TARGET}"
    OUTPUT_QUIET
  COMMAND_ERROR_IS_FATAL ANY)
string(TIMESTAMP STOP "%s%f" UTC)
math(EXPR CENTISECONDS "(${STOP} - ${START}) / 10000")
math(EXPR SECONDS "${CENTISECONDS} / 100")
math(EXPR CENTISECONDS "${CENTISECONDS} % 100")
if(CENTISECONDS LESS 10)
  set(CENTISECONDS "0${CENTISECONDS}")
endif()

# The flash and RAM sizes of the object's allocated sections.
set(FLASH "-")
set(RAM "-")
if(READELF)
  execute_process(
    COMMAND "${READELF}" --section-headers --wide "${OBJECT}"
    OUTPUT_VARIABLE HEADERS
    COMMAND_ERROR_IS_FATAL ANY)
  set(FLASH 0)
  set(RAM 0)
  string(REPLACE "\n" ";" SECTIONS "${HEADERS}")
  foreach(SECTION IN LISTS SECTIONS)
    if(
      SECTION MATCHES
      "\\] +[^ ]+ +([A-Z0-9_]+) +[0-9a-f]+ [0-9a-f]+ ([0-9a-f]+) [0-9a-f]+ +([A-Z]*A[A-Z]*) "
    )
      set(TYPE "${CMAKE_MATCH_1}")
      math(EXPR SIZE "0x${CMAKE_MATCH_2}")
      if(CMAKE_MATCH_3 MATCHES "W")
        math(EXPR RAM "${RAM} + ${SIZE}")
      endif()
      if(NOT TYPE STREQUAL "NOBITS")
        math(EXPR FLASH "${FLASH} + ${SIZE}")
      endif()
    endif()
  endforeach()
endif()

# The run time of the filter's first prediction and update.
execute_process(
  COMMAND "${DRIVER}"
  OUTPUT_VARIABLE OUTPUT
  COMMAND_ERROR_IS_FATAL ANY)
foreach(METHOD IN ITEMS predict update)
  set(${METHOD}_TIME "-")
  set(${METHOD}_INSTRUCTIONS "-")
  if(OUTPUT MATCHES "(^|\n)${METHOD} ([0-9.]+) ([0-9.]+) ([0-9]+|-)\n")
    set(${METHOD}_TIME "${CMAKE_MATCH_2} ± ${CMAKE_MATCH_3} %")
    set(${METHOD}_INSTRUCTIONS "${CMAKE_MATCH_4}")
  endif()
endforeach()

file(
  WRITE "${MEASUREMENTS}/${BACKEND}.txt"
  "${predict_TIME};${update_TIME};${predict_INSTRUCTIONS};"
  "${update_INSTRUCTIONS};${SECONDS}.${CENTISECONDS};${FLASH};${RAM}")

# The documentation: a row per metric measured, a column per backend.
set(METRICS
    "`predict` time (ns)"
    "`update` time (ns)"
    "`predict` instructions"
    "`update` instructions"
    "Compile time (s)"
    "Flash (B)"
    "RAM (B)")
string(REPLACE "," ";" BACKENDS "${BACKENDS}")
set(HEADER "| Metric |")
set(ALIGNMENT "| :-- |")
foreach(COLUMN IN LISTS BACKENDS)
  if(COLUMN STREQUAL "none")
    string(APPEND HEADER " Value |")
  else()
    string(APPEND HEADER " ${COLUMN} |")
  endif()
  string(APPEND ALIGNMENT " --: |")
  set(VALUES_${COLUMN} "-;-;-;-;-;-;-")
  if(EXISTS "${MEASUREMENTS}/${COLUMN}.txt")
    file(READ "${MEASUREMENTS}/${COLUMN}.txt" VALUES_${COLUMN})
  endif()
endforeach()

set(TABLE "${HEADER}\n${ALIGNMENT}\n")
list(LENGTH METRICS COUNT)
math(EXPR LAST "${COUNT} - 1")
foreach(INDEX RANGE ${LAST})
  list(GET METRICS ${INDEX} ROW)
  set(ROW "| ${ROW} |")
  set(MEASURED FALSE)
  foreach(COLUMN IN LISTS BACKENDS)
    list(GET VALUES_${COLUMN} ${INDEX} VALUE)
    string(APPEND ROW " ${VALUE} |")
    if(NOT VALUE STREQUAL "-")
      set(MEASURED TRUE)
    endif()
  endforeach()
  if(MEASURED)
    string(APPEND TABLE "${ROW}\n")
  endif()
endforeach()

cmake_host_system_information(RESULT PROCESSOR QUERY PROCESSOR_DESCRIPTION)
string(STRIP "${PROCESSOR}" PROCESSOR)
if(CONFIG)
  set(CONFIG " `${CONFIG}`")
endif()

file(
  WRITE "${DOCUMENT}"
  "The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) "
  "of the sample, measured with ${COMPILER}${CONFIG} on ${PROCESSOR}:\n\n"
  "${TABLE}")
