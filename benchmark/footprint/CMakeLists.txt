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

if(NOT BUILD_TESTING)
  return()
endif()

# Add a given embedded footprint measurement.
#
# Cross-compiled, the test passes when the policy's step entry point, linked
# alone, measures the expected sizes in bytes. Otherwise, the configuration is
# only compiled. The test is named after the directory it is declared in, its
# configuration, policy, and optimization.
#
# * NAME The name of the configuration file without extension.
# * POLICY The update form policy of the entry point.
# * OPTIMIZATION The optimization level: `Os` or `O2`.
# * FLASH The expected read-only memory size: `.text`, `.rodata`, `.data`.
# * RAM The expected read-write memory size: `.data`, `.bss`.
# * STACK The expected worst-case stack size.
function(footprint NAME POLICY OPTIMIZATION)
  cmake_parse_arguments(PARSE_ARGV 3 EXPECTED "" "FLASH;RAM;STACK" "")

  if(NOT CMAKE_CROSSCOMPILING)
    if(NOT TARGET kalman_freestanding_footprint_${NAME}_build_driver)
      build("${NAME}" BACKENDS "freestanding")
    endif()
    return()
  endif()

  set(MEMORY kalman_footprint_memory_${OPTIMIZATION})
  if(NOT TARGET ${MEMORY})
    add_library(${MEMORY} OBJECT "memory.cpp")
    target_compile_options(
      ${MEMORY}
      PRIVATE "-${OPTIMIZATION}" "-fno-tree-loop-distribute-patterns")
    target_link_libraries(${MEMORY} PRIVATE kalman_footprint_options)
  endif()

  set(CONFIGURATION kalman_footprint_${NAME}_${OPTIMIZATION})
  if(NOT TARGET ${CONFIGURATION})
    add_library(${CONFIGURATION} OBJECT "${NAME}.cpp")
    target_compile_options(${CONFIGURATION} PRIVATE "-${OPTIMIZATION}")
    target_link_libraries(
      ${CONFIGURATION}
      PRIVATE kalman kalman_linalg_freestanding kalman_footprint_options)
  endif()

  string(TOLOWER "${NAME}_${POLICY}_${OPTIMIZATION}" MEASUREMENT)
  set(IMAGE kalman_footprint_${MEASUREMENT})
  add_executable(${IMAGE} $<TARGET_OBJECTS:${CONFIGURATION}>
                 $<TARGET_OBJECTS:${MEMORY}>)
  set_target_properties(${IMAGE} PROPERTIES LINKER_LANGUAGE "CXX")
  target_link_options(
    ${IMAGE}
    PRIVATE
      "-nostdlib" "LINKER:--gc-sections" "LINKER:--entry=${POLICY}_step"
      "LINKER:--defsym=memcpy=footprint_copy"
      "LINKER:--defsym=memset=footprint_fill")
  add_test(
    NAME kalman_freestanding_footprint_${MEASUREMENT}_pass
    COMMAND
      ${CMAKE_COMMAND}
      "-DNM=${CMAKE_NM}"
      "-DREADELF=${CMAKE_READELF}"
      "-DIMAGE=$<TARGET_FILE:${IMAGE}>"
      "-DENTRY_USAGE=$<TARGET_OBJECTS:${CONFIGURATION}>"
      "-DLEAF_USAGE=$<TARGET_OBJECTS:${MEMORY}>"
      "-DNAME=${NAME}"
      "-DPOLICY=${POLICY}"
      "-DOPTIMIZATION=${OPTIMIZATION}"
      "-DFLASH=${EXPECTED_FLASH}"
      "-DRAM=${EXPECTED_RAM}"
      "-DSTACK=${EXPECTED_STACK}"
      "-DROW=${CMAKE_CURRENT_BINARY_DIR}/row/${MEASUREMENT}.md"
      -P
      "${CMAKE_CURRENT_SOURCE_DIR}/footprint.cmake")
endfunction(footprint)

if(CMAKE_CROSSCOMPILING)
  add_library(kalman_footprint_options INTERFACE)
  target_compile_options(
    kalman_footprint_options
    INTERFACE "-ffunction-sections" "-fdata-sections" "-fstack-usage")
  configure_file("footprint.md.in" "${CMAKE_BINARY_DIR}/footprint.md" @ONLY)
else()
  add_library(kalman_footprint_memory OBJECT "memory.cpp")
  set_target_properties(
    kalman_footprint_memory
    PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

# Changing a criterion requires its reason in the commit message.
footprint("ekf_1x2x0" "information_form" "O2" FLASH 1016 RAM 84 STACK 64)
footprint("ekf_1x2x0" "information_form" "Os" FLASH 1060 RAM 84 STACK 56)
footprint("ekf_1x2x0" "joseph_form" "O2" FLASH 760 RAM 84 STACK 64)
footprint("ekf_1x2x0" "joseph_form" "Os" FLASH 764 RAM 84 STACK 56)
footprint("ekf_1x2x0" "simple_form" "O2" FLASH 692 RAM 84 STACK 56)
footprint("ekf_1x2x0" "simple_form" "Os" FLASH 712 RAM 84 STACK 56)
footprint("ekf_3x1x0" "information_form" "O2" FLASH 2236 RAM 176 STACK 288)
footprint("ekf_3x1x0" "information_form" "Os" FLASH 2216 RAM 176 STACK 320)
footprint("ekf_3x1x0" "joseph_form" "O2" FLASH 1872 RAM 176 STACK 384)
footprint("ekf_3x1x0" "joseph_form" "Os" FLASH 1900 RAM 176 STACK 400)
footprint("ekf_3x1x0" "simple_form" "O2" FLASH 1364 RAM 176 STACK 200)
footprint("ekf_3x1x0" "simple_form" "Os" FLASH 1376 RAM 176 STACK 208)
footprint("ekf_4x1x0" "information_form" "O2" FLASH 2792 RAM 276 STACK 544)
footprint("ekf_4x1x0" "information_form" "Os" FLASH 2784 RAM 276 STACK 552)
footprint("ekf_4x1x0" "joseph_form" "O2" FLASH 2488 RAM 276 STACK 696)
footprint("ekf_4x1x0" "joseph_form" "Os" FLASH 2496 RAM 276 STACK 696)
footprint("ekf_4x1x0" "simple_form" "O2" FLASH 1700 RAM 276 STACK 312)
footprint("ekf_4x1x0" "simple_form" "Os" FLASH 1700 RAM 276 STACK 312)
footprint("ekf_9x4x0" "information_form" "O2" FLASH 9552 RAM 1464 STACK 2560)
footprint("ekf_9x4x0" "information_form" "Os" FLASH 9242 RAM 1464 STACK 2560)
footprint("ekf_9x4x0" "joseph_form" "O2" FLASH 10292 RAM 1464 STACK 3208)
footprint("ekf_9x4x0" "joseph_form" "Os" FLASH 10594 RAM 1464 STACK 3208)
footprint("ekf_9x4x0" "simple_form" "O2" FLASH 6868 RAM 1464 STACK 1504)
footprint("ekf_9x4x0" "simple_form" "Os" FLASH 6994 RAM 1464 STACK 1512)
footprint("kf_1x1x0" "information_form" "O2" FLASH 120 RAM 32 STACK 0)
footprint("kf_1x1x0" "information_form" "Os" FLASH 120 RAM 32 STACK 0)
footprint("kf_1x1x0" "joseph_form" "O2" FLASH 124 RAM 32 STACK 0)
footprint("kf_1x1x0" "joseph_form" "Os" FLASH 124 RAM 32 STACK 0)
footprint("kf_1x1x0" "simple_form" "O2" FLASH 112 RAM 32 STACK 0)
footprint("kf_1x1x0" "simple_form" "Os" FLASH 112 RAM 32 STACK 0)
footprint("kf_1x1x1" "information_form" "O2" FLASH 132 RAM 36 STACK 0)
footprint("kf_1x1x1" "information_form" "Os" FLASH 132 RAM 36 STACK 0)
footprint("kf_1x1x1" "joseph_form" "O2" FLASH 136 RAM 36 STACK 0)
footprint("kf_1x1x1" "joseph_form" "Os" FLASH 136 RAM 36 STACK 0)
footprint("kf_1x1x1" "simple_form" "O2" FLASH 124 RAM 36 STACK 0)
footprint("kf_1x1x1" "simple_form" "Os" FLASH 124 RAM 36 STACK 0)
footprint("kf_2x1x0" "information_form" "O2" FLASH 1324 RAM 88 STACK 136)
footprint("kf_2x1x0" "information_form" "Os" FLASH 1388 RAM 88 STACK 112)
footprint("kf_2x1x0" "joseph_form" "O2" FLASH 756 RAM 88 STACK 8)
footprint("kf_2x1x0" "joseph_form" "Os" FLASH 724 RAM 88 STACK 0)
footprint("kf_2x1x0" "simple_form" "O2" FLASH 684 RAM 88 STACK 48)
footprint("kf_2x1x0" "simple_form" "Os" FLASH 632 RAM 88 STACK 48)
footprint("kf_2x1x1" "information_form" "O2" FLASH 1360 RAM 100 STACK 136)
footprint("kf_2x1x1" "information_form" "Os" FLASH 1420 RAM 100 STACK 112)
footprint("kf_2x1x1" "joseph_form" "O2" FLASH 784 RAM 100 STACK 8)
footprint("kf_2x1x1" "joseph_form" "Os" FLASH 748 RAM 100 STACK 0)
footprint("kf_2x1x1" "simple_form" "O2" FLASH 696 RAM 100 STACK 48)
footprint("kf_2x1x1" "simple_form" "Os" FLASH 660 RAM 100 STACK 48)
footprint("kf_6x2x0" "information_form" "O2" FLASH 5220 RAM 600 STACK 1112)
footprint("kf_6x2x0" "information_form" "Os" FLASH 5390 RAM 600 STACK 1120)
footprint("kf_6x2x0" "joseph_form" "O2" FLASH 5392 RAM 600 STACK 1408)
footprint("kf_6x2x0" "joseph_form" "Os" FLASH 5472 RAM 600 STACK 1408)
footprint("kf_6x2x0" "simple_form" "O2" FLASH 3472 RAM 600 STACK 648)
footprint("kf_6x2x0" "simple_form" "Os" FLASH 3606 RAM 600 STACK 656)
footprint("kf_8x4x0" "information_form" "O2" FLASH 8416 RAM 1216 STACK 2064)
footprint("kf_8x4x0" "information_form" "Os" FLASH 7874 RAM 1216 STACK 2072)
footprint("kf_8x4x0" "joseph_form" "O2" FLASH 8992 RAM 1216 STACK 2576)
footprint("kf_8x4x0" "joseph_form" "Os" FLASH 8702 RAM 1216 STACK 2584)
footprint("kf_8x4x0" "simple_form" "O2" FLASH 5904 RAM 1216 STACK 1216)
footprint("kf_8x4x0" "simple_form" "Os" FLASH 5834 RAM 1216 STACK 1224)
