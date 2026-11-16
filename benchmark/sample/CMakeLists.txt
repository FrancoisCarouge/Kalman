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

if(NOT BUILD_TESTING OR CMAKE_CROSSCOMPILING)
  return()
endif()

FetchContent_Declare(
  nanobench
  GIT_REPOSITORY "https://github.com/martinus/nanobench.git"
  GIT_SHALLOW TRUE
  SYSTEM
  FIND_PACKAGE_ARGS NAMES nanobench)
FetchContent_MakeAvailable(nanobench)

target_link_libraries(nanobench PRIVATE kalman_support_options)

# The benchmark header, included ahead of the sample source.
set(HEADER "${CMAKE_CURRENT_SOURCE_DIR}/fcarouge/benchmark.hpp")
if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(HEADER "/FI${HEADER}")
else()
  set(HEADER "SHELL:-include ${HEADER}")
endif()

# The standard library ignored by the integer sanitizer, when enabled.
set(IGNORELIST "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND CMAKE_CXX_FLAGS MATCHES "-fsanitize=[^ ]*integer")
  set(IGNORELIST
      "-fsanitize-ignorelist=${CMAKE_CURRENT_SOURCE_DIR}/sanitizer_ignorelist.txt"
  )
  target_compile_options(nanobench PRIVATE ${IGNORELIST})
endif()

# Add a given sample benchmark.
#
# The sample is compiled unchanged into an object, measured for its compile
# time and sizes, and into a driver, with the benchmark header included ahead,
# measuring its filter's first prediction and update. The test writes the
# sample's `<name>_benchmark.md` documentation: in its sample directory with
# `BUILD_BENCHMARKING`, in the build tree otherwise. The test is named after the
# sample: `kalman_sample_[<backend>_]<name>_bench`.
#
# * NAME The name of the sample.
# * BACKENDS Optional list of backends to use against the sample.
function(bench NAME)
  cmake_parse_arguments(PARSE_ARGV 1 BENCH "" "" "BACKENDS")

  if(NOT BENCH_BACKENDS)
    set(BENCH_BACKENDS "none")
  endif()

  set(SOURCE "${PROJECT_SOURCE_DIR}/sample/${NAME}/${NAME}.cpp")
  set(DOCUMENT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}_benchmark.md")
  if(BUILD_BENCHMARKING)
    set(DOCUMENT "${PROJECT_SOURCE_DIR}/sample/${NAME}/${NAME}_benchmark.md")
  endif()
  list(JOIN BENCH_BACKENDS "," BACKENDS)

  foreach(BACKEND IN LISTS BENCH_BACKENDS)
    if(BACKEND STREQUAL "none")
      set(SAMPLE kalman_sample_${NAME})
      set(LIBRARY kalman_unit_mp_units)
    else()
      set(SAMPLE kalman_sample_${BACKEND}_${NAME})
      set(LIBRARY kalman_linalg_${BACKEND})
    endif()

    add_library(${SAMPLE}_bench_object OBJECT "${SOURCE}")
    target_link_libraries(
      ${SAMPLE}_bench_object
      PRIVATE kalman kalman_realtime ${LIBRARY} kalman_support_options)

    add_executable(${SAMPLE}_bench_driver "${SOURCE}")
    target_compile_options(
      ${SAMPLE}_bench_driver
      PRIVATE ${HEADER} ${IGNORELIST})
    target_link_libraries(
      ${SAMPLE}_bench_driver
      PRIVATE
        kalman kalman_main ${LIBRARY} kalman_support_options
        nanobench::nanobench)
    if(IPO_SUPPORTED)
      set_target_properties(
        ${SAMPLE}_bench_driver
        PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()

    add_test(
      NAME ${SAMPLE}_bench
      COMMAND
        ${CMAKE_COMMAND}
        "-DBINARY_DIR=${CMAKE_BINARY_DIR}"
        "-DCONFIG=$<CONFIG>"
        "-DOBJECT_TARGET=${SAMPLE}_bench_object"
        "-DOBJECT=$<TARGET_OBJECTS:${SAMPLE}_bench_object>"
        "-DREADELF=${CMAKE_READELF}"
        "-DDRIVER=$<TARGET_FILE:${SAMPLE}_bench_driver>"
        "-DBACKEND=${BACKEND}"
        "-DBACKENDS=${BACKENDS}"
        "-DMEASUREMENTS=${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
        "-DCOMPILER=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        "-DDOCUMENT=${DOCUMENT}"
        -P
        "${CMAKE_CURRENT_SOURCE_DIR}/bench.cmake")
    set_tests_properties(${SAMPLE}_bench PROPERTIES RUN_SERIAL TRUE)
  endforeach()
endfunction(bench)

if(BUILD_BENCHMARKING)
  bench("ekf_1x2x0_terrain_estimation_unit" BACKENDS "mp_units_eigen")
  bench("ekf_3x1x0_beacon_position_unit" BACKENDS "mp_units_eigen")
  bench("ekf_3x2x0_wind_airspeed_scale_unit" BACKENDS "mp_units_eigen")
  bench("ekf_4x1x0_thermal_soaring" BACKENDS "eigen" "eigexed")
  bench("ekf_4x2x0_range_bearing_tracking_unit" BACKENDS "mp_units_eigen")
  bench("ekf_6x4x0_3d_apollo_position" BACKENDS "eigen")
  bench("ekf_9x4x0_gimbal_attitude_unit" BACKENDS "mp_units_eigen")
  bench("kf_1x1x0_building_height_unit")
endif()

bench("kf_1x1x0_building_height")

if(BUILD_BENCHMARKING)
  bench("kf_1x1x0_liquid_temperature_unit")
  bench("kf_1x1x0_liquid_temperature")
  bench("kf_1x1x1_1d_dog_position_unit")
  bench("kf_1x1x1_1d_dog_position")
  bench("kf_2x1x0_pairs_hedge_ratio_unit" BACKENDS "mp_units_eigen")
  bench("kf_2x1x0_stock_price_unit" BACKENDS "mp_units_eigen")
  bench("kf_2x1x1_1d_rocket_altitude" BACKENDS "eigen" "eigexed")
  bench("kf_2x1x1_delayed_target_tracking_unit" BACKENDS "mp_units_eigen")
  bench("kf_6x2x0_2d_vehicle_position_unit" BACKENDS "mp_units_eigen")
  bench("kf_6x2x0_2d_vehicle_position" BACKENDS "eigen" "eigexed")
  bench("kf_8x4x0_2d_bounding_box" BACKENDS "eigen" "eigexed")
endif()
