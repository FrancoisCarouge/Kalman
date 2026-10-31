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

add_subdirectory("ekf_1x2x0_terrain_estimation_unit")
add_subdirectory("ekf_3x1x0_beacon_position_unit")
add_subdirectory("ekf_4x1x0_thermal_soaring")
add_subdirectory("ekf_6x4x0_3d_apollo_position")
add_subdirectory("ekf_9x4x0_gimbal_attitude_unit")
add_subdirectory("kf_1x1x0_building_height_unit")
add_subdirectory("kf_1x1x0_building_height")
add_subdirectory("kf_1x1x0_liquid_temperature_unit")
add_subdirectory("kf_1x1x0_liquid_temperature")
add_subdirectory("kf_1x1x1_1d_dog_position_unit")
add_subdirectory("kf_1x1x1_1d_dog_position")
add_subdirectory("kf_2x1x0_pairs_hedge_ratio_unit")
add_subdirectory("kf_2x1x0_stock_price_unit")
add_subdirectory("kf_2x1x1_1d_rocket_altitude")
add_subdirectory("kf_6x2x0_2d_vehicle_position_unit")
add_subdirectory("kf_6x2x0_2d_vehicle_position")
add_subdirectory("kf_8x4x0_2d_bounding_box")
