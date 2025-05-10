#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.1
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

# sample("ekf_4x1x0_soaring" BACKENDS "eigen" "eigexed" "naive" "naixed")
sample("kf_1x1x0_building_height_unit")
sample("kf_1x1x0_building_height")
sample("kf_1x1x0_liquid_temperature_unit")
sample("kf_1x1x0_liquid_temperature")
sample("kf_1x1x1_dog_position")
sample("kf_1x1x1_dog_position_unit")
sample("kf_2x1x1_rocket_altitude" BACKENDS "eigen" "eigexed" "naive" "naixed")
sample("kf_6x2x0_vehicle_location_unit" BACKENDS "quantity")
sample("kf_6x2x0_vehicle_location" BACKENDS "eigen" "eigexed")
sample("kf_8x4x0_deep_sort_bounding_box" BACKENDS "eigen" "eigexed")
