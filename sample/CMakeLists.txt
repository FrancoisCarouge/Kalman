#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.4.0
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

foreach(SAMPLE "kf_1x1x0_building_height.cpp" "kf_1x1x0_liquid_temperature.cpp"
               "kf_1x1x1_dog_position.cpp")
  get_filename_component(NAME ${SAMPLE} NAME_WE)
  add_executable(kalman_sample_${NAME}_driver ${SAMPLE})
  target_link_libraries(kalman_sample_${NAME}_driver PRIVATE kalman kalman_main
                                                             kalman_options)
  separate_arguments(TEST_COMMAND UNIX_COMMAND $ENV{COMMAND})
  add_test(NAME kalman_sample_${NAME}
           COMMAND ${TEST_COMMAND} $<TARGET_FILE:kalman_sample_${NAME}_driver>)
endforeach()

foreach(BACKEND IN ITEMS "eigen" "naive")
  foreach(SAMPLE "ekf_4x1x0_soaring.cpp" "kf_2x1x1_rocket_altitude.cpp")
    get_filename_component(NAME ${SAMPLE} NAME_WE)
    add_executable(kalman_sample_${BACKEND}_${NAME}_driver ${SAMPLE})
    target_link_libraries(
      kalman_sample_${BACKEND}_${NAME}_driver
      PRIVATE kalman kalman_main kalman_linalg_${BACKEND} kalman_options)
    separate_arguments(TEST_COMMAND UNIX_COMMAND $ENV{COMMAND})
    add_test(NAME kalman_sample_${BACKEND}_${NAME}
             COMMAND ${TEST_COMMAND}
                     $<TARGET_FILE:kalman_sample_${BACKEND}_${NAME}_driver>)
  endforeach()
endforeach()

foreach(BACKEND IN ITEMS "eigen")
  foreach(SAMPLE "kf_6x2x0_vehicle_location.cpp"
                 "kf_8x4x0_deep_sort_bounding_box.cpp")
    get_filename_component(NAME ${SAMPLE} NAME_WE)
    add_executable(kalman_sample_${BACKEND}_${NAME}_driver ${SAMPLE})
    target_link_libraries(
      kalman_sample_${BACKEND}_${NAME}_driver
      PRIVATE kalman kalman_main kalman_linalg_${BACKEND} kalman_options)
    separate_arguments(TEST_COMMAND UNIX_COMMAND $ENV{COMMAND})
    add_test(NAME kalman_sample_${BACKEND}_${NAME}
             COMMAND ${TEST_COMMAND}
                     $<TARGET_FILE:kalman_sample_${BACKEND}_${NAME}_driver>)
  endforeach()
endforeach()
