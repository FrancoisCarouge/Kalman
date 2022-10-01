#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
Version 0.1.0
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

set(SOURCES "baseline.cpp" "predict_1x1x0.cpp" "predict_1x1x1.cpp"
            "update_1x1x0.cpp" "update_1x1x1.cpp")

foreach(STATE_SIZE RANGE 1 2)
  foreach(OUTPUT_SIZE RANGE 1 2)
    configure_file(eigen_update_xx0.cpp
                   eigen_update_${STATE_SIZE}x${OUTPUT_SIZE}x0.cpp)
    list(APPEND SOURCES eigen_update_${STATE_SIZE}x${OUTPUT_SIZE}x0.cpp)
  endforeach()
  foreach(INPUT_SIZE RANGE 1 2)
    configure_file(eigen_predict_x1x.cpp
                   eigen_predict_${STATE_SIZE}x1x${INPUT_SIZE}.cpp)
    list(APPEND SOURCES eigen_predict_${STATE_SIZE}x1x${INPUT_SIZE}.cpp)
  endforeach()
endforeach()

foreach(BENCHMARK ${SOURCES})
  get_filename_component(NAME ${BENCHMARK} NAME_WE)
  add_executable(kalman_benchmark_${NAME}_driver ${BENCHMARK})
  target_include_directories(kalman_benchmark_${NAME}_driver PRIVATE "."
                                                                     "include")
  target_link_libraries(
    kalman_benchmark_${NAME}_driver
    PRIVATE eigen kalman benchmark::benchmark_main benchmark::benchmark)
  add_test(kalman_benchmark_${NAME} kalman_benchmark_${NAME}_driver
           "--benchmark_out=${NAME}.json")
endforeach()