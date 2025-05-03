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

include(FetchContent)

FetchContent_Declare(
  google_test
  GIT_REPOSITORY "https://github.com/google/googletest"
  GIT_TAG "main"
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS NAMES GTest)
FetchContent_MakeAvailable(google_test)

FetchContent_Declare(
  google_benchmark
  GIT_REPOSITORY "https://github.com/google/benchmark"
  GIT_TAG "main"
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS NAMES benchmark)
FetchContent_MakeAvailable(google_benchmark)

set(PROCESSOR_AFFINITY TRUE)

foreach(BENCHMARK "baseline.cpp" "predict_1x1x0.cpp" "predict_1x1x1.cpp"
                  "update_1x1x0.cpp" "update_1x1x1.cpp")
  get_filename_component(NAME ${BENCHMARK} NAME_WE)
  add_executable(kalman_benchmark_${NAME}_driver ${BENCHMARK})
  target_include_directories(kalman_benchmark_${NAME}_driver PRIVATE "include")
  target_link_libraries(
    kalman_benchmark_${NAME}_driver PRIVATE benchmark::benchmark
                                            benchmark::benchmark_main kalman)
  separate_arguments(TEST_COMMAND UNIX_COMMAND $ENV{COMMAND})
  add_test(
    NAME kalman_benchmark_${NAME}
    COMMAND ${TEST_COMMAND} $<TARGET_FILE:kalman_benchmark_${NAME}_driver>
            "--benchmark_out=${NAME}.json")
endforeach()

foreach(BACKEND IN ITEMS "eigen" "naive")
  foreach(STATE_SIZE RANGE 1 2)
    foreach(INPUT_SIZE RANGE 1 2)
      configure_file(predict_linalg_x1x.cpp
                     predict_${BACKEND}_${STATE_SIZE}x1x${INPUT_SIZE}.cpp)
      add_executable(
        kalman_benchmark_predict_${BACKEND}_${STATE_SIZE}x1x${INPUT_SIZE}_driver
        predict_${BACKEND}_${STATE_SIZE}x1x${INPUT_SIZE}.cpp)
      target_include_directories(
        kalman_benchmark_predict_${BACKEND}_${STATE_SIZE}x1x${INPUT_SIZE}_driver
        PRIVATE "include")
      target_link_libraries(
        kalman_benchmark_predict_${BACKEND}_${STATE_SIZE}x1x${INPUT_SIZE}_driver
        PRIVATE benchmark::benchmark benchmark::benchmark_main kalman
                kalman_linalg_${BACKEND})
      separate_arguments(TEST_COMMAND UNIX_COMMAND $ENV{COMMAND})
      add_test(
        NAME kalman_benchmark_predict_${BACKEND}_${STATE_SIZE}x1x${INPUT_SIZE}
        COMMAND
          ${TEST_COMMAND}
          $<TARGET_FILE:kalman_benchmark_predict_${BACKEND}_${STATE_SIZE}x1x${INPUT_SIZE}_driver>
          "--benchmark_out=predict_${BACKEND}_${STATE_SIZE}x1x${INPUT_SIZE}.json"
      )
    endforeach()
  endforeach()
endforeach()

foreach(BACKEND IN ITEMS "eigen")
  foreach(STATE_SIZE RANGE 1 2)
    foreach(OUTPUT_SIZE RANGE 1 2)
      configure_file(update_linalg_xx0.cpp
                     update_${BACKEND}_${STATE_SIZE}x${OUTPUT_SIZE}x0.cpp)
      add_executable(
        kalman_benchmark_update_${BACKEND}_${STATE_SIZE}x${OUTPUT_SIZE}x0_driver
        update_${BACKEND}_${STATE_SIZE}x${OUTPUT_SIZE}x0.cpp)
      target_include_directories(
        kalman_benchmark_update_${BACKEND}_${STATE_SIZE}x${OUTPUT_SIZE}x0_driver
        PRIVATE "include")
      target_link_libraries(
        kalman_benchmark_update_${BACKEND}_${STATE_SIZE}x${OUTPUT_SIZE}x0_driver
        PRIVATE benchmark::benchmark benchmark::benchmark_main kalman
                kalman_linalg_${BACKEND})
      separate_arguments(TEST_COMMAND UNIX_COMMAND $ENV{COMMAND})
      add_test(
        NAME kalman_benchmark_update_${BACKEND}_${STATE_SIZE}x${OUTPUT_SIZE}x0
        COMMAND
          ${TEST_COMMAND}
          $<TARGET_FILE:kalman_benchmark_update_${BACKEND}_${STATE_SIZE}x${OUTPUT_SIZE}x0_driver>
          "--benchmark_out=update_${BACKEND}_${STATE_SIZE}x${OUTPUT_SIZE}x0.json"
      )
    endforeach()
  endforeach()
endforeach()
