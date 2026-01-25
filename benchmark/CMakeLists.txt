#[[ __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
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

FetchContent_Declare(
  nanobench
  GIT_REPOSITORY "https://github.com/martinus/nanobench.git"
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS NAMES nanobench)
FetchContent_MakeAvailable(nanobench)

# set(SIZES 1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25;26;27;28;29;30;31;32)
set(SIZES 1;2)

foreach(STATE_SIZE IN ITEMS ${SIZES})
  foreach(IO_SIZE IN ITEMS ${SIZES})
    configure_file("benchmark.cpp"
                   "benchmark_${STATE_SIZE}x${IO_SIZE}x${IO_SIZE}.cpp")
    add_executable(kalman_benchmark_${STATE_SIZE}x${IO_SIZE}x${IO_SIZE}_driver
                   "benchmark_${STATE_SIZE}x${IO_SIZE}x${IO_SIZE}.cpp")
    target_link_libraries(
      kalman_benchmark_${STATE_SIZE}x${IO_SIZE}x${IO_SIZE}_driver
      PRIVATE kalman kalman_main nanobench::nanobench kalman_linalg_eigen)
  endforeach()
endforeach()

configure_file("plot.cpp" "plot.cpp")
add_executable(kalman_benchmark_plot_driver "plot.cpp")
target_link_libraries(kalman_benchmark_plot_driver PRIVATE kalman_main)
add_test(
  NAME kalman_benchmark_plot_driver
  COMMAND kalman_benchmark_plot_driver "${CMAKE_SOURCE_DIR}/benchmark"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark/$<CONFIG>")
