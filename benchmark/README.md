# Benchmarks

## Embedded Footprint

The [embedded footprint](https://francoiscarouge.github.io/Kalman/footprint.xhtml) measures the flash, RAM, and worst-case stack sizes of one filter step, one prediction and one update, for the [filter configurations](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark/footprint), per update form policy and optimization level, on a bare-metal Arm Cortex-M4 target.

Measure on Linux with the Arm GNU Toolchain:

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/kalman"
cmake -S "kalman" -B "build" --toolchain "cmake/toolchain_arm_cortex_m4.cmake"
cmake --build "build" --parallel
ctest --test-dir "build" --output-on-failure --parallel
```

The tests write the measurements to `build/footprint/index.html`, and fail on any size change, larger or smaller, from its `footprint()` line in [footprint/CMakeLists.txt](https://github.com/FrancoisCarouge/Kalman/blob/master/benchmark/footprint/CMakeLists.txt). The criteria hold for the toolchain of the continuous integration runner, `gcc-arm-none-eabi` 14.2 on Ubuntu 26.04. To accept a change, update the line, with its reason in the commit message.
