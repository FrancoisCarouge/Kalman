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

## Samples

The [sample benchmarks](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark/sample) measure each [sample](https://github.com/FrancoisCarouge/Kalman/tree/master/sample), per backend, with [nanobench](https://github.com/martinus/nanobench). The sample compiles unchanged, with its own options, assertions and hardening included: the benchmark declares the `benchmark` filter decorator, like the library's `print`, measuring the first prediction and the first update, and the `kalman` class template of the sample's namespace, the library filter decorated with `benchmark`.

| Metric | Measure |
| --- | --- |
| `predict`, `update` time (ns) | Median time per call ± its median absolute percentage error, the call repeated with the sample's arguments on a copy of the filter. |
| `predict`, `update` instructions | Median instructions per call, where the hardware performance counters are available. |
| Compile time (s) | Wall time to rebuild the sample's object. |
| Flash (B) | Allocated sections of the sample's object, but zero-initialized data: code, read-only data, data. |
| RAM (B) | Writable allocated sections of the sample's object: data, zero-initialized data. |

Flash and RAM need `readelf`, found by CMake with the ELF toolchains. Measure:

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/kalman"
cmake -S "kalman" -B "build" -DCMAKE_BUILD_TYPE="Release" -DBUILD_BENCHMARKING=ON
cmake --build "build" --parallel
ctest --test-dir "build" --tests-regex "_bench$" --output-on-failure
```

With a multi-configuration generator, build with `--config "Release"` and test with `--build-config "Release"`. The tests run serially, and write each sample's `sample/<name>/<name>_benchmark.md`, embedded in its documentation. Without `BUILD_BENCHMARKING`, only `kf_1x1x0_building_height` is measured, into the build tree. A new sample takes its `bench()` line, with its backends, in [sample/CMakeLists.txt](https://github.com/FrancoisCarouge/Kalman/blob/master/benchmark/sample/CMakeLists.txt).
