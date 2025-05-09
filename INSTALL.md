# Installation

Download and install the [latest release package](https://github.com/FrancoisCarouge/Kalman/releases). Alternatively, you may install and use the library in your projects by cloning the repository, configuring, and installing the project:

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/kalman"
cmake -S "kalman" -B "build"
cmake --build "build" --parallel
sudo cmake --install "build"
```

The standard shared CMake configuration file provides the library target to use in your own target:

```cmake
find_package(fcarouge-kalman)
target_link_libraries(your_target PRIVATE fcarouge-kalman::kalman)
```

In your sources, include the library header and use the filter. See [the samples](https://github.com/FrancoisCarouge/Kalman/tree/master/sample) for more.

```cpp
#include "fcarouge/kalman.hpp"

fcarouge::kalman filter;
```

# Development Build & Run

## Tests & Samples

Build and run the tests and samples:

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/kalman"
cmake -S "kalman" -B "build"
cmake --build "build" --config "Debug" --parallel
ctest --test-dir "build" --build-config "Debug" --output-on-failure --parallel
```

## Benchmarks

See the [Benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark) section.

## Installation Packages

### Linux

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/kalman"
cmake -S "kalman" -B "build"
cmake --build "build" --target "package" --parallel
cmake --build "build" --target "package_source" --parallel
```

### Windows

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/kalman"
cmake -S "kalman" -B "build"
cmake --build "build" --target "package" --parallel --config "Release"
cmake --build "build" --target "package_source" --parallel --config "Release"
```