# Installation

Requires CMake ≥ 4.3 and a C++26 compiler (Clang 20+, GCC 14+, or MSVC `/std:c++latest`).

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

Alternatively, fetch the library directly from your project's CMake configuration:

```cmake
include(FetchContent)

FetchContent_Declare(
  fcarouge-kalman
  GIT_REPOSITORY "https://github.com/FrancoisCarouge/Kalman"
  FIND_PACKAGE_ARGS NAMES fcarouge-kalman)
FetchContent_MakeAvailable(fcarouge-kalman)

target_link_libraries(your_target PRIVATE fcarouge-kalman::kalman)
```

A single, amalgamated `kalman.h` header is also [published](https://francoiscarouge.github.io/Kalman/amalgamate/fcarouge/kalman.h), or generated locally with `cmake --build "build" --target "amalgamate"` into `build/amalgamate/fcarouge/kalman.h`.

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
