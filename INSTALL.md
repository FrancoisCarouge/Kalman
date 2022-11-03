# Installation

## From Repository Sources

Install and use the library in your projects by cloning the repository, configuring, and installing the project on all platforms:

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/Kalman.git" "kalman"
cmake -S "kalman" -B "build"
cmake --build "build" --parallel
sudo cmake --install "build"
```

# Build & Run

## Tests & Samples

Build and run the tests and samples on all platforms:

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/Kalman.git" "kalman"
cmake -S "kalman" -B "build"
cmake --build "build" --config "Debug" --parallel
ctest --test-dir "build" --build-config "Debug" --tests-regex "kalman_(test|sample)" --output-on-failure --parallel
```

## Benchmarks

See the [Benchmark](benchmark/) section.

## Installation Package

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/Kalman.git" "kalman"
cmake -S "kalman" -B "build"
cmake --build "build" --target "package" --parallel
cmake --build "build" --target "package_source" --parallel
```
