# Benchmarks

Build and run the benchmarks on all platforms:

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/kalman"
cmake -S "kalman" -B "build"
cmake --build "build" --config "Release" --parallel
ctest --test-dir "build" --build-config "Release" --tests-regex "kalman_benchmark"
```

Plot the results on Linux:

```shell
./kalman/benchmark/script/plot.sh
```

# Results

Run on Microsoft Windows 10 on native x64 with Visual Studio 2022 compiler 19.33 in release mode.

![Eigen Update](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/eigen_update.svg)
![Eigen Predict](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/eigen_predict.svg)
![Float](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/float.svg)
![Float 1x1x0](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/float_1x1x0.svg)
![Float 1x1x1](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/float_1x1x1.svg)
![Baseline](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/baseline.svg)
![Update Float 1x1x0](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/update_1x1x0.svg)
![Update Float 1x1x1](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/update_1x1x1.svg)
![Predict Float 1x1x0](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/predict_1x1x0.svg)
![Predict Float 1x1x1](https://raw.githubusercontent.com/FrancoisCarouge/Kalman/master/benchmark/image/predict_1x1x1.svg)
