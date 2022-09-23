# Benchmarks

Build and run the benchmarks on all platforms:

```shell
git clone --depth 1 https://github.com/FrancoisCarouge/Kalman.git "kalman"
cmake -S "kalman" -B "build" -G "Ninja Multi-Config"
cmake --build "build" --config "Release" --parallel
ctest --test-dir "build" --tests-regex "kalman_benchmark"
```

Plot the results on Linux:

```shell
./kalman/benchmark/script/plot.sh
```

# Results

Run on Microsoft Windows 10 on native x64 with Visual Studio 2022 compiler 19.33 in release mode.

![Eigen Update](image/eigen_update.svg)
![Eigen Predict](image/eigen_predict.svg)
![Float](image/float.svg)
![Float 1x1x0](image/float_1x1x0.svg)
![Float 1x1x1](image/float_1x1x1.svg)
![Baseline](image/baseline.svg)
![Update Float 1x1x0](image/update_1x1x0.svg)
![Update Float 1x1x1](image/update_1x1x1.svg)
![Predict Float 1x1x0](image/predict_1x1x0.svg)
![Predict Float 1x1x1](image/predict_1x1x1.svg)
