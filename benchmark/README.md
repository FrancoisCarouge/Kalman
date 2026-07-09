# Benchmarks

Build and run the benchmarks on all platforms:

```shell
git clone --depth 1 "https://github.com/FrancoisCarouge/kalman"
Remove-Item -Path build -Force -Recurse
cmake -S "kalman" -B "build" -G "Visual Studio 17 2022"
cmake --build "build" --config "Release" --parallel 10
ctest --test-dir "build" --build-config "Release" --tests-regex "kalman_benchmarks_driver" --verbose --parallel 1 --repeat-until-fail 5
```

# Results

Run on Microsoft Windows 11 on native x64 with Visual Studio 2022 compiler 19.44 in release mode.

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
