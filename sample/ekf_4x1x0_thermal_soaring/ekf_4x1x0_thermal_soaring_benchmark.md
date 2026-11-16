The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | eigen | eigexed |
| :-- | --: | --: |
| `predict` time (ns) | 18.41 ± 0.44 % | 19.43 ± 3.79 % |
| `update` time (ns) | 32.39 ± 1.05 % | 32.49 ± 1.43 % |
| Compile time (s) | 2.84 | 5.94 |
| Flash (B) | 6396 | 7381 |
| RAM (B) | 12 | 12 |
