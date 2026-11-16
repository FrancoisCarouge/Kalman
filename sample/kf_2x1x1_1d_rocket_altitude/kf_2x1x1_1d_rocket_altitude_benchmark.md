The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | eigen | eigexed |
| :-- | --: | --: |
| `predict` time (ns) | 31.99 ± 3.21 % | 39.72 ± 11.77 % |
| `update` time (ns) | 33.73 ± 1.97 % | 33.85 ± 4.86 % |
| Compile time (s) | 3.16 | 9.85 |
| Flash (B) | 23600 | 23607 |
| RAM (B) | 12 | 12 |
