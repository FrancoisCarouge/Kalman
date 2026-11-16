The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | eigen | eigexed |
| :-- | --: | --: |
| `predict` time (ns) | 71.60 ± 4.50 % | 197.33 ± 11.51 % |
| `update` time (ns) | 260.37 ± 1.21 % | 548.95 ± 9.81 % |
| Compile time (s) | 9.10 | 16.46 |
| Flash (B) | 132728 | 133790 |
| RAM (B) | 60 | 60 |
