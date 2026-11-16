The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | eigen | eigexed |
| :-- | --: | --: |
| `predict` time (ns) | 96.96 ± 12.04 % | 73.30 ± 4.75 % |
| `update` time (ns) | 235.67 ± 9.13 % | 215.20 ± 6.24 % |
| Compile time (s) | 8.50 | 15.22 |
| Flash (B) | 148435 | 156659 |
| RAM (B) | 60 | 60 |
