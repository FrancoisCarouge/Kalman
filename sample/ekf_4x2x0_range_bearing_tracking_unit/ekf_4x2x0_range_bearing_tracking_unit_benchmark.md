The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | mp_units_eigen |
| :-- | --: |
| `predict` time (ns) | 24.80 ± 1.79 % |
| `update` time (ns) | 121.20 ± 13.78 % |
| Compile time (s) | 14.05 |
| Flash (B) | 114804 |
| RAM (B) | 60 |
