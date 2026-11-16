The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | mp_units_eigen |
| :-- | --: |
| `predict` time (ns) | 9.66 ± 2.25 % |
| `update` time (ns) | 135.15 ± 1.81 % |
| Compile time (s) | 27.27 |
| Flash (B) | 69026 |
| RAM (B) | 60 |
