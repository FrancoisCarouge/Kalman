The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | mp_units_eigen |
| :-- | --: |
| `predict` time (ns) | 23.62 ± 2.47 % |
| `update` time (ns) | 33.09 ± 1.34 % |
| Compile time (s) | 12.65 |
| Flash (B) | 7670 |
| RAM (B) | 20 |
