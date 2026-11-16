The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | mp_units_eigen |
| :-- | --: |
| `predict` time (ns) | 13.41 ± 0.82 % |
| `update` time (ns) | 52.39 ± 2.70 % |
| Compile time (s) | 12.71 |
| Flash (B) | 3763 |
| RAM (B) | 12 |
