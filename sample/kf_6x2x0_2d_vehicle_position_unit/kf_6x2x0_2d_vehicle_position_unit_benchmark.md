The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | mp_units_eigen |
| :-- | --: |
| `predict` time (ns) | 112.04 ± 3.40 % |
| `update` time (ns) | 249.20 ± 3.98 % |
| Compile time (s) | 20.00 |
| Flash (B) | 123090 |
| RAM (B) | 60 |
