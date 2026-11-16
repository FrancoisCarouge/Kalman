The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | mp_units_eigen |
| :-- | --: |
| `predict` time (ns) | 11.12 ± 1.97 % |
| `update` time (ns) | 121.11 ± 9.18 % |
| Compile time (s) | 22.46 |
| Flash (B) | 111768 |
| RAM (B) | 60 |
