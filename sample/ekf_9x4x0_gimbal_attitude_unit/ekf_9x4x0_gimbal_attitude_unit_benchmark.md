The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | mp_units_eigen |
| :-- | --: |
| `predict` time (ns) | 154.64 ± 1.29 % |
| `update` time (ns) | 513.40 ± 3.49 % |
| Compile time (s) | 18.57 |
| Flash (B) | 142919 |
| RAM (B) | 60 |
