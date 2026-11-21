The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | mp_units_sparse |
| :-- | --: |
| `predict` time (ns) | 64810.47 ± 3.28 % |
| `update` time (ns) | 900969.00 ± 2.88 % |
| Compile time (s) | 81.05 |
| Flash (B) | 2082994 |
| RAM (B) | 12 |
