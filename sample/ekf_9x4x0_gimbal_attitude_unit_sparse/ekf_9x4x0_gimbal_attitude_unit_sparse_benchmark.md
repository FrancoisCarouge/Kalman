The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | mp_units_sparse |
| :-- | --: |
| `predict` time (ns) | 122.12 ± 4.80 % |
| `update` time (ns) | 783.08 ± 3.82 % |
| Compile time (s) | 10.73 |
| Flash (B) | 73353 |
| RAM (B) | 12 |
