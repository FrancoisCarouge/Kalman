The [benchmark](https://github.com/FrancoisCarouge/Kalman/tree/master/benchmark#samples) of the sample, measured with Clang 21.1.8 `Release` on 24 core Intel(R) Core(TM) Ultra 9 275HX:

| Metric | eigen |
| :-- | --: |
| `predict` time (ns) | 42.18 ± 2.07 % |
| `update` time (ns) | 294.13 ± 8.23 % |
| Compile time (s) | 6.06 |
| Flash (B) | 130328 |
| RAM (B) | 60 |
