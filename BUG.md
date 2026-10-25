# Defects

Defects found while reviewing and expanding the unit tests. This file records analysis only: what is wrong, where, how it shows, and what it affects. It does not propose changes.

- Reviewed revision: `6ad1d91f1`.
- Each defect was reproduced with `clang++-21 -std=c++26` and `g++-14 -std=c++26` against the Eigen backend or built-in arithmetic types.
- "Filter" names the internal struct deduced by the factory, as listed in `AGENTS.md`.
- Line numbers refer to the reviewed revision.
- Defect numbers are stable across reviews: a resolved or withdrawn defect leaves a gap, and a new defect takes the next number.

## 13. The update copies the identity onto the stack

- Revision: `9facffd80`, since `d80652ace`. Every update form, for example `x_z_p_q_r_hh_f_us_ps.hpp` line 93, declares `const auto i{make_one<evaluate<product<gain, output_model>>>()};`: `make_one()` returns the constant identity by value, so each call copies it into the update's frame, where the filters used to read a static member.
- Shows as: the update's stack frame grows by one gain-by-output-model matrix. Cortex-M7, Arm GCC 15.2 `-O2`, a filter of N states and one output on a fixed-size backend whose identity is a constant expression: 3 double states 1976 to 2048 bytes, 24 double states 56328 to 60936 (4608 bytes, one 24x24 double matrix), 24 float 28192 to 30496.
- Affects: embedded users with per-function stack limits. ArduPilot's EKFGSF update went from 1416 to 1488 bytes of its 1500 limit; PX4's EKFGSF update from 760 to 800 bytes, its landing target from 432 to 448. No test or CI job measures stack, so the change passed unnoticed.

Resolved and removed: 1 (`89cb86053`), 2 (`f3989173b`), 3 (`6b54da0bc`), 4 (`4fded89ea`, a declaration mixing arithmetic types is now rejected), 5 (`f897f39c8`), 7 (`fde518be7`, an lvalue operand is now rejected at the call site), 8 (`59c6e0d19`), 9 (`2e80e5f49`, narrowing element values of vector characteristics are now rejected), 10 (`24a2f52cc`, the named values are constant-initialized or built at the point of use), 11 (`6f3031910`, constant-evaluable factories), 12 (`f8d89c95c`, the getters return a constant reference from an lvalue filter and a value from an rvalue one). Withdrawn: 6, the rejection of narrowing values is intended.
