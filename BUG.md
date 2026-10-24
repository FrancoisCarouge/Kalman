# Defects

Defects found while reviewing and expanding the unit tests. This file records analysis only: what is wrong, where, how it shows, and what it affects. It does not propose changes.

- Reviewed revision: `6ad1d91f1`.
- Each defect was reproduced with `clang++-21 -std=c++26` and `g++-14 -std=c++26` against the Eigen backend or built-in arithmetic types.
- "Filter" names the internal struct deduced by the factory, as listed in `AGENTS.md`.
- Line numbers refer to the reviewed revision.
- Defect numbers are stable across reviews: a resolved or withdrawn defect leaves a gap, and a new defect takes the next number.

No open defect.

Resolved and removed: 1 (`89cb86053`), 2 (`f3989173b`), 3 (`6b54da0bc`), 4 (`4fded89ea`, a declaration mixing arithmetic types is now rejected), 5 (`f897f39c8`), 7 (`fde518be7`, an lvalue operand is now rejected at the call site), 8 (`59c6e0d19`), 9 (`2e80e5f49`, narrowing element values of vector characteristics are now rejected), 10 (`24a2f52cc`, the named values are constant-initialized or built at the point of use), 11 (`6f3031910`, constant-evaluable factories), 12 (`f8d89c95c`, the getters return a constant reference from an lvalue filter and a value from an rvalue one). Withdrawn: 6, the rejection of narrowing values is intended.
